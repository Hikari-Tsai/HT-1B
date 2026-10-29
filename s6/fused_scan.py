"""Fused selective recurrence for TRAINING and inference, with an exact backward.

Same recurrence as SelectiveSSM.forward, evaluated in one Triton program per
(batch, channel) instead of a Python loop over samples:

    s[t] = exp(dt[t]*a)*s[t-1] + dt[t]*b[t]*u[t]
    y[t] = sum_n s[t,n]*c[t,n] + skip*u[t]

The backward pass runs the adjoint recurrence in reverse time from the stored
forward states. Checkpoint parameters are unchanged: accelerate_training only
swaps the forward method. Non-CUDA or non-float32 inputs fall back to the
reference loop, so CPU behaviour is identical to the original code.
"""
import types

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from .layers import SelectiveSSM


@triton.jit
def _forward(U, Bm, Cm, DT, A, D, S0, Y, SEND, ST, T,
             H: tl.constexpr, N: tl.constexpr, K: tl.constexpr, SAVE: tl.constexpr):
    bh = tl.program_id(0)
    batch = bh // H
    h = bh % H
    ns = tl.arange(0, K)
    m = ns < N
    a = tl.load(A + h*N + ns, m, 0.)
    skip = tl.load(D + h)
    s = tl.load(S0 + bh*N + ns, m, 0.)
    for t in range(T):
        row = batch*T + t
        u = tl.load(U + row*H + h)
        dt = tl.load(DT + row*H + h)
        b = tl.load(Bm + row*N + ns, m, 0.)
        c = tl.load(Cm + row*N + ns, m, 0.)
        # Same operation order as the reference: exp(dt*a)*s + (dt*b)*u.
        s = tl.exp(dt*a)*s + dt*b*u
        tl.store(Y + row*H + h, tl.sum(s*c, 0) + skip*u)
        if SAVE:
            tl.store(ST + (bh*T + t)*N + ns, s, m)
    tl.store(SEND + bh*N + ns, s, m)


@triton.jit
def _backward(U, Bm, Cm, DT, A, D, S0, ST, SEND, DY, DSEND,
              DU, DDT, DBP, DCP, DAP, DDP, DS0, T,
              H: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    bh = tl.program_id(0)
    batch = bh // H
    h = bh % H
    ns = tl.arange(0, K)
    m = ns < N
    a = tl.load(A + h*N + ns, m, 0.)
    skip = tl.load(D + h)
    lam = tl.load(DSEND + bh*N + ns, m, 0.)   # dL/ds[T-1] from the returned state
    s_cur = tl.load(SEND + bh*N + ns, m, 0.)  # s[T-1]
    da = tl.zeros([K], dtype=tl.float32)
    dd = 0.
    for i in range(T):
        t = T - 1 - i
        row = batch*T + t
        u = tl.load(U + row*H + h)
        dt = tl.load(DT + row*H + h)
        b = tl.load(Bm + row*N + ns, m, 0.)
        c = tl.load(Cm + row*N + ns, m, 0.)
        g = tl.load(DY + row*H + h)
        prev = (tl.load(ST + (bh*T + t - 1)*N + ns, m & (t > 0), 0.) +
                tl.load(S0 + bh*N + ns, m & (t == 0), 0.))
        lam = lam + g*c
        decay = tl.exp(dt*a)
        tl.store(DCP + (row*H + h)*N + ns, g*s_cur, m)
        tl.store(DBP + (row*H + h)*N + ns, lam*dt*u, m)
        tl.store(DU + row*H + h, g*skip + tl.sum(lam*dt*b, 0))
        tl.store(DDT + row*H + h, tl.sum(lam*(a*decay*prev + b*u), 0))
        da += lam*dt*decay*prev
        dd += g*u
        lam = lam*decay
        s_cur = prev
    tl.store(DS0 + bh*N + ns, lam, m)
    tl.store(DAP + bh*N + ns, da, m)
    tl.store(DDP + bh, dd)


class SelectiveScan(torch.autograd.Function):
    """u,dt:[B,T,H]  b,c:[B,T,N]  a:[H,N]  skip:[H]  s0:[B,H,N] -> y:[B,T,H], s_end:[B,H,N]."""

    @staticmethod
    def forward(ctx, u, b, c, dt, a, skip, s0, save):
        batch, length, width = u.shape
        n = a.shape[1]
        k = triton.next_power_of_2(n)
        y = torch.empty_like(u)
        end = torch.empty_like(s0)
        states = u.new_empty(batch, width, length, n) if save else u.new_empty(1)
        _forward[(batch*width,)](u, b, c, dt, a, skip, s0, y, end, states, length,
                                 width, n, k, save, num_warps=1, enable_fp_fusion=False)
        if save:
            ctx.save_for_backward(u, b, c, dt, a, skip, s0, states, end)
        ctx.saved = save
        return y, end

    @staticmethod
    def backward(ctx, dy, dend):
        if not ctx.saved:
            raise RuntimeError('SelectiveScan forward ran without saving states for backward')
        u, b, c, dt, a, skip, s0, states, end = ctx.saved_tensors
        batch, length, width = u.shape
        n = a.shape[1]
        k = triton.next_power_of_2(n)
        dy = dy.contiguous() if dy is not None else torch.zeros_like(u)
        dend = dend.contiguous() if dend is not None else torch.zeros_like(end)
        du, ddt = torch.empty_like(u), torch.empty_like(dt)
        db_parts = u.new_empty(batch, length, width, n)
        dc_parts = u.new_empty(batch, length, width, n)
        da_parts = u.new_empty(batch, width, n)
        dd_parts = u.new_empty(batch, width)
        ds0 = torch.empty_like(s0)
        _backward[(batch*width,)](u, b, c, dt, a, skip, s0, states, end, dy, dend,
                                  du, ddt, db_parts, dc_parts, da_parts, dd_parts, ds0,
                                  length, width, n, k, num_warps=1, enable_fp_fusion=False)
        # b and c are shared by all channels: reduce the per-channel parts.
        return (du, db_parts.sum(2), dc_parts.sum(2), ddt, da_parts.sum(0),
                dd_parts.sum(0), ds0, None)


def fused_forward(self, u, state=None):
    """Drop-in SelectiveSSM.forward; falls back to the reference loop off-GPU."""
    if u.device.type != 'cuda' or u.dtype != torch.float32:
        return SelectiveSSM.forward(self, u, state)
    b, c = self.bc(u).chunk(2, dim=-1)
    dt = F.softplus(self.dt(u))
    a = -self.log_a.exp()
    if state is None:
        state = u.new_zeros(u.shape[0], self.width, self.state_dim)
    inputs = (u, b, c, dt, a, self.skip, state)
    save = torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
    return SelectiveScan.apply(*(x.contiguous() for x in inputs), save)


def accelerate_training(model):
    """Swap every SelectiveSSM forward for the fused version; parameters untouched."""
    for layer in model.modules():
        if isinstance(layer, SelectiveSSM):
            layer.forward = types.MethodType(fused_forward, layer)
    return model
