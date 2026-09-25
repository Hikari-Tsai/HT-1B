"""Portable SSM reference layers, explicit states, and temporal modulation.

No fused CUDA kernels. S4D uses exact ZOH diagonal dynamics and FFT convolution.
SelectiveSSM uses Mamba-1's input-dependent B,C,delta and delta*B*u write term.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def detach_state(state):
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_state(v) for v in state)
    if isinstance(state, list):
        return [detach_state(v) for v in state]
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    return state


class DiagonalSSM(nn.Module):
    """Complex diagonal S4D, one conjugate half stored; [batch,time,channels]."""
    def __init__(self, width, state_dim):
        super().__init__()
        self.width, self.state_dim = width, state_dim
        self.log_decay = nn.Parameter(torch.full((width, state_dim), math.log(.5)))
        self.frequency = nn.Parameter(math.pi*torch.arange(state_dim).float().repeat(width, 1))
        self.log_dt = nn.Parameter(torch.linspace(-9., -3., width))
        self.readout = nn.Parameter(torch.randn(width, state_dim, 2)/math.sqrt(state_dim))
        self.skip = nn.Parameter(torch.ones(width))

    def forward(self, u, state=None):
        batch, length, _ = u.shape
        a = torch.complex(-self.log_decay.exp(), self.frequency)
        step = self.log_dt.exp()[:, None]*a
        # B=1, exact zero-order-hold discretization; a cannot be zero.
        b_bar = torch.expm1(step)/a
        c = torch.view_as_complex(self.readout.contiguous())
        k = torch.arange(length+1, device=u.device, dtype=u.dtype)
        powers = torch.exp(step[..., None]*k)
        kernel = 2*torch.einsum('hn,hn,hnt->ht', c, b_bar, powers[..., :-1]).real
        source = u.transpose(1, 2)
        nfft = 2*length
        result = torch.fft.irfft(torch.fft.rfft(source, n=nfft)*
                                torch.fft.rfft(kernel, n=nfft)[None], n=nfft)[..., :length]
        if state is None:
            state = torch.zeros(batch, self.width, self.state_dim, device=u.device, dtype=a.dtype)
        result = result + 2*torch.einsum('bhn,hn,hnt->bht', state, c, powers[..., 1:]).real
        next_state = powers[..., -1][None]*state + b_bar[None]*torch.einsum(
            'bht,hnt->bhn', source.flip(-1).to(a.dtype), powers[..., :-1])
        result = result + self.skip[None, :, None]*source
        return result.transpose(1, 2), next_state


class TemporalFiLM(nn.Module):
    """NablaFX-style block maxpool -> LSTM -> feature-wise affine transform.

Requires block-aligned chunks. Its current-block pooling requires buffering
film_block samples; this is not a sample-causal layer with zero lookahead.
"""
    def __init__(self, width, block_size, controls=4):
        super().__init__()
        self.width, self.block_size = width, block_size
        self.lstm = nn.LSTM(width+controls, 2*width, batch_first=True)

    def forward(self, x, controls, state=None):
        batch, length, width = x.shape
        if length % self.block_size:
            raise ValueError('TFiLM input length must be a multiple of film_block')
        blocks = x.reshape(batch, -1, self.block_size, width)
        pooled = blocks.amax(dim=2)
        params = controls[:, None, :].expand(-1, pooled.shape[1], -1)
        modulation, state = self.lstm(torch.cat((pooled, params), -1), state)
        gamma, beta = modulation.chunk(2, dim=-1)
        return ((1+gamma[:, :, None])*blocks + beta[:, :, None]).reshape_as(x), state


class S4FiLMBlock(nn.Module):
    def __init__(self, width, state_dim, film_block):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.ssm = DiagonalSSM(width, state_dim)
        self.film = TemporalFiLM(width, film_block)

    def forward(self, x, controls, state=None):
        ssm_state, film_state = (None, None) if state is None else state
        z, ssm_state = self.ssm(torch.tanh(self.linear(x)), ssm_state)
        z, film_state = self.film(z, controls, film_state)
        return torch.tanh(z), (ssm_state, film_state)


class SelectiveSSM(nn.Module):
    """Mamba-1-style diagonal selective recurrence (Python reference scan).

Raw A and D are learned but input-independent. B,C,delta depend on each u[n].
The unbounded record history is summarized in a fixed [B,H,N] state.
"""
    def __init__(self, width, state_dim):
        super().__init__()
        self.width, self.state_dim = width, state_dim
        self.log_a = nn.Parameter(torch.arange(1, state_dim+1).float().log().repeat(width, 1))
        self.bc = nn.Linear(width, 2*state_dim, bias=False)
        self.dt = nn.Linear(width, width)
        with torch.no_grad():
            self.dt.bias.copy_(torch.log(torch.expm1(torch.logspace(-4, -1, width))))
        self.skip = nn.Parameter(torch.ones(width))

    def forward(self, u, state=None):
        batch, length, _ = u.shape
        b, c = self.bc(u).chunk(2, dim=-1)
        dt = F.softplus(self.dt(u))
        a = -self.log_a.exp()
        if state is None:
            state = u.new_zeros(batch, self.width, self.state_dim)
        outputs = []
        for n in range(length):
            delta = dt[:, n, :, None]
            state = torch.exp(delta*a)*state + delta*b[:, n, None, :]*u[:, n, :, None]
            outputs.append((state*c[:, n, None, :]).sum(-1) + self.skip*u[:, n])
        return torch.stack(outputs, dim=1), state


class S6Block(nn.Module):
    """Projection -> causal depthwise Conv/SiLU/SSM, multiplied by a SiLU gate."""
    def __init__(self, width, state_dim, conv_kernel, expansion=2):
        super().__init__()
        self.inner, self.kernel = width*expansion, conv_kernel
        self.proj = nn.Linear(width, 2*self.inner)
        self.conv = nn.Conv1d(self.inner, self.inner, conv_kernel, groups=self.inner)
        self.ssm = SelectiveSSM(self.inner, state_dim)
        self.out = nn.Linear(self.inner, width)

    def forward(self, x, state=None):
        conv_state, ssm_state = (None, None) if state is None else state
        u, gate = self.proj(x).chunk(2, -1)
        u = u.transpose(1, 2)
        if conv_state is None:
            conv_state = u.new_zeros(u.shape[0], self.inner, self.kernel-1)
        extended = torch.cat((conv_state, u), dim=-1)
        next_conv = extended[..., -(self.kernel-1):] if self.kernel > 1 else extended[..., :0]
        u = F.silu(self.conv(extended).transpose(1, 2))
        z, ssm_state = self.ssm(u, ssm_state)
        return F.gelu(self.out(z*F.silu(gate))), (next_conv, ssm_state)


class SoftsignGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, 2*width)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, -1)
        return a*F.softsign(b)


class CompressorConditioning(nn.Module):
    """Threshold/ratio+spectra -> FiLM; attack/release+spectra -> GRU TFiLM."""
    def __init__(self, width, spectrum_dim):
        super().__init__()
        self.static = nn.Linear(spectrum_dim+2, 2*width)
        self.temporal = nn.GRU(spectrum_dim+2, 2*width, batch_first=True)
        self.glu1, self.glu2 = SoftsignGLU(width), SoftsignGLU(width)

    def forward(self, x, spectrum, controls, state=None):
        p = controls[:, None].expand(-1, x.shape[1], -1)
        gamma, beta = self.static(torch.cat((spectrum, p[..., :2]), -1)).chunk(2, -1)
        x = self.glu1((1+gamma)*x + beta)
        dynamic, state = self.temporal(torch.cat((spectrum, p[..., 2:]), -1), state)
        gamma, beta = dynamic.chunk(2, -1)
        return self.glu2((1+gamma)*x + beta), state
