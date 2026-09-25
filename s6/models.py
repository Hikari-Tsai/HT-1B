"""PhysicsNeMo 2.2.2 models for the two discussed CL-1B sequence topologies.

These are documented reference implementations, not author checkpoint ports.
All state is explicit and excluded from the model's learned parameter checkpoint.
"""
import torch
from torch import nn
from physicsnemo.core import Module, ModelMetaData

from .layers import S4FiLMBlock, S6Block, CompressorConditioning


class AudioModel(Module):
    def __init__(self, architecture='s4_tfilm', width=16, state_dim=32, blocks=8,
                 film_block=128, context=64, spectrum_dim=8, conv_kernel=4,
                 expansion=2):
        super().__init__(meta=ModelMetaData(auto_grad=True,
                                          amp=False, jit=False, cuda_graphs=False))
        if architecture not in ('s4_tfilm', 's6_tfilm'):
            raise ValueError('architecture must be s4_tfilm or s6_tfilm')
        for name, value in dict(width=width, state_dim=state_dim, blocks=blocks,
                                film_block=film_block, context=context,
                                spectrum_dim=spectrum_dim, conv_kernel=conv_kernel,
                                expansion=expansion).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if architecture == 's6_tfilm' and blocks != 2:
            raise ValueError('s6_tfilm uses exactly two S6 blocks around the conditioning stage')
        self.architecture, self.context = architecture, context
        self.required_multiple = film_block if architecture == 's4_tfilm' else 1
        if architecture == 's4_tfilm':
            self.input = nn.Linear(1, width)
            self.blocks = nn.ModuleList([S4FiLMBlock(width, state_dim, film_block) for _ in range(blocks)])
        else:
            self.input = nn.Linear(context, width)
            # n_fft=2*context: 64 history samples -> 128-point FFT by default.
            self.spectrum = nn.Linear(context+1, spectrum_dim)
            self.first = S6Block(width, state_dim, conv_kernel, expansion)
            self.condition = CompressorConditioning(width, spectrum_dim)
            self.second = S6Block(width, state_dim, conv_kernel, expansion)
        self.output = nn.Linear(width, 1)
        if architecture == 's6_tfilm':
            nn.init.normal_(self.output.weight, std=.02)
            nn.init.zeros_(self.output.bias)

    def forward(self, dry, controls, state=None, gain_db=None):
        """[B,T] dry + [B,4] normalized knobs -> ([B,T] wet, next_state).

        Controls order: threshold, ratio, attack, release. Hold controls constant
        within a chunk. gain_db is an optional [B] external makeup multiplier.
        For S4-TFiLM, callers buffer multiples of required_multiple; pad only the
        final record chunk and discard its final state. S6 accepts arbitrary T.
        """
        if dry.ndim != 2 or dry.shape[1] == 0 or controls.shape != (dry.shape[0], 4):
            raise ValueError('Expected dry [batch,time] and controls [batch,4]')
        if dry.shape[1] % self.required_multiple:
            raise ValueError('Input length must be a multiple of film_block')
        if self.architecture == 's4_tfilm':
            states = [None]*len(self.blocks) if state is None else state
            if len(states) != len(self.blocks):
                raise ValueError('Wrong number of S4 states')
            z = self.input(dry[..., None])
            next_state = []
            for block, previous in zip(self.blocks, states):
                z, current = block(z, controls, previous)
                next_state.append(current)
            wet = torch.tanh(self.output(z).squeeze(-1))
        else:
            previous = {} if state is None else state
            history = previous.get('history')
            if history is None:
                history = dry.new_zeros(dry.shape[0], self.context-1)
            audio = torch.cat((history, dry), dim=1)
            windows = audio.unfold(1, self.context, 1)
            z = self.input(windows)
            magnitude = torch.fft.rfft(windows, n=2*self.context).abs()/self.context
            spectrum = self.spectrum(magnitude)
            z, first = self.first(z, previous.get('first'))
            z, condition = self.condition(z, spectrum, controls, previous.get('condition'))
            z, second = self.second(z, previous.get('second'))
            # Linear gain with unity offset, NOT sigmoid and never wet/dry targets.
            gain = 1+self.output(z).squeeze(-1)
            wet = dry*gain
            next_state = dict(history=audio[:, -(self.context-1):] if self.context > 1 else audio[:, :0],
                              first=first, condition=condition, second=second)
        if gain_db is not None:
            if gain_db.shape != (dry.shape[0],):
                raise ValueError('gain_db must have shape [batch]')
            wet = wet*torch.pow(10., gain_db[:, None]/20)
        return wet, next_state
