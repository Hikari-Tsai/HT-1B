"""S6/TFiLM latent trajectories constrained by the existing reduced circuit.

Network conditioning uses normalized filename labels; circuit equations use
the separate approximate physical potentiometer positions from the manifest.
The two encodings must never be substituted for each other.
"""
from types import SimpleNamespace

import torch
from torch import nn

from ht1b.config import Circuit
from ht1b.physicsnemo_model import LearnedCircuit, TorchOps, circuit_residual_terms
from .models import AudioModel
from .layers import detach_state


class S6PINN(AudioModel):
    def __init__(self, **options):
        super().__init__(**options)
        if self.architecture != 's6_tfilm':
            raise ValueError('This experiment supports S6/TFiLM only')
        self.output = nn.Linear(self.output.in_features, 3)
        # Start with a smooth constant trajectory instead of samplewise noise
        # multiplied by 48 kHz in a stiff ODE residual. All heads remain trainable.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.physical = LearnedCircuit(Circuit())
        self.register_buffer('physical_prior', torch.stack(
            [p.detach().clone() for p in self.physical.raw.values()]))

    def trajectory(self, dry, controls, state=None):
        previous = {} if state is None else state
        history = previous.get('history')
        if history is None:
            history = dry.new_zeros(dry.shape[0], self.context-1)
        audio = torch.cat((history, dry), dim=1)
        windows = audio.unfold(1, self.context, 1)
        z = self.input(windows)
        spectrum = self.spectrum(torch.fft.rfft(windows, n=2*self.context).abs()/self.context)
        z, first = self.first(z, previous.get('first'))
        z, condition = self.condition(z, spectrum, controls, previous.get('condition'))
        z, second = self.second(z, previous.get('second'))
        raw = self.output(z).double()
        states = torch.cat((30*torch.sigmoid(raw[..., :1]-6),
                            torch.sigmoid(raw[..., 1:]-3)), dim=-1)
        recurrent = dict(history=audio[:, -(self.context-1):] if self.context > 1 else audio[:, :0],
                         first=first, condition=condition, second=second)
        return states, recurrent

    def forward(self, dry, controls, pots, state=None, previous_physical=None):
        states, recurrent = self.trajectory(dry, controls, state)
        if previous_physical is None:
            raise ValueError('Supply the preceding physical state from Dry warmup')
        prior = torch.cat((previous_physical[:, None], states[:, :-1]), dim=1)
        derivatives = (states-prior)*48000
        # pots: threshold, ratio, attack, release, makeup_db; manual mode.
        c = SimpleNamespace(**{key: pots[:, i:i+1].double() for i, key in enumerate(
            ('threshold','ratio','attack','release','makeup_db'))}, mode='manual')
        params = self.physical.values()
        residual, wet = circuit_residual_terms(tuple(states.unbind(-1)),
            tuple(derivatives.unbind(-1)), dry.double()*params.input_volts_per_fs,
            params, c, TorchOps)
        return wet, residual, recurrent, states[:, -1]

    def predict_window(self, dry, controls, pots, warmup=4096, chunk=1024):
        if warmup < 1 or dry.shape[1] <= warmup:
            raise ValueError('A nonempty Dry-only warmup and scored tail are required')
        state = None
        with torch.no_grad():
            for offset in range(0, warmup, chunk):
                latent, state = self.trajectory(dry[:, offset:min(offset+chunk, warmup)], controls, state)
                state = detach_state(state)
            previous = latent[:, -1].detach()
        wet, residual, _, _ = self(dry[:, warmup:], controls, pots, state, previous)
        return wet, residual

    def prior_loss(self):
        return (torch.stack(list(self.physical.raw.values()))-self.physical_prior).square().mean()
