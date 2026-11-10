"""S6-DiffCircuit: S6/TFiLM corrections around a circuit-integrated state (hard constraint).

e/f/s come from s6.circuit_scan.CircuitScan, never from the network. The
network supplies bounded corrections only (design section 3):

  detector  0.5·tanh  log gain on the detector       (about ±4 dB)
  attack    1.0·tanh  log factor on the charge resistance
  release   1.0·tanh  log factor on the discharge resistance
  output    0.25·tanh log gain on the circuit output (about ±2 dB)
  residual  0.05·tanh additive output residual (full scale)

corrections='audio' feeds the network zero knob labels, so every knob acts
only through the circuit; 'knobs' passes the labels (ablation M-HYB-K);
'none' is the circuit alone with learnable parameters (M-CIRC). The output
layer starts at zero, so training starts from the plain circuit.
"""
import torch
from torch import nn

from .circuit_scan import CircuitScan
from .models import AudioModel

ARCHITECTURES = {'s6_tfilm_circuit': 'audio', 's6_tfilm_circuit_knobs': 'knobs', 's6_tfilm_circuit_none': 'none'}
DIVISOR = 8  # gate G1: output ESR < 0.01 on all 18 files (tools/validate_integrator.py)
BOUNDS = dict(detector=.5, attack=1., release=1., output=.25, residual=.05)


class S6Circuit(AudioModel):
    def __init__(self, corrections='audio', **options):
        super().__init__(**options)
        if self.architecture != 's6_tfilm' or corrections not in ARCHITECTURES.values():
            raise ValueError('S6Circuit wraps s6_tfilm with corrections audio, knobs or none')
        self.corrections = corrections
        self.output = nn.Linear(self.output.in_features, len(BOUNDS))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.physical = CircuitScan(divisor=DIVISOR)

    def prior_loss(self):
        return self.physical.prior_loss()

    def _features(self, dry, controls, previous):
        history = previous.get('history')
        if history is None:
            history = dry.new_zeros(dry.shape[0], self.context-1)
        audio = torch.cat((history, dry), dim=1)
        windows = audio.unfold(1, self.context, 1)
        z = self.input(windows)
        spectrum = self.spectrum(torch.fft.rfft(windows, n=2*self.context).abs()/self.context)
        z, first = self.first(z, previous.get('first'))
        labels = controls if self.corrections == 'knobs' else torch.zeros_like(controls)
        z, condition = self.condition(z, spectrum, labels, previous.get('condition'))
        z, second = self.second(z, previous.get('second'))
        recurrent = dict(history=audio[:, -(self.context-1):] if self.context > 1 else audio[:, :0],
                         first=first, condition=condition, second=second)
        return self.output(z), recurrent

    def stream(self, dry, controls, pots, state=None):
        """One chronological chunk -> (wet, {} (no residual loss), next state); state['physical'] holds e/f/s."""
        state = {} if state is None else state
        b, t = dry.shape
        if self.corrections == 'none':
            raw, recurrent = None, {}
        else:
            raw, recurrent = self._features(dry, controls, state)
            raw = raw.double()
        bounded = {} if raw is None else {k: v*torch.tanh(raw[..., i]) for i, (k, v) in enumerate(BOUNDS.items())}
        per_step = {k: bounded[k].reshape(b, -1, DIVISOR).mean(-1) for k in ('detector', 'attack', 'release') if k in bounded}
        wet, _, physical = self.physical(dry, pots, state.get('physical'), per_step)
        if bounded:
            wet = wet*torch.exp(bounded['output'])+bounded['residual']
        return wet.to(dry.dtype), {}, dict(recurrent, physical=physical)

    def forward(self, dry, controls, pots, state=None):
        return self.stream(dry, controls, pots, state)
