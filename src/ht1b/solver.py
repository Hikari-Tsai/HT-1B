"""Coupled implicit backward-Euler solver, independent of neural dependencies."""
import numpy as np
from scipy.optimize import root, least_squares
from .config import Circuit, Controls
from .equations import evaluate, NumpyOps, taper


class ConvergenceError(RuntimeError):
    pass


class CircuitSolver:
    def __init__(self, circuit: Circuit, controls: Controls, sample_rate: int,
                 substeps=1, initial_state=None, tolerance=1e-8):
        if isinstance(sample_rate, bool) or int(sample_rate) != sample_rate or sample_rate <= 0:
            raise ValueError('sample_rate must be a positive integer')
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
            raise ValueError('substeps must be a positive integer')
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        self.p, self.controls = circuit, controls
        self.sample_rate, self.substeps = int(sample_rate), int(substeps)
        self.dt = 1/(sample_rate*substeps)
        self.tolerance = tolerance
        self.upper = np.array([2*circuit.supply, 1., 1.])
        self.state = np.array([0.,0.,0.] if initial_state is None else initial_state, dtype=float)
        if self.state.shape != (3,) or not np.isfinite(self.state).all() or np.any(self.state < 0) or np.any(self.state > self.upper):
            raise ValueError('initial_state must be [C3 volts >= 0, fast in [0,1], slow in [0,1]]')
        self.max_residual = 0.
        self.samples_processed = 0
        ra = circuit.r11+circuit.attack_pot*taper(controls.attack,circuit.pot_taper)
        self.scale = np.array([1+self.dt*(circuit.opamp_gain+1)/(ra*circuit.c3),
                               1+self.dt/min(circuit.gre_attack_fast,circuit.gre_release_fast),
                               1+self.dt/min(circuit.gre_attack_slow,circuit.gre_release_slow)])

    def _step(self, x):
        old = self.state
        def residual(z):
            rates = np.array(evaluate(z, x, self.p, self.controls, NumpyOps)['rhs'])
            return (z-old-self.dt*rates)/self.scale
        solution = root(residual, old, method='hybr', options={'xtol':1e-9})
        z = solution.x
        def valid(z):
            return (np.isfinite(z).all() and np.all(z >= -1e-10) and np.all(z <= self.upper+1e-10)
                    and np.max(np.abs(residual(z))) < self.tolerance)
        if not valid(z):
            start = np.clip(z if np.isfinite(z).all() else old, 1e-12, self.upper-1e-12)
            solution = least_squares(residual, start, bounds=(np.zeros(3),self.upper),
                                     xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=120)
            z = solution.x
        z = np.clip(z, 0., self.upper)
        if not valid(z):
            raise ConvergenceError(f'Step {self.samples_processed}: residual {residual(z)}; try more substeps')
        self.max_residual = max(self.max_residual, float(np.max(np.abs(residual(z)))))
        self.state = z
        return evaluate(z,x,self.p,self.controls,NumpyOps)['output']/self.p.output_volts_per_fs

    def process(self, samples, return_states=False):
        x = np.asarray(samples,dtype=float)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('process expects finite mono samples')
        y = np.empty(len(x))
        states = np.empty((len(x),3)) if return_states else None
        for i, value in enumerate(x):
            for _ in range(self.substeps):
                out = self._step(value*self.p.input_volts_per_fs)
            y[i] = out
            if return_states: states[i] = self.state
            self.samples_processed += 1
        return (y,states) if return_states else y
