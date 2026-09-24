# HT-1B circuit solver and physics-informed identification

User authorizes two deliverables: numerical audio processing from circuit equations,
and PhysicsNeMo training from audio files. Implement in HT-1B; leave HT-76 untouched.

## Model boundary
This release is a reduced circuit model, not a complete transistor/valve netlist.
Keep the front-PCB resistor network and sidechain C3 charge/discharge topology.
Use finite-gain rail-limited U2B, constant-drop diodes, quasi-static C8, ideal
transformers, and a calibrated memoryless amplifier macro-model. The undocumented
GRE has two bounded optical states. Every approximation and unmeasured parameter
must be documented. Three switch modes select fixed/manual/diode-OR control;
fixed timing results from the GRE approximation and is not asserted to match hardware.

## Shared equations
A single backend-neutral expression builder defines front-network KCL, rectifier,
C3 currents, mode selection, optical targets/rates, and audio output for NumPy,
Torch and SymPy. States are C3 voltage and two optical occupancies. An implicit
backward-Euler solver solves all three states together with input held constant
per integration step, resolving gain-reduction feedback in the same step.
Use bounded nonlinear least squares, scale residuals, check convergence and fail
explicitly on failure. Preserve states across process blocks. Numerical parameters
are in SI units; normalized audio is converted using explicit volts-per-FS.

## PhysicsNeMo
Pin nvidia-physicsnemo[sym] 2.2.2. Subclass PDE and compile its continuous equations
with make_computations. Supply backward differences as time derivatives explicitly.
Use a PhysicsNeMo FullyConnected trajectory network with full-record time and
record identity as inputs; no waveform-history claim for this coordinate PINN.
Train bounded GRE parameters jointly, with physics residual, paired waveform loss,
and initial-state loss. Record-level holdout, no per-window state resets. Export
identified parameters to the numerical solver for unseen recordings; the trajectory
network itself is not an arbitrary-audio plugin. Check deployment using direct
solver metrics, separately from PINN reconstruction metrics.

## Audio contract
Accept one stereo WAV (left=dry, right=wet), or two paired mono WAVs; reject sample-rate/length mismatches, nonfinite or empty arrays.
No independent normalization/resampling/automatic channel mixing. Manifest provides
controls, pair paths, split, delay in samples and initial states. Optional output
allows physics-only dry files but provides no empirical hardware calibration.
Default examples use paired data. Positive delay means wet lags dry.
Write FLOAT WAV to avoid silent clipping; report peaks > 1. Each record has constant
controls. No real hardware recordings are provided. Synthetic data must be labeled.

## Deliverables
Package/CLI, JSON configs/manifest, synthetic data generator, solver WAV command,
PhysicsNeMo training/resume/export, metrics, tests, Chinese README/equations guide.
Validate analytic RC decay, KCL residuals, coupled solver convergence, streaming
continuity, sample-rate convergence, silence/bounds, input validation, actual
PhysicsNeMo residual parity/gradients, CPU training loss reduction and checkpoint
reload/export/inference. No claim of real-device fidelity or real-time performance.
