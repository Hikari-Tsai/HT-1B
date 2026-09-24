# HT-1B Implementation Plan

**Goal:** Deliver an executable reduced circuit solver and a PhysicsNeMo inverse-model training workflow.
**Architecture:** Share physical expressions between numerical and symbolic backends. Use paired WAVs to identify GRE parameters with a trajectory PINN and export parameters to the numerical solver.
**Tech Stack:** Python 3.12, NumPy/SciPy/SoundFile, PyTorch, PhysicsNeMo 2.2.2.
**Spec:** ../specs/2026-09-24-circuit-and-physicsnemo.md

## Constraints
Work directly in the authorized HT-1B project on codex/cl1b-research. Keep source PDFs local. Label approximations and synthetic data. Do not claim full circuit or hardware validation.

## Task 1 — direct solver
- [x] Add behavioral tests: analytic RC decay, front-network passive attenuation,
  coupled KCL convergence, chunk continuity, silence and invalid input.
- [x] Observe expected failure before implementing.
- [x] Create config.py (validated SI parameters), equations.py (shared expressions),
  solver.py (implicit state solver) and audio.py (WAV/manifest contract).
- [x] Run tests; compare integration refinement and physical residuals.

## Task 2 — physics-informed training
- [x] Add tests for symbolic residual consistency, parameter gradients and pair validation.
- [x] Implement physicsnemo_model.py with PDE, FullyConnected coordinate network,
  physically bounded parameter transforms and dimensionless residual scaling.
- [x] Implement training.py with train/validation records, initial conditions,
  deterministic sampling, checkpoint/resume, export and direct-solver validation.
- [x] Exercise the real installed PhysicsNeMo API on CPU, including an optimization step.

## Task 3 — user workflows
- [x] Implement cli.py commands simulate, demo-data, train, export, evaluate.
- [x] Run end-to-end synthetic WAV -> train -> checkpoint -> export -> unseen-WAV processing.
- [x] Write docs/EQUATIONS.md, docs/TRAINING.md, config and README commands.
- [x] Record test counts, actual training behavior and remaining fidelity limits.
