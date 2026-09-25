# PhysicsNeMo audio sequence training implementation plan

**Goal:** Implement the two architectures already discussed and requested by the user, without starting training. Keep stereo WAV left=dry, right=wet.

**Design:** Add an independent `s6` package alongside the existing inverse PINN. Both models subclass PhysicsNeMo 2.2.2 `Module`; supervised waveform training does not invent a physics loss. S4 uses a complex diagonal SSM, causal FFT convolution with explicit carried state, and blockwise LSTM TFiLM. S6 uses two gated selective-SSM blocks around magnitude-spectrum/knob FiLM and GRU Temporal FiLM, then predicts a multiplicative gain. These are configurable reference implementations of the discussed topologies, not bit-for-bit ports or validated paper reproductions.

**Constraints:** No training run or optimizer updates during this task. CPU float32 reference and optional CUDA; no CUDA-only Mamba dependency. Maintain sample rate and amplitudes. Check dry/wet alignment, metadata and split overlap. Keep long-recording memory bounded by reading chunks. Carry detached state through chronological TBPTT chunks; reset only at record/split boundaries. TFiLM chunks are multiples of its block size. Validation uses its own warmup and never shares training state.

**Files and steps:**

- [x] `s6/tests/test_neural.py`: first specify channel/delay handling, disjoint partitions, chunk-vs-whole outputs, S6 causality, finite gradients, meaningful loss, checkpoint prediction roundtrip, and read-only CLI check. No optimizer calls.
- [x] `s6/data.py`: strict manifest with raw filename labels, half-open sample intervals and lazy stereo reads; explicit held-out files or one-file chronological split with guard interval; reject overlaps including shifted wet intervals.
- [x] `s6/layers.py`, `models.py`: explicit tensor state, detach helper, S4D/TFiLM and S6 gain models, PhysicsNeMo metadata and save/load support.
- [x] `s6/losses.py`, `training.py`: waveform plus multi-resolution spectral loss, chronological streaming validation, epoch checkpoints and epoch-boundary resume with config/data identity checks.
- [x] `s6/__main__.py`: `prepare`, `check`, `train`, and `evaluate`; check only reads data and performs inference. Training starts only through the explicit train command.
- [x] `s6/configs/neural-s4-tfilm.json`, `s6/configs/neural-s6-tfilm.json`, `s6/README.md`, README entry: reproducible commands, reference differences, long-release/warmup caveats, no claim of measured accuracy or real-time performance.
- [x] Run the new tests plus existing non-training regressions, compile/import and a read-only check on the supplied stereo WAV. Verify no training artifacts were created and leave changes reviewable locally.

**Validation:** `.venv/bin/python -m pytest s6/tests/test_neural.py -q`; then run existing tests excluding tests that invoke the existing PINN trainer. Forward/backward tests use synthetic tensors and never step an optimizer. The supplied real WAV is only read for data validation and an untrained forward pass. Save/load tests write random untrained weights to temporary directories only.

**Completed verification:** 35 tests passed with `pytest --ignore=tests/test_training.py`; compileall and diff whitespace checks passed. Both default architectures completed read-only 1024-sample train/validation forward checks on the supplied 48 kHz stereo WAV. No trainer call or optimizer update was executed. Epoch bundle creation and resume guards were tested with untrained weights in temporary directories.
