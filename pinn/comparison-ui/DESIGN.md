# CL 1B comparison workspace

## 2026-09-25: Unified five-model comparison

Mode: Redesign · Preserve. Keep warm paper/charcoal colors, typefaces, 4/8 px spacing,
small control radii, original anchor IDs, offline audio, CSV, blind-test storage and
theme behavior. Move the five models to one common-window scorecard, with metric,
aggregation and knob filters; expose original studies through native details groups.
Dials: variance 4, motion 1, density 7, assets 1, fidelity 10. Zero-based bars and
numeric tables carry the comparison. No new animation or external assets.
Main risk: accidentally comparing mismatched scoring intervals or hiding old anchor
targets. Re-evaluate fixed weights on identical windows and auto-expand ancestor
details when following historical anchors. Data and original tools remain available.

Audience: the model's developer, comparing measured audio and selecting a checkpoint.
Artifact: an offline analysis instrument, not a product landing page.
Visual language: warm paper / charcoal, precise rules, large tabular measurements;
slate = numerical baseline, amber = PhysicsNeMo, teal = author model.
Mode: adapt a scientific audio workstation to a responsive browser page.
Dials (1–10): variance 4, motion 2, density 7, assets 2, brand fidelity 2.

Positioning: evidence-first narrative; desktop working distance; restrained and warm;
enough capacity for five conditions, three methods and four checkpoints.
Typography: Bahnschrift / Microsoft JhengHei headings, Segoe UI body, Consolas numbers.
Spacing: 4/8/12/16/24/32/48 px. Small radii only for controls. Charts keep consistent
method colors; all numerical charts use an explicit zero baseline and units.

Primary flow: compare averages → select condition → switch aligned audio → inspect
training progression. ESR is the initial metric and step 3000 the initial checkpoint.
The scope warning remains visible. Provenance and formulas have an expandable panel.
Controls: metric, checkpoint, condition, waveform/residual, track, seek, loop, volume,
light/dark theme, CSV export. Audio never autoplays on load or changes its gain per model.

Responsive: two columns above 1000 px, one column below; wrap control groups and allow
the evidence table to scroll horizontally. Respect reduced motion; label every control,
provide focus rings and textual equivalents of charts. Generate one self-contained HTML
with verified metrics, 40 WAV clips, waveform envelopes and no CDN/network dependency.

Verification: check every WAV's shape and sample rate; recompute metrics and compare to
the source results; check embedded data and JS syntax. Browser visual acceptance is
not claimed without an available browser inspection session.

## Extension: spectral, dynamics and personal listening evaluation

Mode: Extension. Preserve the existing theme, method colors, spacing, five-record
selection, aligned audio transport, training charts and offline single-file delivery.
Dials: variance 4, motion 2, density 7, assets 2, fidelity to existing UI 10.
Add spectral and RMS selectors to the same metric control; add manual dynamics
annotations immediately below the listening panel. No automatic attack/release labels
are inferred from the analog threshold. Empty annotations remain visibly unmeasured.

Blind evaluation opens a native modal dialog with an opaque backdrop: metric rankings,
waveforms and method colors are absent. Five neutral lettered candidates are shuffled
per round. Require listening and explicit scores before reveal. Freeze record/step for
the round; retain incomplete round in memory and finished rounds in localStorage when
available, with JSON export and honest storage status. No fabricated subjective data.
This is a personal comparison aid rather than an implementation of a formal MUSHRA study.
