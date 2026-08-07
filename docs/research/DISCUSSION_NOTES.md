# Discussion notes

Insights, Q&A, and design reasoning (newest entries first) from research conversations. IDs: `NOTE-YYYYMMDD-NN`. Each entry includes **Timestamp** (`YYYY-MM-DD HH:MM:SS`, local system time at write).

**Related:** [experiment log](EXPERIMENT_LOG.md) · [planning notes](../onset_output_targets_planning.md) · [paper outline](PAPER_OUTLINE.md) · [pipeline architecture](PIPELINE_ARCHITECTURE.md) · [AR onset design](AR_ONSET_DESIGN.md) · [decisions checklist](DECISIONS_CHECKLIST.md)

## Session 2026-08-07 — QK-LN R2 diagnosis (evidence before next train)

### NOTE-20260807-09: Dense gap head is audio-blind (Phase 3 FAIL)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 15:35:57 |
| **Topic** | Tide gate result for `gap_residual` v1 |

**Verdict.** [EXP-20260807-14](EXPERIMENT_LOG.md#exp-20260807-14-tide-gap-residual-overfit--timing-near1-audio-blind): teacher **632/634**, ablation **identical** under reverse/shuffle/zeros (`same_pred` **1.0**). Relative Δ CE alone does not force audio use when the head is `Dense(gap_vocab)` over decoder state — same class as absolute index classification ([EXP-20260804-05](EXPERIMENT_LOG.md#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer)).

**Implication.** Before R2: replace Dense gap with a **content-based** gap (logits for Δ from `q · k(memory[prev+Δ])`, dense ids exact; log buckets as needed). Re-run tide + ablation; only then Phase 4.

**Related.** [NOTE-20260807-07](#note-20260807-07-relative-gap-δ-alignment-head--v1-spec) · [NOTE-20260807-08](#note-20260807-08-gap-alignment-phase-2--model--loss-landed) · [EXP-20260804-06](EXPERIMENT_LOG.md#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate)

### NOTE-20260807-08: Gap alignment Phase 2 — model + loss landed

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 15:24:59 |
| **Topic** | Gap head + CE wiring (no long GPU train) |

**Context.** [NOTE-20260807-07](#note-20260807-07-relative-gap-δ-alignment-head--v1-spec) Phase 0–1 shipped targets. Phase 2 wires the head and objective.

**Landed.**

| Piece | Detail |
| ----- | ------ |
| Config | `alignment: gap_residual` · `keep_absolute_pointer_head` · `patch_delta_*` · `gap_loss_weight` · helpers `gap_alignment_active` / `absolute_pointer_head_active` |
| Model | `gap_logits` Dense over `PatchGapVocab`; absolute pointer optional for A/B |
| Loss | Gap CE on `target_gap_ids` (teacher-forced prev); times = resolved `prev+Δ` + residual; **no** soft α / hard R on gap path |
| Decode / ablation | Teacher-fed + free-run + KV + audio ablation resolve via Δ |
| Tide config | [`configs/ar/tide_gap_residual.json`](../../configs/ar/tide_gap_residual.json) |

**Tests.** 80+ unit tests green across losses/models/targets/datasets/ablation; inference/kv/trainers also green. Log: `logs/gap_phase2_unit_test.log`.

**Phase 3.** Ran — **FAIL** on audio grounding ([EXP-20260807-14](EXPERIMENT_LOG.md#exp-20260807-14-tide-gap-residual-overfit--timing-near1-audio-blind), [NOTE-20260807-09](#note-20260807-09-dense-gap-head-is-audio-blind-phase-3-fail)).

**Next.** Content-based gap head, then re-gate tide. Hold R2 / design default.

**Related.** [NOTE-20260807-07](#note-20260807-07-relative-gap-δ-alignment-head--v1-spec) · `src/stepcovnet/onset_ar/{models,losses,trainers,inference}.py`

### NOTE-20260807-07: Relative gap (Δ) alignment head — v1 spec

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 14:53:53 |
| **Topic** | Spec for replacing absolute `patch_idx` CE with a relative gap head |

**Context.** Absolute pointer + full CE collapses on open-set localization (ptrloss ep2 timing **~0.0035**). Hard prev-local R=4 and soft distance α=0.5 raise timing but skill **vanishes** when the prior/mask is removed ([EXP-20260807-12](EXPERIMENT_LOG.md#exp-20260807-12-r4-weights-collapse-without-hard-window), [EXP-20260807-13](EXPERIMENT_LOG.md#exp-20260807-13-soft-distance-prior-beats-full-ce-still-prior-dependent)). Token LM already uses `delta_bucketed`; the alignment head is still absolute. Hard-R / force-advance stay diagnostic-only ([NOTE-20260807-06](#note-20260807-06-hard-r-is-diagnostic-not-the-holistic-system)).

**Representation (v1).**

| Step | Target | Decode |
| ---- | ------ | ------ |
| First onset | Absolute patch as Δ from virtual `prev=0` | `patch = clamp(Δ, 0…T′−1)` |
| Later | `Δpatch = patch − prev` (≥0; same-patch allowed) | `patch = clamp(prev + Δ, 0…T′−1)` |
| All | `residual_sec` in landed patch | `t = patch·P·hop + residual` |

Content gather uses the **resolved** absolute index after choosing Δ. Train/decode must not require soft α or hard R.

**Vocab edges (fit: `_tmp/r2_qk_ln_gap/`, R2 50t/50v).**

| Stat | Train | Val |
| ---- | ----- | --- |
| Later Δ p50 / p95 / p99 / max | 2 / 4 / 7 / 82 | 2 / 5 / 12 / 134 |
| First-abs p50 / p95 / max | 19.5 / 109 / 141 | 22 / 137 / 187 |
| Later Δ=0 fraction | ~2.0% | ~1.5% |

Defaults in `PatchGapVocab`: **`delta_max_dense=256`** (exact ids `0…256`), **`n_log_buckets=16`** for overflow. On R2 every later gap and first-abs is dense-exact (max 187 &lt; 256). Vocab size = **273**.

**Overflow.** `Δ > 256` → last log bucket (encode clamps); decode uses bin center. Rare on R2; acceptable lossy path for OOD long pauses/intros. No separate OVERFLOW id in v1.

**Mask.** `onset_step_mask` gates gap CE (same as pointer); pad/EOS steps ignored. Batch fields: `target_delta_patches` (raw Δ), `target_gap_ids` (vocab), keep `target_patch_indices` for residual/A/B.

**Success criteria (phased).**

1. **Unit:** dense Δ round-trip exact; gap-id → patch → time matches pointer+residual on tide/synthetic.
2. **Tide gate:** no soft α / hard R; teacher timing ≈1; audio ablation must move the **gap** head.
3. **R2 probe:** beat ptrloss baseline (**timing ~0.0035**) **without** decode prior; log EXP.
4. **Cleanup:** default alignment = gap+residual; soft α / hard R diagnostic-only in design doc.

**Out of scope for v1.** Anneal-α on absolute pointer; ladder R3+; hard-R stacking.

**Implementation status.** Phase 0–2 done ([NOTE-20260807-08](#note-20260807-08-gap-alignment-phase-2--model--loss-landed)). Phase 3 Dense gap **FAIL** ([EXP-20260807-14](EXPERIMENT_LOG.md#exp-20260807-14-tide-gap-residual-overfit--timing-near1-audio-blind) / [NOTE-20260807-09](#note-20260807-09-dense-gap-head-is-audio-blind-phase-3-fail)). Next: content-based gap, then re-gate.

**Related.** [NOTE-20260807-06](#note-20260807-06-hard-r-is-diagnostic-not-the-holistic-system) · [NOTE-20260807-08](#note-20260807-08-gap-alignment-phase-2--model--loss-landed) · [EXP-20260807-12](EXPERIMENT_LOG.md#exp-20260807-12-r4-weights-collapse-without-hard-window) · [EXP-20260807-13](EXPERIMENT_LOG.md#exp-20260807-13-soft-distance-prior-beats-full-ce-still-prior-dependent) · `src/stepcovnet/onset_ar/targets.py` · JRN-20260807-03

### NOTE-20260807-06: Hard R is diagnostic, not the holistic system

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:39:00 |
| **Topic** | Steering: hard prev-local windows are crutches |

**Context.** Prev-local R shrink (32→8→4) produced large timing gains, and the agent proposed force-advance / further R tricks.

**User steering.** Improving val by hard-restricting the pointer to “notes are nearby” is not the long-term approach. The goal is a **holistic** system that can predict open-ended timing — including long pauses — not one that overfits the train/val gap histogram. Hard-R gains are informative, not a product recipe to stack.

**Implication.**

1. Keep hard-R EXPs as **evidence** that diffuse full-suffix CE was a real failure mode.
2. Stop recommending R-shrink / force-advance / min_ahead as **Now** unless the user explicitly asks for a diagnostic-only probe.
3. Prefer learned localization under mono/full support that does not hard-fail long gaps.

**Follow-up.** Unmasked eval → **Done** [EXP-20260807-12](EXPERIMENT_LOG.md#exp-20260807-12-r4-weights-collapse-without-hard-window): **collapse** (timing **0.156→0.0016**).

**Follow-up.** Soft distance prior → **Done** [EXP-20260807-13](EXPERIMENT_LOG.md#exp-20260807-13-soft-distance-prior-beats-full-ce-still-prior-dependent): α=0.5 timing **0.105** / F1 **0.279**; α=0 eval still collapses.

**Evidence-backed next.** Localization that survives **without** a decode prior (relative gap head, or anneal α→0). Soft prior &gt; hard-R as a crutch, not the product path.

**Related.** [EXP-20260807-10](EXPERIMENT_LOG.md#exp-20260807-10-prev-local-r4-beats-r8) · [EXP-20260807-11](EXPERIMENT_LOG.md#exp-20260807-11-r4-error-mix--stickiness-not-mid-window-soup) · [EXP-20260807-12](EXPERIMENT_LOG.md#exp-20260807-12-r4-weights-collapse-without-hard-window) · [EXP-20260807-13](EXPERIMENT_LOG.md#exp-20260807-13-soft-distance-prior-beats-full-ce-still-prior-dependent) · JRN-20260807-03 · whats-next § No hard locality hacks

### NOTE-20260807-05: Inside r=32 the soup shrinks to ~33-way; GT is left-skewed

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 11:53:22 |
| **Topic** | In-window error geometry on prev-local v3 ckpt |

**Context.** [EXP-20260807-06](EXPERIMENT_LOG.md#exp-20260807-06-prev-relative-local-ce-beats-ptrloss-ep2) raised timing to **0.0228** but patch-acc stayed ~**5.5%**.

**Discovery** (12 val / 8 train, decode mask on). Val: GT `target−prev` p50 **2**, pred `pred−prev` p50 **15**, H/Huni **0.95**, **63%** wrong-far-in-window. Mask integrity OK (0 outside / behind).

**Implication.** Failures are not “stuck at prev”; the model still spreads mass across the allowed window while true gaps hug the left edge. Shrinking **R** is the one-knob match to measured gap mass (p50=2, p95≈5, p99≈12).

**Follow-up.** R=8 probe → **Done** [EXP-20260807-08](EXPERIMENT_LOG.md#exp-20260807-08-prev-local-r8-beats-r32): timing **0.0697**, patch-acc **16.6%**. In-window/residual mix → **Done** [EXP-20260807-09](EXPERIMENT_LOG.md#exp-20260807-09-r8-still-diffuse--residual-secondary): still H/Huni **0.96**; **83%** patch_wrong vs **10%** residual-tax.

**Follow-up (cont).** R=4 → **Done** [EXP-20260807-10](EXPERIMENT_LOG.md#exp-20260807-10-prev-local-r4-beats-r8): timing **0.156**, F1 **0.251** (skill **−0.084**).

**Follow-up (cont).** R=4 error-mix → **Done** [EXP-20260807-11](EXPERIMENT_LOG.md#exp-20260807-11-r4-error-mix--stickiness-not-mid-window-soup): **31%** at_prev; R=2 starves **20%**.

**Open.** Superseded by [NOTE-20260807-06](#note-20260807-06-hard-r-is-diagnostic-not-the-holistic-system): no more hard-mask stacking; learn localization without hard R.

**Related.** [EXP-20260807-07](EXPERIMENT_LOG.md#exp-20260807-07-in-window-errors-are-left-skewed-gt-vs-diffuse-mid-window-preds) · [EXP-20260807-08](EXPERIMENT_LOG.md#exp-20260807-08-prev-local-r8-beats-r32) · [EXP-20260807-09](EXPERIMENT_LOG.md#exp-20260807-09-r8-still-diffuse--residual-secondary) · [EXP-20260807-10](EXPERIMENT_LOG.md#exp-20260807-10-prev-local-r4-beats-r8) · [EXP-20260807-11](EXPERIMENT_LOG.md#exp-20260807-11-r4-error-mix--stickiness-not-mid-window-soup) · `_tmp/r2_qk_ln_gap/r4_error_mix.json`

### NOTE-20260807-04: Prev-relative local CE works — but hard-mask CE needs OOD skip

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 11:47:29 |
| **Topic** | Decode-consistent `[prev, prev+R]` CE + traps that poison sparse CE |

**Context.** Target-centered local CE failed ([EXP-20260807-05](EXPERIMENT_LOG.md#exp-20260807-05-clean-local-ce-r32--no-beat--88-preds-outside-window)). Prev-relative probe hit two CE poison modes before a fair train.

**Traps.**

1. **First onset (`prev=0`)** — ~38% of val songs start after patch 32. Upper-bound `[0, R]` masks the label → mean CE ~1e6. Fix: no upper bound when `prev==0`.
2. **Section gaps `>R`** — rare (val **0.13%**, max gap **134** patches) but enough to keep mean CE ~**1e6** after (1). Fix: drop those steps from pointer CE; keep decode at fixed R.

**Result.** [EXP-20260807-06](EXPERIMENT_LOG.md#exp-20260807-06-prev-relative-local-ce-beats-ptrloss-ep2): offline timing **0.0228** vs ptrloss **0.0035**; timing skill **+0.0145**. F1 skill still **−0.40**.

**Implication.** Binding failure is no longer “800-way soup”; it is **in-window miss** (~5.5% patch-acc). → **Done** [NOTE-20260807-05](#note-20260807-05-inside-r32-the-soup-shrinks-to-33-way-gt-is-left-skewed): ~33-way diffuse mid-window vs left-skewed GT.

**Related.** `_tmp/r2_qk_ln_gap/prev_local_gaps.json` · `pointer_mask.prev_relative_ce_step_mask` · configs `ladder_r2_prev_local_ce_probe.json` (v3)

### NOTE-20260807-03: Far_ahead at ep2 is diffuse mono-suffix mass, not confident wrong peaks

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 10:34:45 |
| **Topic** | Entropy / top-k of mono-masked pointer softmax — ep2 vs ep31 |

**Context.** [NOTE-20260807-02](#note-20260807-02-at-selected-weights-both-splits-are-weak--late-train-localization-is-memorization) left open whether ~97% far_ahead is near-uniform over the allowed suffix vs peaked wrong. Scale-up declined.

**Discovery** (`_tmp/r2_qk_ln_gap/far_ahead_entropy.json`, 8 train / 12 val; mono-masked softmax).

| Ckpt / split | far_ahead frac | mean n_allowed | H / H_uniform | mean top-1 | mean top-5 | target rank p50 |
| ------------ | -------------- | -------------- | ------------- | ---------- | ---------- | --------------- |
| **ep2 val** | 0.974 | **809** | **0.921** | **0.016** | 0.062 | **153** |
| ep2 train | 0.942 | 759 | 0.909 | 0.020 | 0.077 | 66 |
| ep31 val | 0.956 | 809 | 0.584 | **0.202** | 0.419 | 94 |
| ep31 train | 0.742 | 746 | 0.643 | 0.100 | 0.310 | 11 |

**Implication.**

1. At the **selected** ep2 weights, far_ahead is **diffuse**: entropy ≈ uniform over ~**800** allowed patches; top-1 mass **~1.6%**. Not “confident wrong peaks.”
2. Late ep31 **does** peak (val top-1 **~20%**) but on the **wrong** patches (target still rank ~**94**) — memorized wrong modes, matching worse-than-uniform NLL.
3. Binding failure at the fair operating point: the model never concentrates mass near the target early; CE fights a ~800-way mono-suffix soup.

**Evidence-backed next (fixed R2, no scale).** Clean **local pointer CE** short probe (`pointer_local_ce_radius` > 0, hard time, **no** STE, ckpt `val_pointer_loss`). → **Done** [EXP-20260807-05](EXPERIMENT_LOG.md#exp-20260807-05-clean-local-ce-r32--no-beat--88-preds-outside-window): **not supported**; **88%** of val preds outside ±32 — train/infer mismatch.

**Follow-up.** Decode-consistent prev-relative CE → **Done** [EXP-20260807-06](EXPERIMENT_LOG.md#exp-20260807-06-prev-relative-local-ce-beats-ptrloss-ep2) / [NOTE-20260807-04](#note-20260807-04-prev-relative-local-ce-works--but-hard-mask-ce-needs-ood-skip).

**Open.** In-window error modes; best R; soft outside-window penalty.

**Related.** [EXP-20260807-04](EXPERIMENT_LOG.md#exp-20260807-04-far_ahead-entropy--ep2-diffuse-ep31-peaked-wrong) · [EXP-20260807-05](EXPERIMENT_LOG.md#exp-20260807-05-clean-local-ce-r32--no-beat--88-preds-outside-window) · [EXP-20260807-06](EXPERIMENT_LOG.md#exp-20260807-06-prev-relative-local-ce-beats-ptrloss-ep2) · `_tmp/r2_qk_ln_gap/diagnose_far_ahead_entropy.py` · JRN-20260807-02

### NOTE-20260807-02: At selected weights both splits are weak — late train localization is memorization

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 10:16:09 |
| **Topic** | Train/val gap at the fair `val_pointer_loss` ckpt (ep2) vs the late patch-acc ckpt (ep31) |

**Context.** After [EXP-20260807-02](EXPERIMENT_LOG.md#exp-20260807-02-val_pointer_loss-selection-picks-ep2--timing-worse-than-patch-acc-ckpt), Current phase still said “attack transfer with reg/data,” but which knob was unmeasured. Prior error-modes ([NOTE-20260807-01](#note-20260807-01-train-localizes-val-does-not--gap-is-the-binding-failure)) were on the **ep31** export.

**Discovery.**

| Source | Finding |
| ------ | ------- |
| Epoch curves (`_tmp/r2_qk_ln_gap/epoch_curves.json`) | Best val NLL @ **ep2**: train patch-acc **0.0138** ≈ val **0.0137**. Late epochs: train climbs to **~0.08+** while val stays **~0.015–0.019** and val NLL drifts to/above uniform |
| Offline teacher, ptrloss ep2 | Train (8): timing **13/3361 = 0.0039**; val (50): **124/35439 = 0.0035** — same floor. `logs/r2_qk_ln_ptrloss_teacher_train.log` |
| Error-modes ptrloss ep2 | Train patch-acc **2.7%**, NLL **5.57** (−1.7 vs uniform), median \|Δ\| **107**, far_ahead **97%**. Val **1.2%**, NLL **6.12** (−1.2), median **157**, far_ahead **98%**. `_tmp/r2_qk_ln_ptrloss_diag/error_modes.json` |
| Contrast ep31 (old) | Train **11.5%** / NLL **3.64** / median **12** vs val **1.7%** / NLL **7.65** (worse than uniform) — the “train localizes” story is **late memorization**, not the selected operating point |

**Implication.**

1. At the **exported** weights there is **no large train≫val gap** — both splits are weak. Dropout/reg aimed at “stop late overfit” does not create early transferable skill.
2. Late train localization (11% patch-acc) is real but **anti-correlated** with val NLL; it is not a transfer success waiting for a better monitor.
3. Binding failure reframed: **50-song R2 never learns a generalizable pointer early**; it only later memorizes train. Architecture grounding (ablation PASS on late train) is necessary but not sufficient for transferable skill.

**Evidence-backed next (no scale-up).** Stay on fixed R2. First close the open from [NOTE-20260807-01](#note-20260807-01-train-localizes-val-does-not--gap-is-the-binding-failure): is **far_ahead** diffuse mass over the mono-allowed suffix vs confident wrong peaks (entropy / top-1–top-k mass on ep2 vs ep31)? → **Done** [NOTE-20260807-03](#note-20260807-03-far_ahead-at-ep2-is-diffuse-mono-suffix-mass-not-confident-wrong-peaks): ep2 diffuse (H/Huni **0.92**); next = clean local CE probe.

**Open.** Closed for entropy; see NOTE-03 for local-CE radius.

**Related.** [EXP-20260807-02](EXPERIMENT_LOG.md#exp-20260807-02-val_pointer_loss-selection-picks-ep2--timing-worse-than-patch-acc-ckpt) · [EXP-20260807-03](EXPERIMENT_LOG.md#exp-20260807-03-gap-diagnosis--selected-ep2-both-splits-weak) · `_tmp/r2_qk_ln_gap/` · JRN-20260807-01 · JRN-20260807-02

### NOTE-20260807-01: Train localizes; val does not — gap is the binding failure

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 00:26:58 |
| **Topic** | Error-mode diagnosis on [`ladder_r2_qk_ln_probe`](../../configs/ar/ladder_r2_qk_ln_probe.json) ckpt before proposing another loss |

**Context.** After the defect sweep ([NOTE-20260806-03](#note-20260806-03-remaining-defect-inventory-after-encode-then-pe)), R2 still sits at ~**1.7%** val patch-acc / timing **~0.007**. Speculative attn-mass was deferred until measurements justified a lever.

**Discovery (existing logs + new diag).**

| Source | Finding |
| ------ | ------- |
| Train log (34 ep) | Best **val** patch-acc **0.0185 @ ep 31** with `val_pointer_loss` **7.39 ≈ uniform (~7.37)**. Best **val NLL** was earlier (**5.97 @ ep 2**) with lower patch-acc. Train ends ~**6.9%** patch-acc / pointer CE **~4.2** |
| Offline val (50 songs) | Timing **0.0070**, `patch_wrong` **34828/35439**, F1 skill **−0.43** |
| Ablation (12 val) | Pointer **PASS** (shuffle same_pred **0**); zeros query moves; tokens barely move |
| New diag (`_tmp/r2_qk_ln_diag/error_modes.json`, 8 train / 12 val) | **Train:** patch-acc **0.115**, NLL **3.64** vs uniform **7.31** (−**3.68** nats), median \|Δpatch\| **12**. **Val:** patch-acc **0.017**, NLL **7.65** vs uniform **7.37** (**+0.28**, *worse* than uniform), median \|Δpatch\| **127**. Signed Δ strongly positive (pred ahead). **96.6%** of val preds are `far_ahead` of `prev+1` |

**Implication.**

1. The pointer **can** learn on train songs — this is not “CE cannot localize.” Attn-mass / more CE strength is the wrong next bet.
2. Binding failure is **train→val transfer** (and/or selection): val is anti-calibrated vs uniform while train is strongly peaked.
3. `val_pointer_patch_accuracy` alone picked an epoch whose val NLL is ~uniform; early NLL-better epochs were discarded — selection is part of the problem.

**Evidence-backed next (in order).**

1. **Train-split ablation** on the same ckpt — confirm train pointer grounding (expect PASS, not keys-only). → **Done** [EXP-20260807-01](EXPERIMENT_LOG.md#exp-20260807-01-qk-ln-train-ablation-pass--ckpt-on-val_pointer_loss): full gate PASS; matched timing **0.043**, NLL **3.64**.
2. **Selection fix:** early-stop / checkpoint with a metric that prefers NLL under uniform (e.g. `val_pointer_loss` min) — justified by ep2 vs ep31 numbers above. → **Done** in ladder content-pointer configs (`val_pointer_loss`); retrain still required to export a new best.
3. Only after (1)–(2): attack the **generalization gap** (reg / data / recipe), not another pointer-head aux — start with the `val_pointer_loss` retrain so selection is not still confounding the gap. → **Done** [EXP-20260807-02](EXPERIMENT_LOG.md#exp-20260807-02-val_pointer_loss-selection-picks-ep2--timing-worse-than-patch-acc-ckpt): ep2 selected; offline timing **worse** than ep31. Gap remains.

**Open.** Whether val “far ahead” is mostly near-uniform mass over the mono-allowed suffix (geometric bias) vs confident wrong peaks — NLL at/above uniform on val favors the latter or diffuse high-entropy wrong mass.

**Related.** [EXP-20260806-07](EXPERIMENT_LOG.md#exp-20260806-07-pointer-qk-layernorm--tide-pass-r2-patch-acc-visible-timing-flat) · [EXP-20260807-01](EXPERIMENT_LOG.md#exp-20260807-01-qk-ln-train-ablation-pass--ckpt-on-val_pointer_loss) · `logs/r2_qk_ln_error_modes.log` · JRN-20260807-01

## Session 2026-08-06 — pointer time objective is broken (not a recipe miss)

### NOTE-20260806-03: Remaining defect inventory after encode-then-PE

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 19:57:00 |
| **Topic** | Exhaustive follow-up — do not stop at one architecture win |

**Closed this pass (code + unit / tide proof).**

| # | Defect | Fix |
| - | ------ | --- |
| 1 | Ablation timing clause always fails when matched &lt; `GATE_TIMING_MATCH_EPS` (0.02) | Skip timing≈matched when matched below eps; rely on `same_pred` |
| 2 | No logged patch top-1; ckpt on floor `val_timing_match_teacher` | `pointer_patch_accuracy` metric; ladder configs can monitor it |
| 3 | Patch-acc / ablation NLL omitted teacher mono mask | Mono before metric / NLL |
| 4 | PE keys footgun if `encoder_memory` reused without pe-free keys | Refuse silent fallback in kv_decode / trainer / parallel decode |
| 5 | Infer `pe_free_keys` from config only | Prefer `cross_memory` topology |
| 6 | `Lambda(content + 0·PE)` not safe-reloadable | `ContentOnlyCrossMemory` registered layer |

**Ruled out with proof.**

| Hypothesis | Result |
| ---------- | ------ |
| Decoder cross content-only (no PE residual) | **Tide regresses** — peak timing **0.67** vs encode-then-PE mix **~0.94**. Default stays **False**; R2 A/B only |

**Proved this pass.**

| Item | Result |
| ---- | ------ |
| QK LayerNorm | Tide gate **PASS** (~0.99). R2 short: offline timing **0.0070** ≈ ctx-pefree **0.0069**; patch-acc **~1.7%** ([EXP-20260806-07](EXPERIMENT_LOG.md#exp-20260806-07-pointer-qk-layernorm--tide-pass-r2-patch-acc-visible-timing-flat)) |
| Ablation floor fix | R2 val gate pointer **PASS** (shuffle same_pred **0**) — previously always FAIL on timing≈floor |

**Still open (binding for R2 localization).**

| # | Issue | Status |
| - | ----- | ------ |
| A | Hard `λ_time` still `grad_pointer=0` in ladder recipe | Known; STE exists, recipe closed vs hard CE |
| B | Shuffle query cos still ~1 (pooling); zeros is the decoder probe | Not a wiring bug; optional attn-mass aux |
| C | Absolute skill still ≪ null on R2 | Need patch-acc ≫ chance / positive F1 skill |

### NOTE-20260806-02: Pe-free keys skipped the encoder — encode-then-PE fixes it

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 13:02:01 |
| **Topic** | Architecture root cause after STE recipe branch closed |

**Verdict.** `pointer_keys_pe_free: true` used ``Dense(MERT)`` (`patch_embed`) as keys — **before** any encoder layer — because post-encoder `memory` was PE-dominated when PE was applied *before* encoding. That avoided PE but threw away contextualization.

**Fix.** Encode on `patch_embed` **without** absolute PE, then add `enc_pos`. Pe-free keys = encoder output; `memory` = content + PE for positional decoder paths.

**Proof ([EXP-20260806-04](EXPERIMENT_LOG.md#exp-20260806-04-encode-then-pe--contextualized-pe-free-keys-beat-hard-short-probe)).**
- Unit: content ≠ raw `patch_embed`
- Tide gate **PASS** (timing **0.94**)
- R2 50-ep: val + offline timing **0.0069** vs hard short **0.005**; val pointer NLL **6.14** &lt; uniform

**Full R2 ([EXP-20260806-05](EXPERIMENT_LOG.md#exp-20260806-05-full-r2-encode-then-pe-beats-prior-hard-time-full-run)).** Best val **0.00945** / offline **0.0094** vs prior hard **0.0085 / 0.0086**; F1 **0.046** vs **0.021**. Fix confirmed at full scale. Still negative null skill.

**Open.** See [NOTE-20260806-03](#note-20260806-03-remaining-defect-inventory-after-encode-then-pe) — content-only ruled out; QK LN + metric/gate fixes landed.

### NOTE-20260806-01: Hard `λ_time` never trains the pointer; soft `E[patch]` is non-localizing; offline drops monotonic

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 10:35:59 |
| **Topic** | Root cause of “R2 never learns meaningful timing” after hard-time / wiring probes |

**Verdict.** Recipe probes (λ_residual, monotonic off, ramp/dropout) are the wrong lever. Three concrete defects explain the floor:

1. **Hard-argmax `λ_time` → `grad_pointer = 0`.** In `predicted_times_from_outputs`, `use_soft_expected=false` uses `tf.argmax`, which is non-differentiable. Isolating `λ_time=1`, `pointer_loss_weight=0`, `λ_residual=0`: hard → `grad_pointer=0.0` / `grad_residual=0.707`; soft → `grad_pointer=0.068` / same residual. STE (`soft + stop_gradient(hard−soft)`) restores soft grads while keeping hard forward loss. Documented earlier as a gap ([NOTE-20260805-01](#note-20260805-01-pointer-gate-pass-was-keys-only--the-decoder-never-read-the-audio) item 3) then contradicted by EXP-05’s hypothesis that hard time “backprops into the pointer.”

2. **Soft `E[patch]` MAE is non-localizing.** A bimodal `p` with `E[patch]=target` has soft abs-err **0** and `p(target)=0` (hard err = many patches). Soft R2 after the wiring fix kept train **and** val timing at ~**10⁻³** while tokens overfit ([EXP-20260805-03](EXPERIMENT_LOG.md#exp-20260805-03-fixed-stack-r2-content-pointer-still-at-timing-floor)) — consistent with minimizing soft seconds without peaking the pointer. STE alone reintroduces that soft gradient; it is necessary for “hard metric + differentiable time,” **not sufficient** as the sole localization loss.

3. **Offline teacher decode omits monotonic; in-train / ablation keep it.** `eval_ar_onset_offline.py` calls `decode_teacher_fed_times_numpy` without `target_patch_indices` / `monotonic`. Same hard-time R2 ckpt: in-train val **0.0085**, ablation matched **0.0071**, offline **0.00079** (~**11%** of ablation). Toy: constant argmax@0 → free mean abs err **0.67** s; teacher-forced mono → **0.24** s without any pointer learning. The train/offline “cliff” is mostly a **metric bug**. Ablation still shows real failure: `pointer_nll` **11.2** ≫ uniform **7.37**, `patch_wrong` **98.5%**, shuffle `query_cosine` ≈ **1.0**.

**Implication.** Do **not** run another tide-parity hyperparam probe next. Fix the objective + metric:

| Fix | Proof already in hand | Remaining proof |
| --- | -------------------- | --------------- |
| A. Localizing pointer loss (full/local CE primary; optional entropy / attention-mass aux). Keep hard decode for metrics. | Soft non-localizing toy; hard grad=0; val NLL worse than uniform | 50-ep R2: train **and** offline `patch_wrong` fall; val `pointer_nll` &lt; uniform |
| B. If retaining `λ_time` on seconds: STE or soft-for-loss + hard-for-metric; mask `time_loss` to correct-patch steps (or `stop_gradient` patch) so residual is not corrupted when patch is wrong | STE unit grad; residual_loss already ~0 while time_loss ~20 s on wrong patches | Short probe vs hard baseline |
| C. Offline teacher path mirrors train/ablation monotonic | Call-site + metric bridge 0.0085≈0.0071≫0.00079 | Re-score same ckpt offline with `monotonic=True` → ~ablation timing |
| D. Keep demanding shuffle-sensitive queries (architecture) | Ablation shuffle query cos ≈ 1 | Gate must keep failing until queries move |

**Follow-up ([EXP-20260806-01](EXPERIMENT_LOG.md#exp-20260806-01-ste--local-ce--offline-monotonic-fix)–[03](EXPERIMENT_LOG.md#exp-20260806-03-ste-without-correct-patch--grads-live-still-below-hard-ce)).** Fix **C landed and proved**: offline teacher on hard R2 **0.00079 → 0.0086**. STE variants (local CE **0.0037**, full CE+correct-patch **0.004**, full CE no-mask **0.0041** with live `time_loss` ~**21**) all **miss** hard short probe (**0.005**). Soft/`STE` seconds are not the multi-song lever. Next: architecture — shuffle-sensitive pointer queries.

**Defer.** `lambda_time_ramp_epochs`, `dropout_rate: 0`, R3+ — until a localizing objective beats hard-time in-train.

**Related:** [EXP-20260806-01](EXPERIMENT_LOG.md#exp-20260806-01-ste--local-ce--offline-monotonic-fix) · [EXP-20260805-03](EXPERIMENT_LOG.md#exp-20260805-03-fixed-stack-r2-content-pointer-still-at-timing-floor) · [EXP-20260805-05](EXPERIMENT_LOG.md#exp-20260805-05-hard-pointer-time-probe-raises-timing-10-but-still-at-floor) · [EXP-20260805-06](EXPERIMENT_LOG.md#exp-20260805-06-full-r2-hard-pointer-time--timing-rises-still-below-skill) · [NOTE-20260805-01](#note-20260805-01-pointer-gate-pass-was-keys-only--the-decoder-never-read-the-audio)

## Session 2026-08-05 — content-pointer decoder is audio-blind

### NOTE-20260805-06: Tide `lambda_residual: 30` does not beat lam5 hard-time R2

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 00:53:53 |
| **Topic** | Recipe probe after full hard-time R2 still at offline floor |

**Verdict.** Hard time + `lambda_residual: 30` on 50t/50v peaks in-train val timing **0.0065 @ ep 16** ([EXP-20260805-07](EXPERIMENT_LOG.md#exp-20260805-07-hard-time--tide-lambda_residual30--no-beat-over-lam5-r2)). That is ~**1.3×** the lam5 hard short probe (**0.005 @ ep 7**) but **below** the full lam5 hard R2 (**0.0085 @ ep 96**). Tide residual weight alone is not the missing multi-song ingredient.

**Next.** Other tide-vs-ladder diffs: `lambda_time_ramp_epochs`, `dropout_rate: 0`, incremental consistency — or offline eval on this ckpt to see whether the train/offline val gap persists.

### NOTE-20260805-05: Hard pointer time moves timing ~10× — soft time was starving the pointer

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 21:51:02 |
| **Topic** | Recipe probe after monotonic ruled out |

**Verdict.** `use_soft_pointer_time: false` on 50t/50v peaks val timing **0.005 @ ep 7** vs **0.00054** with soft time + no monotonic ([EXP-20260805-05](EXPERIMENT_LOG.md#exp-20260805-05-hard-pointer-time-probe-raises-timing-10-but-still-at-floor)). Still at the null floor, but the first axis that measurably moves the needle. Soft expected time likely decouples `lambda_time` from sparse pointer CE on multi-song charts.

**Next.** Full R2 with hard time, or hard time + tide `lambda_residual: 30`.

### NOTE-20260805-04: Monotonic pointer is not why R2 timing stays at zero

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 21:39:29 |
| **Topic** | First recipe probe after wiring fix |

**Verdict.** Offline teacher on 5 train songs with the fixed R2 ckpt: **1/1796** timing matches, **1793** patch wrong, F1 skill **−0.60** vs null ([EXP-20260805-04](EXPERIMENT_LOG.md#exp-20260805-04-r2-no-monotonic-probe-still-at-timing-floor)). Short retrain with `monotonic_pointer: false` peaks val timing **0.00054 @ ep 7** — same floor. Monotonic is ruled out; probe logs show `time_loss` ≫ pointer CE in nats but timing still does not move — investigate hard-argmax time vs soft, and tide λ_residual/dropout parity.

### NOTE-20260805-03: Fixed-stack R2 still has no timing signal — even on train

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 21:16:11 |
| **Topic** | Post-wiring R2 retrain vs tide |

**Verdict.** The decoder-audio package is tide-validated ([EXP-20260805-02](EXPERIMENT_LOG.md#exp-20260805-02-decoder-audio-fix--tide-content-pointer-passes-queryzeros-gate)) but R2 under the same recipe never leaves the timing floor ([EXP-20260805-03](EXPERIMENT_LOG.md#exp-20260805-03-fixed-stack-r2-content-pointer-still-at-timing-floor)): best val timing **0.0014 @ ep 4**, and **train timing also stays ~10⁻³** while tokens overfit to ~0.42. That differs from the pre-fix R2, where train timing climbed to ~0.42 via keys-only lookup.

**Implication.** Do not re-litigate mask polarity or “is the gate wrong?” for this failure. Ask why the pointer time/CE path does not train across 50 songs (recipe, capacity, monotonic+soft interaction, data). Commit the wiring/tests/docs first; then diagnose with train-split probes.

**Related:** [EXP-20260805-03](EXPERIMENT_LOG.md#exp-20260805-03-fixed-stack-r2-content-pointer-still-at-timing-floor)

### NOTE-20260805-02: Decoder-audio fixes land — zeros (not shuffle) is the query probe

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 20:18:58 |
| **Topic** | Shipping the three fixes from [NOTE-20260805-01](#note-20260805-01-pointer-gate-pass-was-keys-only--the-decoder-never-read-the-audio) and what “fixed” means |

**Verdict.** Tide content-pointer retrained with corrected masks, PE-free pointer/decoder cross-attn on `patch_embed`, monotonic pointer, soft time, and a gate that demands zeros query/token sensitivity now **PASSes** with matched timing **0.94** and zeros `query_cosine` **0.42** ([EXP-20260805-02](EXPERIMENT_LOG.md#exp-20260805-02-decoder-audio-fix--tide-content-pointer-passes-queryzeros-gate)). Pre-fix tide had query cos ≈ **1.0** under zeros too.

**Gotcha that almost fooled us.** First retrain still pointed `pointer_cross_attn` at post-encoder `memory`, which stays shuffle-invariant (cos **~0.99**) because PE dominates — so queries stayed fixed while pe-free **keys** flipped the pointer. Fix: cross-attn values/keys for the pointer (and a `patch_embed + Dense(memory)` mix for the decoder) must read the PE-free stream.

**Gate calibration.** Requiring shuffle `query_cosine` ≪ 1 rejects legitimate attention-pooled models (near-uniform / decoder-dominated attention is permutation-invariant). The keys-only bug fails **zeros**; the standing gate now keys decoder pass/fail off zeros token **or** zeros query, while shuffle still must collapse the **pointer**.

**Next.** R2 retrain with the fixed config — prior ladder content-pointer checkpoints are invalid for transfer claims.

**Related:** [EXP-20260805-02](EXPERIMENT_LOG.md#exp-20260805-02-decoder-audio-fix--tide-content-pointer-passes-queryzeros-gate) · [EXP-20260805-01](EXPERIMENT_LOG.md#exp-20260805-01-content-pointer-audio-signal-is-keys-only--decoder-is-audio-blind)

### NOTE-20260805-01: “Pointer gate PASS” was keys-only — the decoder never read the audio

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 19:49:17 |
| **Topic** | Why content-pointer R2 has zero val skill: architecture/wiring, not regularization |

**Verdict.** The content-pointer head can fail the audio ablation on **pointer scores** while the **decoder is completely audio-blind**. On the R2 ep-16 checkpoint, shuffling `mert_patches` leaves `pointer_query` / decoder / token logits at cosine ≈ **1.0**; only `pointer_key(memory)` moves. Swapping shuffled queries onto matched keys does not change patch accuracy; swapping shuffled keys onto matched queries collapses it. So train “grounding” is **chart-conditioned queries × audio keys**, not cross-attentional listening. Val cannot transfer that lookup. Numbers: [EXP-20260805-01](EXPERIMENT_LOG.md#exp-20260805-01-content-pointer-audio-signal-is-keys-only--decoder-is-audio-blind).

**Second defect.** `ArModelConfig.legacy_inverted_attention_masks` defaults to **`True`**. [`configs/ar/tide_overfit_content_pointer.json`](../../configs/ar/tide_overfit_content_pointer.json) never sets it false, so the tide content-pointer run that “proved” the head ([EXP-20260804-06](EXPERIMENT_LOG.md#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate)) trained with **inverted** cross/self-attn masks (`valid_region_true_frac = 0`). Residual patch embeddings still reach `pointer_key`, which is why the pointer ablation could pass with a deaf decoder. R2 configs correctly set the flag false — and the decoder is *still* deaf — so mask polarity alone does not fix multi-song transfer; it does invalidate the tide proof as decoder grounding.

**Why prior notes misled.** [NOTE-20260804-04](#note-20260804-04-r2-content-pointer-fails-on-val-by-generalization-not-wiring--and-the-val-ablation-gate-lies-at-the-timing-floor) treated train pointer PASS as evidence the model reads audio. The standing gate’s token criterion already failed on that same train ablation (`same_token_as_matched` ≈ **0.996** under shuffle) and was under-weighted. Regularization ([NOTE-20260804-05](#note-20260804-05-no-es-120ep-confirms-val-timing-is-stuck-at-ep-16--not-an-early-stop-artifact)) is the wrong next lever.

**Coupled gaps (code inspection + same ablations).**
1. **PE still in “content” keys** — `pointer_key` projects post-`enc_pos` encoder memory, so absolute patch index remains in the key space; the content head is an incomplete fix for index-head blindness.
2. **Monotonic pointer never implemented** — [AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md) §7 specifies `patch_idx ≥ patch_idx_prev`; repo has no monotonic mask in train/decode (free `argmax` over all patches).
3. **Hard-argmax time loss** — with `use_soft_pointer_time: false`, `lambda_time` does not train the pointer (only residual); pointer learns from sparse CE alone.
4. Ruled out as root cause of this failure: val metric mismatch, wrong ES epoch, poisoned val MERT/`chart_index`, train/val loader fork.

**Next.**
1. Default `legacy_inverted_attention_masks` to **false** for new models; fix tide content-pointer config; re-verify tide with a **query/decoder** sensitivity gate.
2. Extend `audio_ablation_ar_onset.py --gate` so pointer PASS also requires `cos_query` (or token/`same_token`) to collapse under shuffle — keys-only collapse must not pass.
3. Force decoder audio + PE-free (or pre-`enc_pos`) pointer keys; add monotonic pointer mask per design; consider soft/local pointer loss.
4. Hygiene: ladder export via `_latest_best_checkpoint_path` over shared `callbacks/ar/ladder` can pick the wrong run’s `best.keras` — not the cause of EXP-07/08, but fix when touching export.

**Related:** [EXP-20260805-01](EXPERIMENT_LOG.md#exp-20260805-01-content-pointer-audio-signal-is-keys-only--decoder-is-audio-blind) · [EXP-20260804-06](EXPERIMENT_LOG.md#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate) · [EXP-20260804-08](EXPERIMENT_LOG.md#exp-20260804-08-r2-content-pointer-val-transfer-diagnosis--generalization-not-wiring) · [NOTE-20260716-01](#note-20260716-01-ar-attention-mask-semantics-were-inverted)

## Session 2026-08-04 — what ladder scaling breaks

### NOTE-20260804-05: No-ES 120ep confirms val timing is stuck at ep 16 — not an early-stop artifact

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 23:00:00 |
| **Topic** | Ruling out early stopping as the cause of ep-16 checkpoint selection |

**Verdict.** Disabling early stopping and training content-pointer R2 to **120** epochs changed nothing on val: best `timing_match_teacher` remains **0.0022 @ ep 16**; offline val timing mean is **bit-identical** to the ES run ([EXP-20260804-09](EXPERIMENT_LOG.md#exp-20260804-09-no-es-120ep--val-timing-never-improves-after-ep-16)). Train timing climbs to **51%** by ep 120 while val falls to **0.15%**.

**Next.** ES patience and epoch budget are ruled out. Change the recipe: dropout, LR schedule, weight decay, or checkpoint on `val_aux_f1_hungarian` (timing metric peaks at noise floor).

### NOTE-20260804-04: R2 content-pointer fails on val by generalization, not wiring — and the val ablation gate lies at the timing floor

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 22:15:00 |
| **Topic** | Why tide-fixed pointer still has zero val skill; how to read the ablation gate on a broken checkpoint |

**Verdict.** The content pointer is **not** still audio-blind at multi-song scale. On **train** songs, corrupting `mert_patches` collapses teacher timing from **3.6%** to **0.08%** under shuffle ([EXP-20260804-08](EXPERIMENT_LOG.md#exp-20260804-08-r2-content-pointer-val-transfer-diagnosis--generalization-not-wiring)). On **val**, matched timing is already **0.10%**, so shuffle at **0.03%** fails the gate trivially — the model is broken, not blind.

**Train/val gap.** Offline per-song eval on the ep-**16** checkpoint: train timing mean **2.5%** (best song **11%**), val mean **0.2%** (best **0.7%**). Training log: val `timing_match_teacher` peaked @ ep **16** (**0.22%**) and never improved while train climbed to **42%** by ep **66**. ES restore is correct; there is no saved ep-**66** weight to compare.

**Implication.** Scheduled sampling and scale were **downstream** of a model that never transferred pointer learning to val. Next lever is **training dynamics** (early val peak, overfit, regularization) — not SS by default. Run train-split ablation (or per-song val probes) whenever val timing is near zero; do not treat val gate FAIL alone as proof of audio-blindness.

### NOTE-20260804-03: The pointer head is absolute-index classification, so the model never had to hear the audio

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 20:32:40 |
| **Topic** | Root cause behind the chance-floor result, and why every gate missed it |

**Verdict.** The AR onset model does not use the audio. Corrupting **only** `mert_patches` — leaving the decoder prefix and every target untouched — changes essentially nothing: R2 val F1 goes **0.1886 → 0.1885** with all features zeroed, and the argmax patch is unchanged on **99.96%** of onset steps. Numbers: [EXP-20260804-05](EXPERIMENT_LOG.md#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer).

**The bug.** `models.py:358` builds the pointer as

```python
pointer_logits = keras.layers.Dense(max_patches, name="pointer_logits")(decoder)
```

Logit *k* is a learned output unit standing for **absolute patch index k**. It is never scored against `memory[k]`. Three consequences follow directly:

- **Not content-addressed.** Nothing in the head's structure ties logit *k* to what patch *k* sounds like. Audio can only reach the decision indirectly, by cross-attention shifting the decoder state, and measurably it does not.
- **Cannot length-generalize.** Index 1523 means a different absolute time in every song. A 90-second song and a 200-second song share no output units in any meaningful way, so nothing transfers across songs of different length.
- **Wasteful and sparsely trained.** **1,443,750** parameters (`max_patches` 3750 × `d_model` 384), where each output unit only receives gradient when an onset happens to land at that exact index.

**Why every gate missed it.** A single-song overfit is fully determined by the teacher-forced prefix, so an audio-blind model can score a perfect 1.0. The tide champion — the **PASS** that gated the whole AR stack — scores an identical **0.9984** with the audio **reversed**, **shuffled**, or **swapped for another song**, and `patch_wrong` is **0.0000** in all of them including all-zero input. The overfit gate cannot distinguish a model that hears from one that does not, so it never did.

**How this closes the loop.** [NOTE-20260804-02](#note-20260804-02-the-ladder-was-climbing-a-chance-floor--scale-was-never-the-binding-problem) established that no rung beat an audio-blind baseline. This says why: the model *is* an audio-blind baseline. The two findings are the same fact measured from opposite ends, which is the strongest confirmation available without a retrain.

**Fix — landed 2026-08-04.** `model.pointer_head: content` computes `logits[k] = q(dec_state) · k(memory[k]) / √d` over encoder memory, still masked by `patch_mask`. Tide now scores **0.9921** on real audio and **0.18** under shuffled audio — the gate is no longer passable blind ([EXP-20260804-06](EXPERIMENT_LOG.md#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate)). Head cost falls **1,443,750 → 295,680** params and stops scaling with `max_patches`. `pointer_head: index` still exists to rebuild old runs, and the inference builder detects the head from the loaded model rather than the config, so existing checkpoints keep working.

**Still to do.** Promote the corruption ablation to a standing gate in `scripts/`; ablate the **token** head, which has not been checked; and re-run a ladder rung to get the first AR number with real skill over its null.

**What this retires.** Every open AR item framed as data, termination, or regularization — EOS weighting, scheduled sampling, density variants, R4–R5, dropout/capacity tuning — was downstream of a model that ignores its input. None of them can be evaluated until the head is fixed.

### NOTE-20260804-02: The ladder was climbing a chance floor — scale was never the binding problem

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 20:05:45 |
| **Topic** | Why "more data does not help" was the wrong reading, and what to fix instead |

**Verdict.** Hungarian F1 @ 20 ms on dense charts has a chance floor of **0.225–0.336**, and **no AR rung has ever cleared it**. Every conclusion drawn from rung-to-rung F1 deltas — the plateau, the non-monotonicity, the density ranking, the EOS diagnosis, the choice of a "generation champion" — was drawn from noise above a floor nobody had measured. Full numbers: [EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke).

**Why the floor is high.** The frozen val set averages **5.52** onsets/sec (mean inter-onset interval **181 ms**). A ±20 ms match window therefore covers ~22% of the timeline, so a predictor that emits the right *number* of onsets and ignores the audio entirely already scores ~0.23, and one that reuses the song's own inter-onset-interval distribution scores ~0.34.

**Model vs its own floor, at matched prediction count:**

| Variant | pred/GT | Reported F1 | Uniform | Metronome | IOI-shuffle | Verdict |
| ------- | ------- | ----------- | ------- | --------- | ----------- | ------- |
| Teacher, any rung | 1.00 | 0.178–0.227 | 0.225 | 0.275 | 0.336 | below all |
| R2 bare free-run | 0.36 | 0.132 | 0.128 | 0.154 | 0.182 | below all |
| R2+meter free-run | 0.82 | 0.234 | 0.208 | 0.250 | 0.295 | beats uniform only |
| R2+onset_density free-run | 0.90 | **0.263** | 0.217 | 0.261 | 0.313 | ties metronome |
| R3+density free-run | 0.16 | 0.034 | 0.072 | 0.079 | 0.084 | below all |
| R3 `min_onset_tokens=200` | 1.05 | 0.200 | 0.228 | 0.282 | 0.336 | below all |

**What the density work actually did.** Free-run F1 moves with `pred/GT` along the null curve and nowhere else. Conditioning taught the decoder *how many* onsets to emit, which is worth real F1 on a count-sensitive metric, and taught it nothing about *when*. [NOTE-20260804-01](#note-20260804-01-scale-up-fails-on-eos-termination-timing-is-recoverable-once-length-is-forced)'s "timing content after step 15 exists" reads the same way: forcing length past early EOS moved `pred/GT` from 0.02 to 1.05, and the resulting **0.200** is below the **0.239** floor at that count.

**Three independent confirmations that do not depend on the chance argument:**

1. Ordered `timing_match` (floor ≈ **0**, nulls 0.000–0.029) measures **0.0026** on val — the model is at the floor of the discriminative metric too.
2. Val `pointer_loss` is **15.6–18.6** nats where uniform guessing over ~810 patches costs **ln(810) ≈ 6.7**. The pointer is *confidently wrong* on held-out audio, not merely uninformed.
3. `n_patch_wrong` is **0.99–1.00** on all 50 val songs, with teacher-forced median timing error **1.1–73 s** against a 20 ms tolerance.

**The real failure is generalization, not scale.** At each run's best epoch, train/val Hungarian F1 is **0.746/0.227** (R2), **0.529/0.199** (R3), **0.736/0.227** (R2+density), **0.539/0.206** (R3+density); train token accuracy **0.98** vs val **0.37**; train `pointer_loss` **0.01–0.04** vs val **~16**. The model memorizes training songs and transfers nothing. Adding rows to a recipe with zero transfer cannot produce a trend, which is exactly what the ladder observed.

**Fixes landed this session:**

| Fix | What |
| --- | ---- |
| Floor is now measured in-harness | `stepcovnet.onset_null_baseline` — audio-blind baselines at matched prediction count + `skill_over_null`; wired into `eval_ar_onset_offline.py`, which now prints `Null F1 @ matched count` and `Skill over strongest null` for both teacher and free-run blocks, and emits `null_baseline` in the JSON |
| A low-floor metric exists on multi-song runs | `timing_match_teacher` was created only when `overfit_one_song` was true, so multi-song runs had **no** near-zero-floor metric to select on. Now always published; `checkpoint_metric: val_timing_match_teacher` already resolves |

**Action order:**

1. **Re-score, do not retrain.** Run the four existing checkpoints through the updated harness for skill-over-null and `timing_match`.
2. **Change selection.** Ladder configs move to `checkpoint_metric: val_timing_match_teacher`; `val_aux_f1_hungarian` never exceeded its own floor, so it was selecting on noise.
3. **Change the promotion rule.** A rung counts only if val skill over the strongest null is positive — beating the previous rung is not sufficient.
4. **Then attack generalization** (regularization, capacity, augmentation, held-out-song curriculum) — the 0.75/0.23 train/val split, not row count.
5. **Hold** R4–R5, EOS weighting, and further density variants.

**Related:** [EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke) · [NOTE-20260804-01](#note-20260804-01-scale-up-fails-on-eos-termination-timing-is-recoverable-once-length-is-forced) · [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md) · [ONSET_METRICS.md](ONSET_METRICS.md)

### NOTE-20260804-01: Scale-up fails on EOS termination; timing is recoverable once length is forced

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 12:25:23 |
| **Topic** | Why R3 (200t) does not climb over R2 on free-run, and what to fix next |

**Superseded by [NOTE-20260804-02](#note-20260804-02-the-ladder-was-climbing-a-chance-floor--scale-was-never-the-binding-problem).** Every F1 below sits at or under an audio-blind floor at its own prediction count, so the termination story reads a count effect as a timing effect. The length-force "recovery" (**0.200**) is below the **0.239** null at that count. The table stays for the record; do not act on it.

**Verdict.** The ladder is not failing because more data destroys timing skill. It fails because free-run **termination** collapses at 200t, and density conditioning that fixed R2 does not transfer.

**Proof (recomputed from logged decodes + existing 5-song probes):**

| Variant | Teacher F1 | Free-run F1 | pred/GT | Length signature |
| ------- | ---------- | ----------- | ------- | ---------------- |
| R2 bare | 0.227 | 0.132 | 0.356 | **252×50**, corr(pred,GT)=0 |
| R3 bare | 0.199 | 0.003 | 0.021 | **15×50**, corr=0 |
| R2+onset_density | 0.227 | **0.263** | 0.900 | 9 modes, corr=**0.78** |
| R3+onset_density | 0.206 | **0.034** | 0.159 | **15×21 + 19×15** (36/50 early), corr=**−0.42** |

Bare R3 length-control probe (`_tmp/ladder_debug/r3_probe*.json`, 5 songs matching bare stop@15):

| Control | Free-run F1 | Stop |
| ------- | ----------- | ---- |
| none | 0.0007 | 15×5 |
| `eos_logit_bias=+3` | 0.0007 | 15×5 |
| `min_onset_tokens=200` | **0.200** | 603×5 |

Forcing past early EOS recovers ~teacher-scale free-run on that subset. Content after step 15 is usable; the model just refuses to emit it.

**Secondary contributors (not the binding ceiling at current F1):**

- Multi-chart conflict rows: **4%** (R2) → **15.5%** (R3), spreads up to **659** steps — candidate poison for EOS, still open ([NOTE-20260803-01](#note-20260803-01-difficulty-is-unconditioned-and-unfiltered-but-it-is-not-what-caps-the-ladder)).
- `eos_token_weight_scale` under `token_class_weight: none` is still a no-op ([NOTE-20260724-01](#note-20260724-01-eos_token_weight_scale-is-a-no-op-under-token_class_weight-none)) — cannot down-weight EOS in CE today.
- Teacher F1 is also non-monotonic (R3 **0.199** &lt; R2 **0.227**), so scale-up is imperfect even under teacher forcing, but free-run is the gate that actually breaks usability.

**Action order:**

1. **P0 (no retrain):** full-val R3+onset_density decode with density-derived `min_onset_tokens` — extend the proven 5-song probe.
2. **P1:** make EOS scale work under `scheme=none`, retrain R3+onset_density with `eos_token_weight_scale &lt; 1`.
3. **P2:** one-chart-per-audio R3′ ablation to test conflict poisoning.
4. **Hold:** no R4; generation champion stays R2+onset_density (**0.263**).

**Related:** [EXP-20260804-01](EXPERIMENT_LOG.md#exp-20260804-01-r3--onset_density-lifts-teacher-but-free-run-still-collapses) · [EXP-20260804-02](EXPERIMENT_LOG.md#exp-20260804-02-r3-early-eos-is-the-scaling-failure--length-force-recovers-free-run) · [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md)

## Session 2026-08-03 — ladder difficulty composition and the audio-only ceiling

### NOTE-20260803-01: Difficulty is unconditioned and unfiltered, but it is not what caps the ladder

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-03 01:31:44 |
| **Topic** | Which chart difficulties the `ladder_v1` manifests select, whether the mix is consistent across splits and rungs, and whether the model's lack of a difficulty input is blocking progress |

**The model has no difficulty input.** `ArOnsetTrainingModel._model_inputs` passes only `mert_patches`, `patch_mask`, `decoder_input_ids`, `decoder_mask`; `onset_ar/datasets.py` never mentions `difficulty` or `meter`. Sampling is over chart **rows**, with no difficulty filter, so the same audio can appear under several charts with different onset counts and one prediction is scored against all of them.

**Composition (audit of all four `ladder_v1` manifests):**

| Split | beginner | easy | medium | hard | challenge | edit | rows / songs |
| ----- | -------- | ---- | ------ | ---- | --------- | ---- | ------------ |
| Val (frozen, identical at every rung) | 28% | 4% | 2% | 8% | 56% | 2% | 50 / 39 |
| R1 train | 20% | — | — | — | 80% | — | 10 / 10 |
| R2 train | 16% | 4% | 2% | 2% | 76% | — | 50 / 49 |
| R3 train | 26% | 4% | 2% | 4% | 63% | 0.5% | 200 / 183 |
| R4 train | 27% | 5% | 4% | 4% | 60% | 0.3% | 300 / 270 |

**Protocol checks pass:** zero train/val **song** overlap at every rung (no leakage), train sets properly nested R1 ⊂ R2 ⊂ R3 ⊂ R4, and the val row set is byte-identical across all four manifests.

**Protocol gap:** rungs are **not** matched on difficulty mix — the challenge share falls 80% → 76% → 63% → 60% as the ladder climbs, so "train size is the only variable" is not strictly true. This cuts *against* the observed non-monotonicity rather than explaining it: R3's mix is closer to val's than R2's is, yet R3 scored **0.199** vs R2's **0.227** ([EXP-20260802-02](EXPERIMENT_LOG.md#exp-20260802-02-ladder-r3--200-train-rows-does-not-beat-r2-on-frozen-val)).

**Audio-only ceiling = 0.9574.** For each val song, take the single onset sequence that maximizes matches across that song's charts, then micro-average at 20 ms tolerance: **tp 35280 / fp 2982 / fn 159 → F1 0.9574**. Multi-chart ambiguity therefore costs at most ~4 F1 points, because charts of one song overlap heavily in time. Against a current **0.227** teacher / **0.132** free-run, difficulty ambiguity is **not** the binding constraint — it becomes one above ~0.9.

**Difficulty labels are partly wrong**, so "normalize by difficulty label" would launder bad metadata:

| Song | Charts (onsets) | Pairwise F1 @ 20 ms |
| ---- | --------------- | ------------------- |
| `see_the_lights.ogg` | challenge 624, medium 351, easy 198, beginner 85 | 0.24–0.70 — sane ladder |
| `idola.ogg` | challenge 880, beginner 138 | 0.27 |
| `started.mp3` | **beginner 735**, challenge 724, hard 524 | 0.81–0.98 |
| `act_000000.mp3` | challenge 979, **beginner 980** | 1.00 |

`started.mp3` lists its beginner slot at **meter 12**, equal to its challenge — the numeric `meter` corroborates the mislabeling, making it a more trustworthy signal than the difficulty string.

**Secondary effects worth remembering.** Val rows are unevenly weighted by audio: `see_the_lights` contributes 8% of the metric from one audio file (4 charts) while a single-chart song contributes 2%. Conflicting-target rows in **training** grow with rung size — 4% of rows at R2, 15% at R3, 18% at R4 — a candidate contributor to R3 < R2, though the high pairwise agreement above suggests the conflict is mild.

**Reading / options (no action taken):**

| Option | Effect |
| ------ | ------ |
| Deduplicate to one chart per song when rebuilding manifests | Removes ambiguity and the uneven audio weighting; costs 11 val rows; requires a fresh ladder (breaks comparability with R1–R3) |
| Condition on **measured onset density** (or numeric `meter`) via encoder input or BOS token | The substantive fix. Sidesteps unreliable labels, makes multi-chart data an asset (3–4× rows), gives difficulty control at generation time, and supplies the **length prior** the model currently lacks |
| Filter to one difficulty tier | Rejected — labels are unreliable, so this does not guarantee homogeneity |

Density conditioning is the option that also attacks the open free-run pathology: R2 emits exactly **252** onsets for all 50 val songs while R3 stops at **15** ([EXP-20260802-04](EXPERIMENT_LOG.md#exp-20260802-04-ladder-r2-vs-r3-offline-val-free-run-compare)). "How dense should this chart be" is the same missing information as difficulty.

**Related:** [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md) § 3 · [EXP-20260802-04](EXPERIMENT_LOG.md#exp-20260802-04-ladder-r2-vs-r3-offline-val-free-run-compare) · [NOTE-20260724-02](#note-20260724-02-hypotheses-eliminated-for-multi-song-free-run-under-generation)

## Session 2026-07-25 — AR 50t/50v rebuild gap triage

### NOTE-20260725-01: The 50t/50v rebuild gap is val-side, not recipe or early stopping

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-25 22:35:29 |
| **Topic** | Why [EXP-20260724-04](EXPERIMENT_LOG.md#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction) reached ~10× worse val Hungarian F1 than [EXP-20260724-01](EXPERIMENT_LOG.md#exp-20260724-01-ar-corrected-mask-50t50v-500-ep-scale-up) on the same config |

Desk analysis of committed artifacts only — no new runs. Three candidate explanations are removed or sharpened:

| Candidate | Verdict |
| --------- | ------- |
| Recipe drift when `smoke_50t_50v.json` was renamed to `scale_50t_50v.json` | **Eliminated** — `git show c282002 -- configs/ar/` shows the rename changed only `early_stopping_patience: 25` and the two output paths; every hyperparameter is identical |
| Early stopping truncated a run whose F1 was still climbing | **Insufficient** — val F1 was indeed still rising when ES fired at ep **131** (0.0012 @ ep10 → 0.0128 @ ep129), but the gap exists at **matched** epochs: ep **50** gives **0.0058** here vs **0.126** in the logged 50-ep run, with comparable `val_loss` (**24.8** vs **21.0**) |
| Training-side degradation (features, code, hardware affecting fit) | **Narrowed to the val side** — at ep 131 the rebuild has train `aux_f1_hungarian` **0.0499** / train `token_accuracy` **0.4691** against val `token_accuracy` **0.0925**. The logged run reported val token accuracy **0.43** at ep 65, i.e. roughly the rebuild's *train* level. The local model fits its train songs at a plausible rate and then fails to transfer at all |

**Reading:** val loss tracks the logged run while val F1 and val token accuracy do not, which is the signature of a val-split or feature problem rather than an optimization problem. Both the subset sample and the MERT features were regenerated locally (91 unique audio, `--device=cuda`), and the original subset index was overwritten, so neither is byte-comparable to the logged run.

**Cheapest discriminating test:** re-extract MERT for the tide song with the current local pipeline and diff numerically against the known-good features behind the champion overfit.

**Result — feature drift eliminated.** Run the same evening as [EXP-20260725-01](EXPERIMENT_LOG.md#exp-20260725-01-ladder-r0--mert-extraction-is-bit-identical-tide-champion-artifact-was-overwritten): the re-extracted tide features are **bit-for-bit identical** (`np.array_equal` True, max abs diff **0.0**), and a real checkpoint decodes at **98.9%** teacher with correct EOS placement. Local extraction is not the cause of the rebuild gap, so the remaining candidates are the val **subset** itself ([NOTE-20260725-02](#note-20260725-02-subset-sampling-gives-every-train-size-a-different-val-set)) and per-song val-split composition.

**Related:** [EXP-20260724-04](EXPERIMENT_LOG.md#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction) · `logs/ar_scale_50t_50v_rebuild.log` · `callbacks/ar/scale_50t_50v_corrected_masks/`

### NOTE-20260725-02: Subset sampling gives every train size a different val set

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-25 22:35:29 |
| **Topic** | Whether a small-to-large scaling ladder can be trusted with the current subset builder |

**Discovery:** `training_index.build_training_index_subset` draws both splits from a single generator, train first:

```python
rng = random.Random(seed)
sampled = rng.sample(train_pool, train_rows) + rng.sample(val_pool, val_rows)
```

`rng.sample` consumes generator state proportional to the draw, so changing `train_rows` shifts the **val** draw as well. Reproduced against a 1755/186 pool with `train_rows` ∈ {10, 50, 200, 300} and identical `seed=42`, `val_rows=50`: four different val samples.

**Why it matters:** every AR scale-up rung logged so far was scored on a **different** held-out set. The apparent plateau across 10 / 50 / 200 rows (**0.11** / **0.126** / **0.120** val teacher F1) may be val-set variance rather than a data-scale ceiling, and cannot be read either way. It also compounds [NOTE-20260725-01](#note-20260725-01-the-50t50v-rebuild-gap-is-val-side-not-recipe-or-early-stopping): the rebuild's val songs were never the logged run's val songs.

**Second-order finding:** the source manifest is gitignored and has drifted — on disk **1755/186** rows, created 2026-07-03, against the **1745/197** recorded in [EXP-20260623-02](EXPERIMENT_LOG.md#exp-20260623-02-p8-trainval-manifest-on-full-final_data). A fixed seed cannot reproduce a subset whose source pool changed.

**Proposed fix (not yet applied):** independent generator per split plus shuffle-then-take, which both stabilizes val and makes train sets **nest** as the ladder climbs; record the source SHA-256 in the subset summary. Protocol and rungs: [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md).

**Related:** [`src/stepcovnet/dataset_prep/training_index.py`](../../src/stepcovnet/dataset_prep/training_index.py) L619–620 · [`scripts/build_training_index_subset.py`](../../scripts/build_training_index_subset.py)

---

## Session 2026-07-24 — AR free-run length collapse diagnosis

### NOTE-20260724-01: `eos_token_weight_scale` is a no-op under `token_class_weight: none`

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-24 22:40:39 |
| **Topic** | EOS loss weighting while diagnosing multi-song free-run under-generation |

**Discovery:** `losses.build_token_class_weights_numpy` returns `None` immediately when `scheme == "none"`, **before** `eos_token_weight_scale` is applied. Every config that pairs `token_class_weight: none` with `eos_token_weight_scale != 1.0` therefore trains with **uniform** token CE, including the champion [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json) (`0.2`) and its `v8` / `v9` variants.

**Correction to prior notes:** the recipe table in [NOTE-20260703-01](#note-20260703-01-class-weights-need-co-tuned-loss-recipe-deferred) lists `eos_token_weight_scale` **0.2** as a champion-v8 differentiator versus the historical gate-tide bundle. That row never took effect — v8 and the historical recipe both trained with unscaled EOS. The historical recipe used `inverse_freq`, so its EOS scale of `1.0` was applied but changed nothing either.

**Why it matters now:** [EXP-20260724-02](EXPERIMENT_LOG.md#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode) shows multi-song free-run collapsing through early EOS (~**70** predictions/song vs ~**700** GT). Down-weighting EOS in the token CE is an obvious first training-side lever, and the config surface implies it is already available. It is not. Enabling it currently requires turning on full inverse-frequency class weighting, which [EXP-20260703-01](EXPERIMENT_LOG.md#exp-20260703-01-ar-tide-token-class-weight-ablation-champion-recipe) showed hurts free-run.

**Proposed fix (not yet applied):** apply the EOS scale for **any** scheme by returning a ones-vector with `EOS_ID` scaled when `scheme == "none"`, and simultaneously set `eos_token_weight_scale: 1.0` in every config that currently relies on the value being ignored, so past results stay reproducible. Touches a graduated champion config, so it needs an explicit decision.

**Related:** [`src/stepcovnet/onset_ar/losses.py`](../../src/stepcovnet/onset_ar/losses.py) · [EXP-20260724-03](EXPERIMENT_LOG.md#exp-20260724-03-ar-decode-length-control--eos-trace-diagnostics)

### NOTE-20260724-02: Hypotheses eliminated for multi-song free-run under-generation

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-24 22:40:39 |
| **Topic** | Source review of the AR decode/target path before retraining |

Reading `onset_ar/{datasets,targets,losses,inference,kv_decode}.py` against the [EXP-20260724-02](EXPERIMENT_LOG.md#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode) failure removes three plausible causes:

| Hypothesis | Verdict |
| ---------- | ------- |
| Charts over `max_steps_per_chart` are truncated and teach `<EOS>` at arbitrary positions | **Eliminated** — `_filter_valid_ar_samples` **skips** charts that exceed the cap (`charts.chart_exceeds_step_cap`); no truncated sequence reaches training |
| Audio capped at `max_audio_seconds` while targets keep later onsets | **Eliminated** — `load_ar_sample` applies `event_targets.clip_times_to_duration(raw_times, duration_sec)` after truncation |
| `<EOS>` is over-represented in the token CE | **Eliminated as over-representation** — one `<EOS>` per ~700-token sequence; if anything it is rare |

**Remaining live hypothesis:** exposure bias. `scale_200t_50v.json` trains with `scheduled_sampling_max_p: 0.0`, so the decoder only ever conditions on ground-truth prefixes. With a `delta_bucketed` (relative) token scheme, free-run drift compounds, and `<EOS>` becomes the argmax once the prefix leaves the training manifold. Consistent with teacher F1 (**0.120**) transferring while free-run F1 (**0.036**) does not.

**Contrast measured on tide** ([EXP-20260724-03](EXPERIMENT_LOG.md#exp-20260724-03-ar-decode-length-control--eos-trace-diagnostics)): a well-fit single-song checkpoint holds EOS probability at ~**0.0017** mean across 635 steps and first crosses 0.5 at step **634** — the correct end. The new `eos_trace` block makes this directly comparable against a collapsing multi-song checkpoint.

---

## Session 2026-07-16 — AR training correctness and throughput

### NOTE-20260716-01: AR attention-mask semantics were inverted

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-16 01:15:49 |
| **Topic** | Keras `MultiHeadAttention` mask semantics during dynamic-padding work |

**Context:** Dynamic padding for [EXP-20260716-01](EXPERIMENT_LOG.md#exp-20260716-01-ar-validation-aggregation--dynamic-length-bucketing) removes most padded positions. The first dynamic run changed loss more than expected, prompting inspection of Keras 3.13.2 mask handling.

**Discovery:** Keras softmax **keeps** positions where `attention_mask=True` and zeros positions where it is false. The AR `PairwiseValidMask`, `CrossAttentionMask`, and `DecoderSelfAttentionMask` implemented the opposite contract: valid pairs were false and padded/future pairs true. Historical models therefore learned with attention directed at padding (while residual paths still carried inputs).

**Compatibility decision:**

- New smoke/scale-up models set `legacy_inverted_attention_masks: false` and use correct keep-valid masks.
- `ArModelConfig` defaults the compatibility flag to `true`, so existing JSON recipes retain their historical behavior unless explicitly migrated.
- Serialized historical mask layers omit `keep_valid`; its default remains `false`, preserving old checkpoint behavior when loaded.
- Dynamic padding is opt-in (`dataset.dynamic_padding: true`); historical configs remain fixed-padding by default.

**Implication:** Tide champion results remain valid as historical overfit measurements, but they do **not** demonstrate useful audio cross-attention. Do not silently compare a corrected-mask scale-up against champion training curves as if only padding changed. The corrected stack needs its own overfit and multi-song validation gates.

**Related:** [EXP-20260716-01](EXPERIMENT_LOG.md#exp-20260716-01-ar-validation-aggregation--dynamic-length-bucketing) · [`src/stepcovnet/onset_ar/models.py`](../../src/stepcovnet/onset_ar/models.py) · [`configs/ar/smoke.json`](../../configs/ar/smoke.json)

---

## Session 2026-07-03 — AR token class weights on champion recipe

### NOTE-20260703-01: Class weights need co-tuned loss recipe (deferred)

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-07-03 01:48:00                                                   |
| **Topic**     | `token_class_weight` on champion v8 vs historical gate-tide bundle      |

**Context:** [EXP-20260703-01](EXPERIMENT_LOG.md#exp-20260703-01-ar-tide-token-class-weight-ablation-champion-recipe) swapped `inverse_freq` / `inverse_sqrt_freq` onto the **v8 champion** stack (`lambda_residual=30`, `d_model=384`, …) only. Teacher timing reached **634/634**; free-run decode failed (**≤360/634**). Easy read: “class weights don’t work.”

**Nuance:** `inverse_freq` **did** fix majority-token collapse in the **older** gate-tide recipe ([NOTE-20260627-02](DISCUSSION_NOTES.md#note-20260627-02-gate-tide-overfit-resolution)) as part of a **bundle**, not in isolation:

| Parameter | Historical gate-tide (`v1` + `inverse_freq`) | Champion v8 (PASS) |
| --------- | ---------------------------------------------- | ------------------ |
| `token_class_weight` | `inverse_freq` | `none` |
| `lambda_residual` | **5.0** | **30.0** |
| `lambda_time_ramp_epochs` | **100** | **0** |
| `d_model` | 256 | 384 |
| `eos_token_weight_scale` | 1.0 | **0.2** |

High `lambda_residual` lets pointer+residual hit perfect **teacher** timing while token argmax stays weak — so class weights on the champion stack are **not a fair single-knob test**.

**Open follow-ups (later, not blocking scale-up):**

1. Re-run class weights with a **matched low-residual recipe** (e.g. `lambda_residual=5`, `lambda_time_ramp_epochs=100`, checkpoint on `val_gate_teacher`, judge `--ar_decode`).
2. **`inverse_sqrt_freq`** or **capped** inverse weights (less aggressive than `inverse_freq`).
3. **Manifest-derived** weights (multi-song freq) instead of single-batch tide histogram.
4. **Focal token CE** — mentioned in [NOTE-20260627-01](DISCUSSION_NOTES.md#note-20260627-01-gate-tide-overfit-plateau-and-open-hypotheses); not implemented.

**Decision (for now):** Champion [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json) stays **`token_class_weight: none`**. Proceed to **`final-data-mert`** / **`gate-val-vs-dense`**. Revisit class weights only with a deliberate co-tuned recipe ablation.

**Related:** [`v9_inverse_freq.json`](../../configs/ar/versions/tide_overfit/v9_inverse_freq.json) · [`v9_inverse_sqrt_freq.json`](../../configs/ar/versions/tide_overfit/v9_inverse_sqrt_freq.json) · [AR_ONSET_DESIGN.md §11](AR_ONSET_DESIGN.md#11-decision-registry-locked-2026-06)

---

## Session 2026-07-01 — AR MERT input normalization A/B

### NOTE-20260701-01: AR tide overfit — reject per-song MERT z-score

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-07-01 01:00:00                                                   |
| **Topic**     | Raw vs dense-style `normalize_onset_spectrogram` on MERT before patching |

**Context:** Dense onset applies per-dimension z-score across time within each song (`datasets.normalize_onset_spectrogram`). AR tide gates used **raw** hidden states from `.mert.npy`. Question: would the same normalization speed tide perfect overfit (`val_overfit_gate` → 1.0)?

**Experiment:** [EXP-20260630-03](EXPERIMENT_LOG.md#exp-20260630-03-ar-tide-mert-normalization-ab). Matched scratch tide recipe (`d_model=384`, `lr=1e-4`, `lambda_residual=30`, 400 ep, seed 42); only `dataset.normalize_mert_features` toggled. **`lambda_incremental_consistency=0`** in both arms (RTX 3070 Ti OOM at 0.01 on epoch 2 — champion iter175 used 0.01; A/B isolates input norm only).

**Results** (`val_overfit_gate` = `min(val_token_accuracy, val_ordered_onset_match)`):

| Milestone | Raw MERT | Normalized MERT |
| --------- | -------- | ---------------- |
| ≥ 0.90 | ep 154 | ep 152 |
| ≥ 0.95 | ep 210 | ep 209 |
| ≥ 0.99 | ep 289 | ep 282 |
| ≥ 0.999 | ep **399** | **never** |
| ≥ 0.9999 (perfect) | ep **399** (gate **1.0**) | **never** (best **0.9984** @ ep 313) |

Normalized was marginally faster early but **plateaued below perfect**; raw reached gate **1.0** at epoch 399 (same ballpark as iter175 @ ~390 without norm).

**Decision:** AR **default = raw MERT** after hop-grid resample (`normalize_mert_features: false`). Do **not** adopt dense per-song z-score for AR training or decode unless a new ablation reopens it. Optional flag remains in `ArDatasetConfig` for experiments. Dense track keeps its own norm path — train/eval must stay consistent there ([NOTE-20260606-17](DISCUSSION_NOTES.md#note-20260606-17-multi-song-eval-missing-normalization)).

**Related:** `mert-input-norm` in [AR_ONSET_DESIGN.md §11](AR_ONSET_DESIGN.md#11-decision-registry-locked-2026-06)

---

## Session 2026-06-30 — AR free-run eval reference

### NOTE-20260630-01: AR free-run primary vs `target_times`

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-30 19:30:00                                                   |
| **Topic**     | Align free-run ordered gate with training labels                      |

**Context:** iter175 scratch teacher **634/634** and free-run **634/634 tokens**, but free-run ordered scored vs raw `gt_times` read **633/634** while teacher ordered (vs `target_times`) read **634/634**. At onset index 318 the model predicts the same time in teacher and free-run; residual is ~19 ms vs `target_times` and ~26 ms vs raw chart (~7 ms hop-quantization gap). Overnight decode sweeps could not beat 633/634 vs raw chart.

**Decision:** On tide overfit, **primary** ordered match for **both** teacher-fed and free-run uses **`target_times`** (training patch+residual decode). **Aux:** `chart_ordered_onset_match` vs raw `gt_times`, Hungarian `event_f1` vs chart. Implementation: `scripts/eval_ar_onset_offline.py`, `run_exp.py` pass gate.

**Related:** [ONSET_METRICS.md](ONSET_METRICS.md), [NOTE-20260628-03](#note-20260628-03-tide-overfit-primary-metric--ordered-onset-match)

---

## Session 2026-06-28 — Tide overfit free-run bar

### NOTE-20260628-02: Tide overfit free-run bar **1.0**

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-28 15:00:45                                                   |
| **Topic**     | `gate-ar-decode` / perfect-overfit pass criterion on tide             |

**Context:** EXP-20260628-02 run2 reached offline AR F1 **0.978** with exact tokens; prior docs used a **0.95** free-run threshold inherited from exposure-bias intuition, not single-chart overfit.

**Decision:** On tide (one chart, 634 onsets), **free-run autoregressive decode event F1 must be 1.0** — same bar as teacher-fed overfit. Partial match is unacceptable when the training set is a single example the model must memorize.

**Implication:** Run1 (**~0.954**) and run2 (**0.978**) remain **fail** on `gate-ar-decode` / perfect-overfit until offline `--ar_decode` hits **634/634** ordered matches @ 20 ms (see [NOTE-20260628-03](#note-20260628-03-tide-overfit-primary-metric-ordered-onset-match)). `gate-val-vs-dense` on multi-song val keeps a separate, lower scoreboard bar.

**Related:** [AR_ONSET_DESIGN.md §10.1](AR_ONSET_DESIGN.md#101-experiment-gates-in-order), [EXP-20260628-02](EXPERIMENT_LOG.md#exp-20260628-02-ar-tide-perfect-overfit-val_overfit_gate)

---

### NOTE-20260628-03: Tide overfit primary metric — ordered onset match

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-28 20:00:00                                                   |
| **Topic**     | Align tide overfit gate with per-step timing truth                     |

**Context:** Teacher Hungarian `event_f1` could read **1.0** while per-step timing was **633/634** @ 20 ms (residual error ~25 ms on one onset). That is not chart-perfect on a single-song overfit.

**Decision:** On tide overfit, **primary** metrics are **`ordered_onset_match`** / **`timing_match`** (teacher) and **`ar_decode_ordered_onset_match`** (free-run, two-pass): ordered pairs with `|pred[i] − ref[i]| ≤ 20 ms` where **`ref` = `target_times`** (training labels); rate = **`n_matched / max(n_pred, n_ref)`**. Pass only at **rate 1.0** (634/634 when counts match). Hungarian `event_f1` vs raw chart remains **aux**. `val_overfit_gate` = `min(val_token_accuracy, val_ordered_onset_match)`. Free-run reference clarified in [NOTE-20260630-01](DISCUSSION_NOTES.md#note-20260630-01-ar-free-run-primary-vs-target_times). Full spec: [ONSET_METRICS.md](ONSET_METRICS.md).

**Related:** `src/stepcovnet/timing_match.py`, `src/stepcovnet/onset_ar/trainers.py` (`ArOrderedOnsetMatchMetric`), `scripts/eval_ar_onset_offline.py`

---

## Session 2026-06-28 — AR `gate-ar-decode` v2

### NOTE-20260628-01: `gate-ar-decode` v2 infra (eager AR-val + KV cache)

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-28 05:15:00                                                   |
| **Topic**     | Free-running validation speed + scheduled-sampling decode training    |

**Context:** After `gate-tide-overfit` pass, the PASS checkpoint (informal alias **`gate_v5`** → `models_wsl/ar/gate_tide_overfit/`) free-runs fail early-EOS (~12/634 onsets). v2 recipe warm-starts that checkpoint, ramps scheduled sampling, checkpoints on `val_ar_decode_event_f1` ([EXP-20260628-01](EXPERIMENT_LOG.md#exp-20260628-01-ar-gate-ar-decode-v2-wsl-150ep-warm-start-gate_v5)).

**Infra fixes (2026-06-28):**

1. **`ArDecodeValidationCallback`** — AR decode moved out of compiled `test_step` (was frozen by `tf.function`; `every_n_epochs` never skipped). Callback runs eagerly in `on_epoch_end`, writes `val_ar_decode_*` into logs before `ModelCheckpoint`.
2. **KV-cache decode** — `kv_decode.ArOnsetKvDecoder` incremental self-attn; encoder once per sequence. Default in `inference.decode_autoregressive_with_stats_numpy`. Prefix loop kept as `use_kv_cache=False`.
3. **Parity** — KV vs prefix logits can differ slightly (Keras MHA seq_len=1 vs full); metrics comparable, traces not bit-identical. Tests: `tests/onset_ar/kv_decode_test.py`.

**Run 1 (pre-restart):** Best `val_ar_decode_event_f1` **~0.50**; full-chart decode from ep 7; teacher-fed F1 degraded under token-only SS. Log archived: `logs/ar_tide_overfit_gate_decode_v2_run1.log`.

**Related:** [AR_ONSET_DESIGN.md §10.6](AR_ONSET_DESIGN.md#106-gate-ar-decode-notes-2026-06-28), `configs/ar/decode/v2.json`

---

## Session 2026-06-27 — AR `gate-tide-overfit` pass

### NOTE-20260627-02: `gate-tide-overfit` resolution

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-27 19:53:03                                                   |
| **Topic**     | AR tide overfit — training recipe that reaches teacher-fed F1 ≈ 1.0   |

**Context:** After EXP-20260627-02 failure and fix chain (EXP-20260627-03), **`gate-tide-overfit` passed** (EXP-20260627-04) with `val_event_onset_f1` **1.0** on tide.

**Root cause (confirmed):**

1. **Token collapse** — unweighted CE + majority token 83 (**305/635**) → argmax accuracy stuck at **0.4803** after one step; fixed with **`token_class_weight: inverse_freq`** and **`dropout_rate: 0`**.
2. **Soft vs hard decode** — F1 and early `time_loss` used soft expected patch; fixed with **`use_soft_pointer_time: false`** (argmax).
3. **Missing residual gradient** — with `lambda_time=0`, only pointer CE trained patch index; F1 uses `patch × duration + residual`. At F1 **~0.83**, debug showed **`n_patch_wrong: 0`**, **`n_patch_ok_timing_wrong: 103`**. Fixed with **`lambda_residual: 5.0`** (MSE on `target_residual_sec`).
4. **λ_time phasing** — immediate `lambda_time=1.0` destabilized early training; **`lambda_time_ramp_epochs: 100`** (linear 0→1) allowed pointer to learn first.

**Locked tide config:** champion [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json); history [`configs/ar/versions/tide_overfit/`](../../configs/ar/versions/tide_overfit/).

**Diagnostics:** `scripts/eval_ar_onset_offline.py` — per-onset patch vs residual error split.

**Next:** **`gate-ar-decode`** — scheduled sampling; free-running decode F1 **1.0** on tide (see [NOTE-20260628-02](DISCUSSION_NOTES.md#note-20260628-02-tide-overfit-free-run-bar-10)).

**Related:** [EXP-20260627-03](EXPERIMENT_LOG.md#exp-20260627-03-ar-tide-overfit-training-fixes-λ-ramp-ablation), [EXP-20260627-04](EXPERIMENT_LOG.md#exp-20260627-04-ar-gate-tide-overfit-pass-wsl-300ep), [NOTE-20260627-01](DISCUSSION_NOTES.md#note-20260627-01-gate-tide-overfit-plateau-and-open-hypotheses)

---

## Session 2026-06-27 — AR Phase 0+1 + `gate-tide-overfit` failure

### NOTE-20260627-01: `gate-tide-overfit` plateau and open hypotheses

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-27 18:46:29                                                   |
| **Topic**     | AR tide overfit — `val_token_accuracy` ~0.48, F1 ≪ 1.0              |

**Context:** Phase 0+1 landed in `onset_ar/` (EXP-20260627-01). First full WSL **`gate-tide-overfit`** run (300 ep, EXP-20260627-02) still fails pass criterion (teacher-fed event F1 ≈ 1.0). Dense tide overfit remains ~98% F1 (EXP-20260606-12) on the same song.

**Confirmed observations:**

- Tide decoder targets: **635** steps; token **83** appears **305** times → **305/635 ≈ 0.4803**, matching logged `val_token_accuracy` on most epochs.
- Token 83 encodes inter-onset delta **17 frames (170 ms)** — dominant spacing on tide (305/633 raw deltas).
- `val_token_loss` decreases while argmax accuracy stays at ~0.48 — consistent with a **fixed majority-class argmax**, not necessarily with learning per-step tokens.
- `val_pointer_loss` ~**6.46** vs uniform baseline log(1607) ≈ **7.38** — pointer head only weakly better than uniform.
- `val_event_onset_f1` peaked **~0.137** mid-training then fell to **0.0**; checkpoint metric is decoded event F1, not token accuracy.
- Epoch 1: train token acc **~0.02**, val **~0.48** — large train/eval gap; dropout 0.1 may contribute.
- **CPU 50-step probe (local, mask-fix code):** after **1** Adam step, train **and** eval accuracy both jump to **0.4803**; after 50 steps eval **0.52**, train **0.53**; argmax dominated by token **83** (544/635 steps). Collapse is not eval-only — happens immediately in both modes.
- Random-init argmax accuracy **~0.005** — plateau is post-training behavior, not a metric init bug.

**Fixes attempted (partial):**

- Trainer: `apply_training_seed`; F1 metric tensor reshape (`e1ad6b9`).
- Attention masks: Keras polarity (`True` = masked) + causal direction — **local uncommitted** change; 15-ep smoke showed moving F1 (~0.11) but full 300-ep run still failed gate.

**Open hypotheses (needs verification):**

- Token-head collapse to majority delta class; class-weighted / focal CE or phased training (`lambda_time=0` first).
- Loss balance: `lambda_time=1.0` with `time_loss` ~15–30 may dominate gradients vs token CE ~1.7.
- Metric/train mismatch: F1 from soft expected patch; pointer CE on hard patch index.
- Label or mask bugs beyond attention (EOS step, padding, teacher-forcing alignment) — not ruled out.
- Overfit hygiene: dropout off, LR, longer smoke before 300 ep.

**Implication:** Resolved in [NOTE-20260627-02](DISCUSSION_NOTES.md#note-20260627-02-gate-tide-overfit-resolution) / [EXP-20260627-04](EXPERIMENT_LOG.md#exp-20260627-04-ar-gate-tide-overfit-pass-wsl-300ep). Historical context only.

**Related:** [EXP-20260627-01](EXPERIMENT_LOG.md#exp-20260627-01-ar-phase-01-implementation--tide-verify), [EXP-20260627-02](EXPERIMENT_LOG.md#exp-20260627-02-ar-gate-tide-overfit-wsl-300ep-tide)

---

## Session 2026-06-14 — AR onset design locked

### NOTE-20260614-01: Autoregressive onset v1 stack and gates

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-14 18:00:00                                                   |
| **Topic**     | AR seq2seq onset formulation vs dense / K-query                       |

**Context:** Dense MERT val best **0.686** (EXP-20260610-03); K-query event plateau ~30% with oracle ~31% (EXP-20260606-11). Chart times are an ordered sparse list — AR avoids Hungarian assignment and may interface with future chart generation.

**Decisions locked (slug registry in [AR_ONSET_DESIGN.md §11](AR_ONSET_DESIGN.md#11-decision-registry)):**

- **Package:** `src/stepcovnet/onset_ar/` (not extending `onset_events/`)
- **Model:** patched frozen MERT (P=8) → encoder–decoder; **pointer+residual** alignment + **`delta_bucketed`** token LM
- **Eval:** primary event F1 **without** min-gap (`eval-min-gap`); checkpoint on **decoded** event F1 (`train-checkpoint`)
- **Training:** teacher forcing first; scheduled sampling ramp **after** `gate-tide-overfit` (`gate-ar-decode`)
- **Scoreboard:** keep dense baseline until `gate-val-vs-dense` (`dense-baseline`)

**Open:** `delta-buckets` vocab edges, `train-aux-time-loss` (λ_time), `ship-path` (F3).

**Implication:** Two parallel tracks — **Track A** full `final_data` dense baseline; **Track B** implement AR and run gates on tide → 10-song smoke → val vs dense.

**Related:** [DECISIONS_CHECKLIST.md § C](DECISIONS_CHECKLIST.md#c-core-model-middle) · [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) (AR track)

---

## Session 2026-06-24 — P8 complete + training manifest wiring

### NOTE-20260624-01: final_data ready for multi-song training

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-24 12:00:00                                                   |
| **Topic**     | P8 manifest + dense/event trainer hookup on `final_data`              |

**Context:** P8 (`422a985`) and manifest-as-pointer wiring (`95367e7`) landed; docs still routed agents to build index / blocked val on P8.

**Discovery:**

- `training_index.json`: **1010** / **110** songs, **1745** / **197** chart rows (`stratified_song_v1`, val_fraction 0.1)
- Dense + event trainers accept `--training_index_path`; 10-song CPU smoke **10/10** batches each track
- Event baseline caps raised to **2048** (`n_max_onsets`, `max_steps_per_chart`, `num_queries`) — required for Raputa (1164 steps)

**Implication:**

- **Current phase:** first full GPU dense train on `data/final_data/training_index.json`, then val eval + threshold sweep
- Legacy `data_dir=val_data_dir=data/final_data` remains but does not replace the P8 song split

**Related:** [EXP-20260623-02](EXPERIMENT_LOG.md#exp-20260623-02-p8-trainval-manifest-on-full-final_data), [EXP-20260624-01](EXPERIMENT_LOG.md#exp-20260624-01-10-song-dense-training-smoke-training_index_path), [DATASET_PREP_PIPELINE.md](DATASET_PREP_PIPELINE.md) §2

---

## Session 2026-06-22 — P9 smoke + doc sync

### NOTE-20260622-01: P9 loaders validated on full final_data

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-22 09:25:00                                                   |
| **Topic**     | P9 training loader smoke on local `data/final_data`                   |

**Context:** P9 committed (`5810ee8`); docs still described loaders as future work.

**Discovery:**

- `discover_training_rows` / `list_training_samples`: **1942** rows; 0 missing audio or `.chart.json`
- Bundle breakdown: ITL 246, Mizuki 1310, Vocaloid 386 chart rows; **822** with `chart_index > 0`
- Multi-chart indexing loads distinct step counts per index; TF onset dataset builds with GT when audio window covers chart offset (first ITL easy chart starts ~6.86s)

**Implication:**

- P9 **done**; ~~P8 is the remaining gate~~ **Update (2026-06-24):** P8 done — `training_index.json` + dense/event `--training_index_path` (EXP-20260623-02, EXP-20260624-01/02)
- Synced [DATASET_PREP_PIPELINE.md](DATASET_PREP_PIPELINE.md) §2/§10/§13/§16, [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [project-layout.md](../agents/project-layout.md), [AGENTS.md](../../AGENTS.md), [README.md](../../README.md)

**Related:** [EXP-20260622-01](EXPERIMENT_LOG.md#exp-20260622-01-p9-final_data-loader-smoke), DATASET_PREP_PIPELINE §10

---

## Session 2026-06-14 — design doc field rationale convention

### NOTE-20260614-01: schema_version and per-field rationale in plan docs

| Field         | Value                                                                 |
| ------------- | --------------------------------------------------------------------- |
| **Timestamp** | 2026-06-14 18:01:36                                                   |
| **Topic**     | Document field decisions in plan docs, not only chat                  |

**Context:** User asked what `schema_version` is; clarified that loaders need a version contract and that each field's *why* should live in documentation.

**Discovery:**

- `schema_version` is an integer **contract tag** on JSON artifacts (chart, manifest, report) so loaders can branch or fail on unknown layouts; v2 planned for deferred stats (`steps_per_second`, `mine_times_sec`, §15.2/15.4).
- Explanations like this belong in the **plan doc** as field rationale tables, not only in conversation.

**Implication:**

- Added [DATASET_PREP_PIPELINE.md §6.3](DATASET_PREP_PIPELINE.md#63-schema-versions-and-field-rationale) (version changelog + field tables).
- New always-on rule [`.cursor/rules/design-doc-field-decisions.mdc`](../../.cursor/rules/design-doc-field-decisions.mdc); `research-notebook.mdc` and [docs/agents/research.md](../agents/research.md) updated.

**Related:** DATASET_PREP_PIPELINE §6.3, §14, §15; AGENTS.md rules table

---

## Session 2026-06-10 — architecture smoke suite

### NOTE-20260610-02: BiLSTM wins arch smoke; transformer OOM

| Field         | Value                                                                              |
| ------------- | ---------------------------------------------------------------------------------- |
| **Timestamp** | 2026-06-10 05:59:51                                                                |
| **Topic**     | Onset backbone comparison @ 10-train / 50-ep smoke budget                          |

**Context:** [EXP-20260610-01](EXPERIMENT_LOG.md#exp-20260610-01-arch-smoke-suite-manifest) — 9 configs via `run_arch_research_suite.py`, Gaussian + MERT, `post_hoc_event_f1_export`.

**Discovery:**

- **BiLSTM** (128 units, depth 2) leads: micro **0.677** @ thr=0.15 — +4.4 pp vs EXP-03 U-Net baseline (0.633 @ 40 ep) at comparable train size.
- **TCN** (4 blocks, dilations 1–8) is #2: **0.664** @ 0.25; scaling to 20-train only reaches **0.657** (underfits vs 10-train TCN on this budget).
- **U-Net** ablations cluster **0.646–0.656** — depth-1 wide best among variants; σ=0.5 peaks at thr=0.10 (0.655).
- **Transformer** (2 layers, 4 heads, 64 filters) **OOM** on epoch 1: attention softmax `[1,4,12714,12714]` ≈ 2.4 GiB — full-sequence self-attention incompatible with ~12.7k-frame songs on current GPU without chunking/local attention.

**Implication:** For dense onset @ MERT, prefer **BiLSTM or TCN** over U-Net tweaks at 10-train smoke scale. Transformer deferred until chunked attention or shorter sequences. Round-2 probes BiLSTM 20-train, TCN depth, binary vs Gaussian on BiLSTM.

**Open (updated EXP-20260610-02):** BiLSTM 20-train reaches **0.680** @ 50 ep — still −0.3 pp vs EXP-12 (0.683 @ 50-train/200ep). Gap likely needs **200 ep + 50 train**, not arch swap alone.

**Related:** EXP-20260610-01/02, `configs/research/arch/manifest_round2.json`, `OnsetModelConfig.onset_architecture`

### NOTE-20260610-03: Round-2 BiLSTM width wins; TCN depth rejected

| Field         | Value                                                                              |
| ------------- | ---------------------------------------------------------------------------------- |
| **Timestamp** | 2026-06-10 11:19:14                                                                |
| **Topic**     | BiLSTM hyperparameter follow-ups @ smoke budget                                    |

**Discovery:** `recurrent_units=256` → **0.680** @ 0.25 (best overall). Binary BiLSTM **0.674** — Gaussian still +0.6 pp on same backbone. `tcn_blocks=6` → **0.655**, below blocks=4 **0.664**.

**Implication:** Scale **BiLSTM 256u** to 50-train / 200ep; do not invest in deeper TCN or full-seq transformer without chunked attention.

**Related:** EXP-20260610-02

---

## Session 2026-06-10 — auto post-hoc event-F1 export

### NOTE-20260610-01: Auto post-hoc event-F1 export implemented

| Field         | Value                                                                              |
| ------------- | ---------------------------------------------------------------------------------- |
| **Timestamp** | 2026-06-10 01:20:52                                                                |
| **Topic**     | Wire post-hoc checkpoint+threshold event-F1 selection into the dense onset trainer |

**Context:** [NOTE-20260609-09](#note-20260609-09-frame-f1-checkpoint-mis-ranks-event-f1) and [NOTE-20260609-11](#note-20260609-11-50train-200ep-frame-export-mis-rank) showed the frame-F1 `ModelCheckpoint` monitor mis-ranks event F1: the exported peak-frame ckpt loses ~1–2 pp to a mid-train ckpt at the tuned POST threshold. Previously this required two manual scripts after every run (`sweep_val_onset_ckpts.py` + `sweep_val_threshold.py`).

**Discovery / change:** Implemented the sweep inside `run_train_from_config`, opt-in via `RunConfig.post_hoc_event_f1_export` (+ `post_hoc_event_f1_thresholds`, default 0.05–0.50 grid).

- `dense_overfit_eval.sweep_thresholds_dense_val_event_f1` predicts **once per val pair** and re-scores cached traces across all thresholds (cost ≈ one inference pass over val, not n_thresholds passes).
- `trainers._select_best_event_f1_checkpoint` loads every `VAL_ONSET_F1_SCORE-*.keras` callback, sweeps thresholds, and picks the (checkpoint, threshold) with max micro event F1.
- `trainers._export_best_event_f1_checkpoint` overwrites the exported `.keras` with that checkpoint and writes `event_f1_sweep.json` (best ckpt/threshold + per-checkpoint table) under `model_output_dir`.

**Implication:** When enabled, the exported model is the event-F1-optimal checkpoint rather than the frame-F1 peak, and the winning POST threshold is recorded for eval. Default is **off** (frame-F1 export unchanged), so fast iteration runs are unaffected. **No training run yet** — feature + tests only; needs a real run to confirm it reproduces the EXP-12 0.683 selection automatically.

**Related:** NOTE-20260609-09/11, EXP-20260609-11/12, `src/stepcovnet/trainers.py`, `src/stepcovnet/dense_overfit_eval.py`, `src/stepcovnet/config.py`; DECISIONS_CHECKLIST A1.

---

## Session 2026-06-09 — Gaussian 50-train 200ep eval

### NOTE-20260609-11: 50-train 200ep frame export mis-ranks event F1

| Field         | Value                                                             |
| ------------- | ----------------------------------------------------------------- |
| **Timestamp** | 2026-06-09 12:33:15                                               |
| **Topic**     | Frame-F1 checkpoint export vs post-hoc event-F1 on 50-train 200ep |

**Context:** EXP-20260609-12 completed 200/200 ep; canonical eval + threshold sweep + callback sweep.

**Discovery:** Exported frame-F1 peak ckpt (0.79480) reaches micro 0.674 @ POST thr=0.35 (ties EXP-07). Mid-train ckpt 0.77102 reaches 0.683 @ same threshold (+0.9 pp). Frame-F1 monitor again mis-ranks event-F1 weights.

**Implication:** Run post-hoc ckpt sweep at tuned POST thr before trusting exported weights.

**Related:** EXP-20260609-12, EXP-20260609-07, NOTE-20260609-09

---

## Index

| ID                                                                                              | Timestamp           | Topic                                                                   |
| ----------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------- |
| [NOTE-20260610-01](#note-20260610-01-auto-post-hoc-event-f1-export-implemented)                 | 2026-06-10 01:20:52 | Auto post-hoc event-F1 checkpoint+threshold export wired into trainer   |
| [NOTE-20260609-11](#note-20260609-11-50train-200ep-frame-export-mis-rank)                       | 2026-06-09 12:33:15 | 50-train 200ep: frame peak export 0.674 vs ckpt 0.683 @ 0.35            |
| [NOTE-20260609-10](#note-20260609-10-final-weight-export-protocol)                              | 2026-06-09 09:46:42 | Final-epoch export via `callback_root_dir=""` — 100-train protocol      |
| [NOTE-20260609-09](#note-20260609-09-frame-f1-checkpoint-mis-ranks-event-f1)                    | 2026-06-09 09:45:12 | Mid-train frame ckpt **0.671** event F1 beats frame-F1 peak **0.654**   |
| [NOTE-20260609-08](#note-20260609-08-callback-export-not-final-weights)                         | 2026-06-09 09:09:04 | no-ES 200ep still **0.654** — `_write_model` exports best frame ckpt    |
| [NOTE-20260609-07](#note-20260609-07-200ep-does-not-fix-ep11-checkpoint)                        | 2026-06-09 05:01:34 | 200ep + patience 25 → identical **0.654**; frame-F1 ES blocks scale     |
| [NOTE-20260609-06](#note-20260609-06-gaussian-100train-underperforms-smaller-runs)              | 2026-06-09 04:04:23 | 100-train Gaussian 40ep **0.654** — below 50-train; need longer train   |
| [NOTE-20260609-05](#note-20260609-05-gaussian-50train-and-100train-rerun)                       | 2026-06-09 03:19:17 | 50-train **0.674** @ thr=0.35 — recommend 100-train Gaussian re-run     |
| [NOTE-20260609-04](#note-20260609-04-gaussian-scaling-breakthrough)                             | 2026-06-09 02:49:21 | Gaussian 20-train **0.667** @ thr=0.25 — beats 100-train 0.635          |
| [NOTE-20260609-03](#note-20260609-03-arch-large-small-data-collapse)                            | 2026-06-09 02:09:14 | 3.5M-param U-Net collapses on 10 songs — epoch-1 restore, F1≈flux       |
| [NOTE-20260609-02](#note-20260609-02-gaussian-vs-binary-10train-comparison)                     | 2026-06-09 01:48:26 | Gaussian 10-train @ tuned POST beats binary (+2.4 pp); ≈100-train scale |
| [NOTE-20260609-01](#note-20260609-01-per-checkpoint-threshold-sweep-protocol)                   | 2026-06-09 01:20:00 | Always sweep POST thr per checkpoint; 0.20 median for dense MERT        |
| [NOTE-20260608-03](#note-20260608-03-spectral-flux-baseline-ceiling)                            | 2026-06-09 00:56:51 | Librosa flux ~32% micro F1 — far below MERT dense @ tuned POST          |
| [NOTE-20260608-02](#note-20260608-02-100train-threshold-recalibration-breakthrough)             | 2026-06-08 00:44:14 | 100-train needs thr≈0.20 not 0.05 — +6.3 pp micro F1 (POST only)        |
| [NOTE-20260608-01](#note-20260608-01-disable-dense-val-event-f1-callback)                       | 2026-06-08 02:00:00 | Disable in-train event F1 callback for faster dense scaling             |
| [NOTE-20260607-06](#note-20260607-06-phase-2-100-train-scaling)                                 | 2026-06-07 20:01:00 | Phase 2: 100-train config + smoke + full run started                    |
| [NOTE-20260607-05](#note-20260607-05-dense-train-eval-metric-alignment)                         | 2026-06-07 12:25:00 | Dense train checkpoint on peak-pick event F1 @ config threshold         |
| [NOTE-20260607-04](#note-20260607-04-phase-1-canonical-dense-eval)                              | 2026-06-07 11:17:32 | Phase 1: uncapped GT + `eval_dense_onset.py`; dual-threshold reporting  |
| [NOTE-20260607-03](#note-20260607-03-val-wide-threshold-sweep-test-time-threshold-ood-bottom-3) | 2026-06-07 09:57:48 | Full val threshold sweep, test-time threshold options, OOD bottom 3     |
| [NOTE-20260607-02](#note-20260607-02-bad-val-threshold-sweep-and-feature-profiles)              | 2026-06-07 09:46:11 | EXP-21 worst val songs: threshold sweep + MERT/pred profile comparison  |
| [NOTE-20260607-01](#note-20260607-01-onset-detection-literature-and-pipeline-options)           | 2026-06-07 05:15:00 | MIR onset literature vs StepCOVNet; viable pipeline changes             |
| [NOTE-20260606-17](#note-20260606-17-multi-song-eval-missing-normalization)                     | 2026-06-06 17:17:55 | Per-pair dense eval skipped feature normalization                       |
| [NOTE-20260606-16](#note-20260606-16-dense-seed-before-model-init)                              | 2026-06-06 16:54:25 | Root cause of dense tide train instability                              |
| [NOTE-20260606-15](#note-20260606-15-timestamp-format-system-clock-at-write)                    | 2026-06-06 16:37:35 | Full timestamps from system clock; no approximate suffixes              |
| [NOTE-20260606-14](#note-20260606-14-uniform-grid-oracle-matches-f1-plateau)                    | 2026-06-06 17:20:00 | Grid oracle ~31%; bisection closes bug hunt                             |
| [NOTE-20260606-13](#note-20260606-13-conv1d-zero-f1--confidence-collapse-from-ordered-training) | 2026-06-06 14:12:41 | Ordered train collapsed all confidences → 0 F1 on conv1d                |
| [NOTE-20260606-12](#note-20260606-12-hungarian-l1-training-loss-implemented)                    | 2026-06-06 14:10:00 | Training loss uses Hungarian L1 assignment                              |
| [NOTE-20260606-11](#note-20260606-11-hungarian-matching-and-clustered-predictions)              | 2026-06-06 11:12:00 | Clustering many preds on one onset vs spread GT                         |
| [NOTE-20260606-10](#note-20260606-10-pipeline-pre-model-post-and-metrics)                       | 2026-06-06 11:00:00 | Pipeline framing → [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) |
| [NOTE-20260606-09](#note-20260606-09-research-notebook-experiments-plus-discussion)             | 2026-06-06 10:30:00 | Notebook covers experiments _and_ discussion                            |
| [NOTE-20260606-08](#note-20260606-08-raw-audio-vs-mel-two-separate-difficulties)                | 2026-06-06 09:21:00 | Input representation vs output/loss are separate problems               |
| [NOTE-20260606-07](#note-20260606-07-num_queries-vs-n_max_onsets)                               | 2026-06-06 09:18:00 | `num_queries` vs chart length; K=634 not magic                          |
| [NOTE-20260606-06](#note-20260606-06-overfit-shortcuts-gt-refs-and-frozen-deltas)               | 2026-06-06 09:15:00 | GT query refs + frozen deltas are pipeline checks, not production       |
| [NOTE-20260606-05](#note-20260606-05-count-from-times-not-a-separate-head)                      | 2026-06-06 09:12:00 | Count should come from times + confidence, not its own head             |
| [NOTE-20260606-04](#note-20260606-04-gap-and-cluster-charts-break-index-alignment)              | 2026-06-06 09:09:00 | Strict index alignment fails on gap/cluster charts                      |
| [NOTE-20260606-03](#note-20260606-03-seq2seq-goal-vs-detr-style-k-slots)                        | 2026-06-06 09:06:00 | Product goal is seq2seq; implementation uses K query slots              |
| [NOTE-20260606-02](#note-20260606-02-train-ordered-vs-eval-hungarian)                           | 2026-06-06 09:03:00 | Training uses ordered assignment; eval uses Hungarian                   |
| [NOTE-20260606-01](#note-20260606-01-how-event-eval-works)                                      | 2026-06-06 09:00:00 | Event eval: padding, K slots, Hungarian, TP/FP/FN                       |

---

## Session 2026-06-09 — autonomous onset research (continued)

### NOTE-20260609-10: Final-weight export protocol

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 09:46:42 |

**Topic:** How to eval epoch-200 weights for 100-train Gaussian

**Context:** `_write_model` copies best `VAL_ONSET_F1_SCORE-*.keras` when `callback_root_dir` is set ([NOTE-20260609-08](DISCUSSION_NOTES.md#note-20260609-08-callback-export-not-final-weights)). Post-hoc sweep already recovered **0.671** (EXP-11); late-epoch weights untested.

**Discovery:** Set `"callback_root_dir": ""` in run config — `_build_callbacks` returns empty list; `_write_model` saves in-memory final-epoch weights to `model_output_dir`. Smoke config: `configs/research/gaussian_10train_3ep_final_weights.json` (10 songs, 3 ep, no callbacks). Compare against sibling run with callbacks on same seed.

**Implication:** Run `gaussian_100train_3ep_final_weights` (or clone 100-train with `callback_root_dir=""` + low epoch) when GPU free; eval with `eval_dense_onset.py` + threshold sweep. Alternative: keep callbacks and rank via `scripts/sweep_val_onset_ckpts.py`.

**Related:** `src/stepcovnet/trainers.py` `_write_model`, EXP-20260609-11, EXP-20260609-12

---

### NOTE-20260609-09: Frame-F1 checkpoint mis-ranks event F1

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 09:45:12 |

**Topic:** Post-hoc sweep recovers +1.7 pp on 100-train Gaussian

**Context:** [EXP-20260609-11](EXPERIMENT_LOG.md#exp-20260609-11-gaussian-100train-callback-sweep) — nine `VAL_ONSET_F1_SCORE` checkpoints from 200ep no-ES run.

**Discovery:** Exported peak-frame ckpt (`0.78830`) → **0.654** @ 0.25. Mid-training ckpt (`0.75754`) → **0.671** @ thr=0.30. Higher frame F1 does not imply higher event F1 after POST threshold tuning.

**Implication:** Always sweep POST threshold per checkpoint; consider event-F1-based checkpoint selection or post-hoc callback ranking. 100-train Gaussian ceiling ~**0.671** — still below 50-train **0.674**.

**Related:** EXP-20260609-10/11, NOTE-20260609-08, NOTE-20260609-01

---

### NOTE-20260609-08: Callback export, not final weights

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 09:09:04 |

**Topic:** `early_stopping_patience=0` still yields ep11-equivalent eval

**Context:** [EXP-20260609-10](EXPERIMENT_LOG.md#exp-20260609-10-gaussian-100train-200ep-no-es) — full 200-epoch train, no early stopping.

**Discovery:** Eval micro F1 **0.654** @ thr=0.25 bit-identical to EXP-08/09. `trainers._write_model` loads best `VAL_ONSET_F1_SCORE-*.keras` from `callback_root_dir` (peak **0.788** @ ~ep11), ignoring final-epoch weights.

**Implication:** To test late-epoch 100-train Gaussian: set `callback_root_dir=""` for final-weight export, or sweep all saved callback checkpoints with `eval_dense_onset.py`. Session best remains 50-train **0.674**.

**Related:** EXP-20260609-08/09/10, `src/stepcovnet/trainers.py` `_write_model`

---

### NOTE-20260609-07: 200ep does not fix ep11 checkpoint

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 05:01:34 |

**Topic:** Longer training cap blocked by frame-F1 early stopping

**Context:** [EXP-20260609-09](EXPERIMENT_LOG.md#exp-20260609-09-gaussian-100train-200ep-early-stop-blocked) — 200 ep, patience 25 vs 40ep/patience 15 in EXP-08.

**Discovery:** Both runs early-stop and restore **epoch 11**; eval micro F1 **0.654** @ thr=0.25 is bit-identical. Session best remains 50-train **0.674**.

**Implication:** Epoch cap is not the bottleneck — **checkpoint metric** is. Next experiments: disable early stop for 100-train Gaussian, or checkpoint on swept event F1, or save top-N epoch weights for POST sweep.

**Related:** EXP-20260609-08/09, EXP-20260609-07

---

### NOTE-20260609-06: Gaussian 100-train underperforms smaller runs

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 04:04:23 |

**Topic:** 100-train Gaussian @ 40ep does not beat 50-train at POST-tuned eval

**Context:** [EXP-20260609-08](EXPERIMENT_LOG.md#exp-20260609-08-gaussian-100train-40ep-undertrained) — early stop ep26, best **ep11**.

**Discovery:** Gaussian @ optimal POST does not monotonically scale with train count under fixed 40ep / frame-F1 checkpoint:

| train | best thr | micro F1  |
| ----- | -------- | --------- |
| 20    | 0.25     | 0.667     |
| 50    | 0.35     | **0.674** |
| 100   | 0.25     | 0.654     |

100-train still beats binary 100-train **0.635** (+1.9 pp) but loses to mid-scale Gaussian runs.

**Implication:** Frame-F1 early stopping underfits at 100 songs; rerun with **200 ep** / higher patience before concluding Gaussian scaling fails. **50-train checkpoint remains session best** for deployment experiments.

**Related:** EXP-20260609-06/07/08, NOTE-20260609-05

---

### NOTE-20260609-05: Gaussian 50-train and 100-train re-run

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 03:19:17 |

**Topic:** 50-train smoke clears 0.67 bar; queue 100-train Gaussian

**Context:** [EXP-20260609-07](EXPERIMENT_LOG.md#exp-20260609-07-gaussian-50train-40ep-threshold-sweep) — early stop @ ep21 restored **epoch 6** weights.

**Discovery:** Gaussian scaling @ swept optimal POST:

| train      | best thr | micro F1 @ optimal | micro @ 0.05 |
| ---------- | -------- | ------------------ | ------------ |
| 20         | 0.25     | 0.667              | 0.642        |
| 50         | 0.35     | **0.674**          | 0.566        |
| 100 binary | 0.20     | 0.635              | 0.572        |

50-train edges 20-train slightly (+0.7 pp) but needs higher POST thr (0.35). Raw @ 0.05 regresses — checkpoint selection (ep6) may be suboptimal for event eval.

**Implication:** **Recommend full 100-train Gaussian** re-run (≥50 ep, sweep POST). Deprioritize binary 100-train as production baseline. Consider longer patience or event-F1 checkpoint for 50+ song runs.

**Related:** EXP-20260609-06/07, EXP-20260608-01

---

### NOTE-20260609-04: Gaussian scaling breakthrough

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 02:49:21 |

**Topic:** Gaussian targets scale better than binary with train size

**Context:** [EXP-20260609-06](EXPERIMENT_LOG.md#exp-20260609-06-gaussian-20train-40ep-threshold-sweep) vs 10-train and 100-train baselines.

**Discovery:** Gaussian train-size curve @ swept optimal POST:

| train songs  | best thr | micro F1  |
| ------------ | -------- | --------- |
| 10           | 0.25     | 0.633     |
| **20**       | 0.25     | **0.667** |
| 100 (binary) | 0.20     | 0.635     |

20-train Gaussian already beats 100-train binary-scale with 5× less data. Even @ fixed thr=0.05, 20-train hits **0.642**.

**Implication:** Promote Gaussian as default TRAIN target; run 50-train smoke next. Revisit whether 100-train binary checkpoint is obsolete for dense onset.

**Related:** EXP-20260609-03/06, EXP-20260608-01

---

### NOTE-20260609-03: arch_large small-data collapse

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 02:09:14 |

**Topic:** Large U-Net fails on 10-song dense onset

**Context:** [EXP-20260609-04](EXPERIMENT_LOG.md#exp-20260609-04-arch-large-10train-40ep-collapse) — 32 filters / depth 3 / dropout 0.1 vs baseline 16f/depth2 (229K params).

**Discovery:** After epoch 1, train frame `onset_f1_score` stayed at 0; early stopping @ ep16 restored **epoch 1** checkpoint. Event eval micro F1 **0.326** @ thr=0.10 — same ballpark as librosa flux (~0.32), −28 pp vs Gaussian.

**10-train comparison @ optimal POST:**

| Variant    | params | best thr | micro F1  |
| ---------- | ------ | -------- | --------- |
| Gaussian   | 230K   | 0.25     | **0.633** |
| Binary     | 230K   | 0.20     | 0.609     |
| arch_large | 3.5M   | 0.10     | 0.326     |

**Implication:** Capacity scaling without more data hurts; prefer Gaussian targets + modest U-Net. Do not scale arch before train-set size.

**Related:** EXP-20260609-01/03/04, EXP-20260608-03

---

### NOTE-20260609-02: Gaussian vs binary 10-train comparison

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 01:48:26 |

**Topic:** Target shape ablation at 10 songs with per-run optimal POST

**Context:** [EXP-20260609-01](EXPERIMENT_LOG.md#exp-20260609-01-binary-10train-40ep-threshold-sweep) vs [EXP-20260609-03](EXPERIMENT_LOG.md#exp-20260609-03-gaussian-10train-40ep-threshold-sweep).

**Discovery:** At each run's swept optimal threshold:

| Target   | best thr | micro F1 @ 0.05 | micro F1 @ optimal |
| -------- | -------- | --------------- | ------------------ |
| Binary   | 0.20     | 0.505           | **0.609**          |
| Gaussian | 0.25     | 0.586           | **0.633**          |

Gaussian wins +2.4 pp at matched protocol; approaches 100-train ceiling (0.635) with 10× less data.

**Implication:** Soft Gaussian targets may be a better TRAIN default for dense onset; arch_large run tests capacity separately.

**Related:** EXP-20260609-01, EXP-20260609-03, EXP-20260608-01

---

### NOTE-20260609-01: Per-checkpoint threshold sweep protocol

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 01:20:00 |

**Topic:** Always sweep POST confidence threshold per checkpoint

**Context:** Binary 10-train eval @ fixed 0.05 gave micro **0.505**; sweep → **0.609** @ 0.20 (same median as 100-train). 75-train @ 0.20 only **0.550** — confirms 100-train **0.635** is real scaling, not universal higher-threshold trick.

**Discovery:** Dense MERT checkpoints consistently peak near thr≈**0.20** on val (median per-song optimal). Fixed 0.05 systematically under-reports event F1 for larger train sets.

**Implication:** Report micro F1 at swept optimal thr alongside default 0.05. Gaussian vs binary comparison must use per-run optimal thr, not shared 0.05.

**Related:** [EXP-20260609-01](EXPERIMENT_LOG.md#exp-20260609-01-binary-10train-40ep-threshold-sweep), [EXP-20260609-02](EXPERIMENT_LOG.md#exp-20260609-02-75train-thr020-apples-to-apples), [EXP-20260608-01](EXPERIMENT_LOG.md#exp-20260608-01-100train-threshold-sweep-post-breakthrough)

---

### NOTE-20260608-03: Spectral-flux baseline ceiling

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-09 00:56:51 |

**Topic:** Librosa onset-strength baseline vs dense MERT

**Context:** Research direction #5 — classical spectral-flux upper bound on val subset.

**Discovery:** New `scripts/eval_spectral_flux_onset.py`; 5-song val pilot with threshold sweep peaks at micro **0.317** @ thr=0.05. Dense 100-train @ thr=0.20 is **0.635** on full val — ~2× absolute F1 gap. Flux is high-recall / low-precision (many spurious peaks on game audio).

**Implication:** MERT+U-Net value is not replaceable by off-the-shelf flux; tune POST on neural checkpoints instead. Full 40-song flux sweep optional for paper table.

**Related:** [EXP-20260608-03](EXPERIMENT_LOG.md#exp-20260608-03-spectral-flux-baseline-5-val-songs), NOTE-20260607-01

---

## Session 2026-06-08 — autonomous onset research

### NOTE-20260608-02: 100-train threshold recalibration breakthrough

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-08 00:44:14 |
| **Tags**      | post, metric, train |

**Context:** EXP-20260607-02 looked like a regression (micro **0.572** @ thr=0.05 vs 75-train **0.577**). User requested autonomous research on better onset approaches.

**Discovery:** Val-wide threshold sweep on the **same** 100-train checkpoint (`val_threshold_sweep.json`) shows optimal global thr=**0.20** → micro event F1 **0.6349** (+6.3 pp vs 0.05). Confirmed via `eval_dense_onset.py @ 0.20`. Oracle per-song thr ceiling micro **0.6457** (median thr 0.20). The 75-train-tuned thr=0.05 does **not** transfer — 100-train logits are better calibrated at higher confidence cutoffs.

**Implication:** Phase 2 scaling **did help**; we mis-reported due to fixed threshold from a different checkpoint. **Always re-sweep val threshold** after retrain or scale-up. Cheap POST win — no architecture change.

**Open:** Does thr=0.20 hold on held-out test? Score-percentile calibration vs fixed thr?

**Related:** [EXP-20260608-01](EXPERIMENT_LOG.md#exp-20260608-01-100train-threshold-sweep-post-breakthrough), NOTE-20260607-03, EXP-20260607-02

---

## Session 2026-06-08 — Disable dense val event F1 callback

### NOTE-20260608-01: Disable dense val event F1 callback

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-08 02:00:00 |

**Topic:** Comment out `DenseValEventF1Callback`; checkpoint on frame F1 again

**Context:** Phase 2 100-train with per-epoch peak-pick event F1 added ~30–45 s/epoch (~1.6× wall time) with no evidence of training failures; user wants faster scaling runs.

**Discovery:** `run_train_from_config` no longer inserts `DenseValEventF1Callback`; `ONSET_CHECKPOINT_MONITOR` reverted to **`val_onset_f1_score`**. Helper `_build_dense_val_event_f1_callback` and callback class remain for later re-enable. Report event F1 via `scripts/eval_dense_onset.py` post-hoc (EXP-21 pattern).

**Smoke A/B (100-train, 3 ep, same config):** With metric **970.9 s** total (ep3 **103 s**); without metric **827.3 s** (**−144 s**, **15%**); steady ep3 **69 s** vs **103 s** (**−34 s**, **33%**). No `val_dense_event_onset_f1` in no-metric log. Artifacts: `callbacks/dense_mert_v2_100train_smoke_nometric/`.

**Implication:** Scaling runs optimize frame F1 @ 0.5 during train; canonical micro event F1 @ ~0.05 is eval-only until callback cost is acceptable or optimized (e.g. every-N epochs). At ~69 s/ep steady, 200 epochs ≈ **3.8 h** vs ~**5.7 h** with callback.

**Related:** `trainers.py`, [NOTE-20260607-05](#note-20260607-05-dense-train-eval-metric-alignment), `scripts/eval_dense_onset.py`

---

## Session 2026-06-07 — Phase 2 100-train scaling

### NOTE-20260607-06: Phase 2 100-train scaling

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-07 20:01:00 |

**Topic:** Phase 2 val-scale run with aligned checkpoint metric

**Context:** Phase 1 established canonical eval ([EXP-20260607-01](EXPERIMENT_LOG.md#exp-20260607-01-phase-1-canonical-dense-val-eval)) and train/eval alignment ([NOTE-20260607-05](#note-20260607-05-dense-train-eval-metric-alignment)). Next scaling step before 144-train.

**Discovery:** Created `configs/dense_mert_v2_100train.json` (100 songs, same arch as 75-train). 3-epoch WSL smoke confirms `val_dense_event_onset_f1` in logs and checkpoint path; ep3 smoke val **0.506** (early-training; not comparable to final eval). Full 200-ep train launched with early stopping (patience 25).

**Implication:** Compare post-train `eval_dense_onset.py` micro F1 @ 0.05 against EXP-21 **0.577** bar. Success → 144-train with same config contract.

**Related:** [EXP-20260607-02](EXPERIMENT_LOG.md#exp-20260607-02-dense-mert-100-train-event-f1-checkpoint), `configs/dense_mert_v2_100train.json`

---

## Session 2026-06-07 — Dense train/eval metric alignment

### NOTE-20260607-05: Dense train/eval metric alignment

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-07 12:25:00 |

**Topic:** Checkpoint and early-stop on the same peak-pick event F1 as eval

**Context:** Phase 1 reported event F1 @ 0.05 but training still checkpointed on frame F1 @ 0.5.

**Discovery:** Added `RunConfig.confidence_threshold` (default **0.05**), `tolerance_sec`, `min_onset_distance_ms`, `early_stopping_patience` (default **25**). Training runs `DenseValEventF1Callback` each epoch (same peak-pick path as `eval_dense_onset.py`) and checkpoints/early-stops on **`val_dense_event_onset_f1`**. Frame `onset_f1_score` remains logged as a diagnostic only.

**Implication:** Phase 2 scaling runs optimize the metric we report. Set `early_stopping_patience=0` to disable early stop.

**Related:** `dense_overfit_eval.py`, `trainers.py`, [DECISIONS_CHECKLIST](DECISIONS_CHECKLIST.md) A1/A3/A6

---

## Session 2026-06-07 — Phase 1 canonical dense eval

### NOTE-20260607-04: Phase 1 canonical dense eval

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-07 11:17:32 |

**Topic:** Phase 1 eval plumbing — uncapped GT and canonical script

**Context:** Prior EXP-21 event eval capped charts at 1024 steps and used a manual `uncapped_gt_songs` workaround for five long val charts.

**Discovery:** `build_gt_batch` now loads full chart length; `scripts/eval_dense_onset.py` wraps `eval_dense_val_event_f1` for reproducible val reports. Re-run on EXP-21 checkpoint matches old @ thr=0.5 (micro **0.450**); @ thr=0.05 micro **0.577** aligns with global sweep.

**Implication:** Checkpoint on frame F1 (`val_onset_f1_score`); **report** dense track numbers at val-tuned threshold (~0.05). Safe to proceed to Phase 2 (100-train) without re-auditing GT caps.

**Related:** [EXP-20260607-01](EXPERIMENT_LOG.md#exp-20260607-01-phase-1-canonical-dense-val-eval), `scripts/eval_dense_onset.py`, [DECISIONS_CHECKLIST](DECISIONS_CHECKLIST.md) A3/A6

---

## Session 2026-06-07 — val-wide threshold sweep and OOD (EXP-21)

### NOTE-20260607-03: Val-wide threshold sweep, test-time threshold, OOD bottom 3

| Field         | Value                    |
| ------------- | ------------------------ |
| **Timestamp** | 2026-06-07 09:57:48      |
| **Tags**      | post, metric, model, val |

**Context:** User requested full 40-song val threshold sweep (global + per-song oracle), test-time threshold guidance, and OOD analysis for bottom 3 val songs (`1_2_fanclub`, `dna`, `hakurei_shrine…`). Script: `scripts/sweep_val_threshold.py` → `models_wsl/dense_mert_v2_75train_200ep/val_threshold_sweep.json`.

**Discovery:**

1. **Global sweep (same grid 0.05–0.6, peak-pick POST):** Optimum at **thr=0.05** (grid floor — F1 still rising at lowest point). Mean event F1 **0.565** vs **0.437 @ 0.5** (+12.8 pp); micro F1 **0.577** vs **0.450** (+12.7 pp). Thr 0.10 is nearly tied (micro 0.575). Default 0.5 leaves substantial recall on the table.

2. **Per-song oracle (upper bound):** Mean F1 **0.580**, micro F1 **0.591** — only **+1.4 pp** above global best @ 0.05. Most songs share one low global threshold; per-song tuning is a small incremental gain, not a silver bullet. Threshold distribution: median **0.05**, **21/40** songs best at 0.05, p75=0.15, max=0.45; none prefer ≥0.5.

3. **Test-time threshold without GT:** Per-song oracle is **not deployable** on test (requires GT F1). Practical options ranked:
   - **Global val-tuned threshold** — sweep on held-out val, apply fixed thr to all test songs. Cheapest; here thr≈0.05–0.10. Risk: val/test calibration drift.
   - **Validation-set grid search on proxy metric** — if a small labeled dev set exists, same sweep; otherwise use val micro-F1 curve.
   - **Score-distribution calibration** — set thr from predicted prob percentiles (e.g. peaks above song-level p95 of frame scores) so dense vs sparse charts self-scale. No GT per song; needs validation that percentile rule generalizes.
   - **Metadata-conditioned threshold** — regress optimal thr from BPM, duration, step density, or MERT stats learned on val. Requires labeled val pairs per song; generalizes only if metadata→calibration relationship holds OOD.
   - **Avoid:** tuning thr on test predictions matched to leaked GT; using train-set oracle thresholds per song ID (test IDs unseen).

4. **OOD verdict for bottom 3:** All three are **not in the 75-song train subset** (seed 42) — held-out val, so zero train exposure. **MERT features are not outliers:** |z| < 1.5 for raw mean/std and per-dim variance vs 75-train distribution. Metadata mildly atypical: fanclub/hakurei shorter duration (−0.84σ), lower step density (−0.75 to −1.03σ), fewer steps (−0.92 to −1.14σ); dna BPM 200 (+1.24σ) but otherwise normal. **Pred calibration is the failure mode:** `pred_prob_at_gt_frames_mean` ≈ 0.004–0.005 vs non-GT ≈ 0.024–0.052 — model fires on wrong frames (GT inversion from NOTE-02). Threshold sweep barely helps (oracle F1 0.058 / 0.113 / 0.115). Verdict: **coverage gap + model miscalibration**, not MERT/audio OOD.

**Implication:** Adopt **global thr≈0.05–0.10** as default POST for EXP-21 checkpoint (+13 pp micro F1, free). Per-song oracle ceiling ~59% micro — remaining gap needs MODEL/TRAIN (include similar val styles in train, Gaussian targets, or calibration-aware loss). Bottom 3 need train coverage or style-specific heads, not threshold or feature fixes.

**Open:** Does thr=0.05 over-predict on truly unseen test? Finer grid below 0.05? Would metadata-conditioned thr beat global 0.05 on a second val fold?

**Related:** [NOTE-20260607-02](#note-20260607-02-bad-val-threshold-sweep-and-feature-profiles), EXP-21, `val_threshold_sweep.json`, `investigate_bad_val.json`.

---

## Session 2026-06-07 — bad val case investigation (EXP-21)

### NOTE-20260607-02: Bad val threshold sweep and feature profiles

| Field         | Value                    |
| ------------- | ------------------------ |
| **Timestamp** | 2026-06-07 09:46:11      |
| **Tags**      | post, metric, model, pre |

**Context:** After EXP-21 eval (mean event F1 0.437 @ thr=0.5), user asked to investigate the 7 worst val songs via threshold sweep and MERT/audio feature comparison vs 5 best songs. Script: `scripts/investigate_bad_val_cases.py` → `models_wsl/dense_mert_v2_75train_200ep/investigate_bad_val.json`.

**Discovery:**

1. **Threshold 0.5 is too high globally.** Best songs peak at thr 0.10–0.20 (+8–16 pp F1). Worst songs also prefer lower thr, but gains are uneven:
   - _Recall-fixable:_ `intersect_thunderbolt` 0.146→0.447 @ 0.05; `strobo_nights_ddrkirbys_summer_night_mix` 0.274→0.527; `the_purpose_song` 0.153→0.349; `bridge_no_one_passes` 0.188→0.297.
   - _Not fixable by POST alone:_ `1_2_fanclub` 0.037→0.058; `dna` 0.044→0.113; `hakurei_shrine…` 0.097→0.115.

2. **Three failure archetypes** (feature profiles @ thr=0.5):
   - **GT inversion** (fanclub, dna, hakurei): `pred_prob_at_gt_frames_mean` ≈ 0.005 vs best mean 0.32; `pred_prob_at_non_gt_mean` same as best (~0.034). Model fires on wrong frames; peak count often near GT but `peak_matched_frac` 4–10%. Frame recall @ 0.5 ≈ 0.
   - **Under-firing / recall** (intersect, purpose, bridge): MERT stats normal; model under-detects dense charts (8+ Hz). `bridge_no_one_passes` has _higher_ prob at GT (0.071) than non-GT (0.038) — signal exists, peaks too sparse.
   - **Calibration mismatch** (strobo): strong GT alignment (`pred_prob_at_gt` 0.158 vs non-GT 0.018, frame recall 12%) but very few peaks @ 0.5; lower thr recovers to ~0.53 F1.

3. **MERT input is not the differentiator.** Raw/normalized feature mean/std, step density, and chart length are similar between worst and best groups. Failure is **model calibration / generalization**, not missing or misaligned MERT cache.

**Implication:** Per-song or global threshold tuning (POST) is low-cost and helps ~4/7 worst cases materially; bottom 3 need **MODEL/TRAIN** fixes (more train coverage of those styles, Gaussian targets, or fine-tuned MERT). Global micro recall 0.34 is partly POST (threshold) and partly MODEL (GT frames get low logits on hard songs). Next: val-wide threshold sweep for aggregate F1; listen to fanclub/dna/hakurei for chart/audio mismatch.

**Open:** Are fanclub/dna/hakurei out-of-distribution vs 75-song train? Does per-song threshold on full val beat a single global optimum?

**Related:** EXP-21 eval (`eval_val_event_f1.json`), [NOTE-20260607-01](#note-20260607-01-onset-detection-literature-and-pipeline-options), `scripts/investigate_bad_val_cases.py`.

---

## Session 2026-06-07 — onset detection literature

### NOTE-20260607-01: Onset detection literature and pipeline options

| Field         | Value                                       |
| ------------- | ------------------------------------------- |
| **Timestamp** | 2026-06-07 05:15:00                         |
| **Tags**      | pre, model, post, metric, train, literature |

**Context:** User asked how others solve long-audio onset detection and which pipeline stages could improve StepCOVNet beyond current dense MERT U-Net (~48% val F1 @ 50 train, EXP-20).

**Discovery:** Standard MIR pipelines are **frame activation → peak picking → event list**, not threshold-every-frame. SOTA on benchmarks (OnsetDB, MAESTRO) uses TCN/CNN on mel at 10 ms + madmom-style peak picking (F1 ~0.90+). SSL encoders (MERT, wav2vec) + task head + optional multi-task (onset+pitch) are recent. Alternatives to binary frame labels: Gaussian bumps, time-to-event (TTE/TSE) density models. Seq2seq Transformers output sparse onset tokens directly (piano transcription). Our dense path skips peak picking; event path uses K-slot set prediction (~30% plateau).

**Implication:** Highest ROI changes likely **POST** (peak pick + threshold sweep on frozen checkpoints) and **TRAIN targets** (Gaussian already in code, unused). Next tier: **MODEL** (TCN/Seq-U-Net, larger capacity, fine-tune MERT not frozen cache). Seq2seq is viable but larger refactor. Domain gap: game/EDM charts ≠ piano/percussion benchmarks — expect lower absolute F1; scaling train data already helps (EXP-19→20).

**Open:** Does madmom-style peak picking on dense MERT logits close recall gap without retrain? Fine-tune vs frozen MERT?

**Related:** [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md), EXP-19/20/21, [onset_output_targets_planning.md](../onset_output_targets_planning.md) § External literature

---

## Session 2026-06-06 — multi-song dense overfit

### NOTE-20260606-17: Multi-song eval missing normalization

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-06 17:17:55 |
| **Tags**      | metric, pre, dense  |

**Context:** EXP-20260606-17 3-song run initially reported ~4% mean event F1 while training val F1 ~99.9%.

**Discovery:** `eval_dense_event_f1` loads features through `create_dataset`, which always applies `normalize_onset_spectrogram`. `eval_dense_event_f1_for_pair` used raw `load_onset_features`. Models trained on normalized MERT inputs produce near-zero logits on raw features.

**Implication:** Multi-song **did** overfit all three charts (~99.9% event F1 after fix). Adding a few songs at 100 ep does not inherently collapse metrics at N=3. Fix in `dense_overfit_eval.py`.

**Related:** EXP-20260606-17, `scripts/run_dense_multi_song_overfit.py`

---

## Session 2026-06-06 — Training stability

### NOTE-20260606-16: Dense seed must precede model init

| Field         | Value                         |
| ------------- | ----------------------------- |
| **Timestamp** | 2026-06-06 16:54:25           |
| **Tags**      | train, reproducibility, dense |

**Context:** EXP-12 hit ~98% event F1 on tide; EXP-13/14 with same config + seed 42 stuck ~35%. User blocked further work until training is stable.

**Discovery:** `run_train_from_config` built the U-Net **before** `_fit_and_save_model` called `tf.random.set_seed(42)`. Weight initialization was unseeded — effectively a lottery each run. After `reproducibility.apply_training_seed()` before model build, two 100-epoch runs are bit-identical (~95% event F1, val_pr_auc 0.991).

**Implication:** EXP-13/14 rankings (MERT > mel) may still hold but MERT numbers were invalid. Re-run suite with fix. Always seed before model construction.

**Related:** EXP-15, EXP-13, EXP-14, `stepcovnet/reproducibility.py`, `scripts/check_dense_mert_reproducibility.py`

---

## Session 2026-06-06 — Documentation conventions

### NOTE-20260606-15: Timestamp format — system clock at write

| Field         | Value               |
| ------------- | ------------------- |
| **Timestamp** | 2026-06-06 16:37:35 |
| **Tags**      | meta, documentation |

**Context:** User asked that all research timestamps use full `YYYY-MM-DD HH:MM:SS` and that agents capture the machine clock when logging — not estimated or `(approx.)` suffixes.

**Discovery:** Prior rule allowed backfilled stagger times with `(approx.)`; many index rows and entries used that suffix.

**Implication:** `.cursor/rules/research-notebook.mdc`, skills, templates, and agent docs now require reading system time at write (shell `Get-Date` / `date`). Strip `(approx.)` from existing entries; new logs use live clock only.

**Related:** JRN-20260606-08, `research-session-workflow` skill

---

## Session 2026-06-06 — Formulation bisection (EXP-11)

### NOTE-20260606-14: Uniform grid oracle matches F1 plateau

| Field         | Value                        |
| ------------- | ---------------------------- |
| **Timestamp** | 2026-06-06 17:20:00          |
| **Tags**      | formulation, overfit, oracle |

**Context:** User asked whether tide ~30% F1 is a code bug before starting full val training.

**Discovery:**

- Baseline MERT checkpoint: **927/1024** slots conf ≥ 0.5 (mass over-firing), 233 TP, 200 Hungarian-L1 within tol.
- **Oracle:** only **195/634** GT onsets have a uniform-grid slot within 20 ms (~31%).
- Half-cheat A (GT refs + learn Δ): 32% F1 — anchoring helps slightly, not overfit.
- Half-cheat B (uniform + frozen Δ): 40% F1 but **159 TP, 0 FP** — confidence-only on fixed grid tops out at oracle-like recall.

**Implication:** Smoke gate failure is **formulation**, not wiring. Current K-query head cannot memorize tide without shortcuts. Next: dense MERT control on tide (EXP-20260606-12), then **AR seq2seq** prototype ([AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md), NOTE-20260614-01).

**Related:** EXP-11, `scripts/run_overfit_tide_bisection.py`, `diagnostics.oracle_uniform_grid_coverage`

---

## Session 2026-06-06 — Hungarian training loss and conv1d zero-F1 debug

### NOTE-20260606-13: Conv1d zero F1 — confidence collapse from ordered training

| Field         | Value                                                              |
| ------------- | ------------------------------------------------------------------ |
| **Timestamp** | 2026-06-06 14:12:41                                                |
| **Tags**      | failure-modes, overfit, debug                                      |
| **Related**   | EXP-20260606-07, EXP-20260606-08, `scripts/debug_onset_overfit.py` |

**Context:** EXP-20260606-07 conv1d reached 0 TP / 0 FP / 634 FN after 50 ep (F1 = 0), while EXP-20260606-02 raw conv1d reached ~28% F1 at ~100 ep.

**Discovery** (diagnostics on `models_wsl/overfit_tide/conv1d/onset_event_model.keras` before Hungarian retrain):

| Signal                            | Value                                   |
| --------------------------------- | --------------------------------------- |
| Max confidence                    | **0.00023** (all slots ≪ 0.5 threshold) |
| Ordered pairs within 20 ms        | **0 / 634**                             |
| Hungarian L1 pairs within 20 ms   | **162 / 634**                           |
| Hungarian eval pairs within 20 ms | **200 / 634**                           |

- Pred times span the song (~0.07–128 s); times are not stuck at one point.
- With **ordered** training, uniform grid slot _i_ → _i_-th sorted GT: **zero** pairs within tolerance → loss pushes **all** confidences toward 0.
- Eval then has nothing above threshold → 0 TP despite ~200 time-accurate Hungarian eval pairings.

**Implication:** Conv1d 0 F1 in EXP-07 is primarily a **train/eval + ordered-assignment** failure mode, not a crash or empty forward pass. Hungarian training should allow confidence to rise on the ~162–200 near matches. EXP-02’s ~28% F1 likely benefited from longer training and/or slightly different dynamics before the suite config.

**Open:** ~~After EXP-08 rerun, compare conv1d F1 to EXP-07.~~ EXP-08: conv1d ~27% F1 (was 0% in EXP-07); mel ~28%; MERT ~29% at 50 ep — still no overfit.

---

## Session 2026-06-06 — Formulation bisection (EXP-11)

---

### NOTE-20260606-12: Hungarian L1 training loss implemented

| Field         | Value                                           |
| ------------- | ----------------------------------------------- |
| **Timestamp** | 2026-06-06 14:10:00                             |
| **Tags**      | training, design-decision                       |
| **Related**   | `losses.py`, `matching.py`, EXP-20260606-08, A5 |

**Context:** EXP-20260606-07 smoke showed weak learning; leading hypothesis was ordered train vs Hungarian eval.

**Discovery:**

- `_l1_training_losses` now calls `matching.assign_onset_pairs_l1` (Hungarian on raw L1 cost, no tolerance gate on pairing).
- Confidence targets unchanged: within `tolerance_sec` → push conf toward 1; outside → toward 0; unmatched slots → 0.
- Eval still uses tolerance-gated Hungarian (`match_onsets_numpy`).

**Implication:** Training gradients follow nearest-slot pairing, aligned with eval semantics. Tide suite re-run at 50 ep compares to EXP-07.

---

---

## Session 2026-06-06 — Pipeline architecture (pre / model / post)

### NOTE-20260606-11: Hungarian matching and clustered predictions

| Field         | Value                                                                      |
| ------------- | -------------------------------------------------------------------------- |
| **Timestamp** | 2026-06-06 11:12:00                                                        |
| **Tags**      | evaluation, matching, failure-modes                                        |
| **Related**   | `matching.py`, `metrics.py`, `losses.py`, `inference.py`; NOTE-20260606-02 |

**Context:** User concern — if the model learns “onset-like” audio at one time, it might fire many query slots near that single time. Would Hungarian matching still yield a high score, or would missed onsets elsewhere force predictions to spread?

**Discovery — Hungarian is one-to-one:**

- Build pairwise costs `|pred_i − gt_j|`. Pairs within **20 ms** tolerance get low cost; outside get a huge penalty (`1e6`).
- Hungarian picks a **global assignment** that minimizes total cost, with **at most one pred per GT** and **at most one GT per pred**.
- After matching, **metrics** (`metrics.py`): matched + conf ≥ threshold → TP; unmatched pred + high conf → **FP**; unmatched GT → **FN**.

**Toy example — clustering fails F1:**

- GT: 100 onsets spread across the song (1 s, 2 s, …).
- Model: 50 slots at ~10.0 s (all high confidence), rest low.
- Hungarian: **one** pred matches the GT nearest 10 s → at most **1 TP**.
- Other 49 clustered preds: **unmatched** → if conf ≥ 0.5, **49 FP**.
- Other 99 GT onsets: no pred within 20 ms → **99 FN**.
- Recall ≈ 1/100, precision ≈ 1/50 → **F1 ≈ 0**. Clustering does **not** game the metric.

**What _would_ raise F1:** preds **spread** so many GT each get a unique partner within tolerance (and confidence on).

**Training vs eval:**

- **Eval** uses Hungarian with tolerance gate (`match_onsets_numpy`).
- **Training loss** uses Hungarian L1 without tolerance gate on assignment (`assign_onset_pairs_l1`); confidence still uses per-pair tolerance after matching.

**Post-processing gap:**

- `inference.predict_onsets` applies **min onset distance** (default 50 ms) after thresholding — collapses nearby preds for deployment.
- **Val / eval F1** today uses **raw K slots** — no min-gap. Extra clustered high-conf slots count as **FP** until post-processing is added to the metric path (or model learns to suppress duplicates).

**Implication:** Hungarian + F1 already penalizes “many preds, one place.” Remaining risks: duplicate preds below conf threshold; chart **clusters** still need distinct preds within tolerance.

**Open:** Add optional min-gap/NMS to metric pipeline for parity with inference? Diversity or repulsion loss on query times?

---

## Session 2026-06-06 — Hungarian training loss and conv1d zero-F1 debug

---

### NOTE-20260606-10: Pipeline — pre, model, post, and metrics

| Field         | Value                                                                                                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp** | 2026-06-06 11:00:00                                                                                                                                                              |
| **Tags**      | architecture, research-strategy                                                                                                                                                  |
| **Related**   | [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) (canonical); `audio.py`, `frontend.py`, `models.py`, `inference.py`, `metrics.py`, `losses.py`; planning § problem reframed |

**Context:** User’s mental model for the full onset system: audio → preprocessing → core model → raw outputs → postprocessing → final predictions → compare to GT for metrics and training feedback. Open research question: best preprocessing, best core model, best postprocessing — each swappable independently where possible.

**Discovery:** The unified pipeline (PRE → MODEL → POST → METRICS → training feedback), repo mapping, ablation axes, and mermaid/ASCII diagrams live in **[PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)**. That doc is the north star for implementation, tests, and the paper methods section.

**Implication — research should isolate one stage at a time** (see architecture doc § Ablation matrix):

1. **PRE** — raw vs mel vs MERT on same event head + metrics.
2. **MODEL** — query slots vs dense vs seq2seq on same PRE where possible.
3. **POST** — threshold / min-gap sweeps on fixed checkpoints.
4. **METRICS / train** — matching and loss alignment with eval.

The “best model in the middle” only makes sense once PRE and POST+metric contract are fixed or explicitly co-varied in a grid.

**Open:** See [PIPELINE_ARCHITECTURE.md § Open design questions](PIPELINE_ARCHITECTURE.md#open-design-questions).

**Next:** Log each ablation as `EXP-…` with stage tag (`pre` / `model` / `post` / `metric` / `train`); update [PAPER_OUTLINE.md](PAPER_OUTLINE.md) when results land.

---

---

## Session 2026-06-06 — Research documentation

### NOTE-20260606-09: Research notebook — experiments plus discussion

| Field         | Value                                 |
| ------------- | ------------------------------------- |
| **Timestamp** | 2026-06-06 10:30:00                   |
| **Tags**      | meta, workflow                        |
| **Related**   | `.cursor/rules/research-notebook.mdc` |

**Context:** User wants documentation beyond formal experiments — tracking discussions and discoveries as we go, toward a future paper.

**Discovery:**

- **Experiment log** = reproducible runs (what we tried, numbers, accept/reject).
- **Discussion notes** (this file) = conversational insights, Q&A, design reasoning, open questions.
- **Planning doc** = stable synthesis; **paper outline** = promoted claims only.

**Implication:** Agent should prepend here after substantive threads, not only after GPU runs (newest entries first).

---

## Session 2026-06-06 — Pipeline architecture (pre / model / post)

---

## Session 2026-06-06 — Onset event model evaluation & design

### NOTE-20260606-08: Raw audio vs mel — two separate difficulties

| Field         | Value                                                                          |
| ------------- | ------------------------------------------------------------------------------ |
| **Timestamp** | 2026-06-06 09:21:00                                                            |
| **Tags**      | features, open-question                                                        |
| **Related**   | EXP-20260606-01, EXP-20260606-06, EXP-20260606-07, planning § raw audio vs mel |

**Context:** Whether raw-audio event onsets are “futile” compared to mel/MERT.

**Discovery:**

- **Difficulty (1):** input representation — raw Conv1D vs cached mel/MERT vs mel-in-graph.
- **Difficulty (2):** output formulation — dense frames vs K queries vs seq2seq; matching and loss.
- Tide ~28% F1 with learnable deltas is mostly a (2) story; dense MERT val F1 ~0.36 is (1)+(2) with an **easier dense target**.
- Raw is **harder**, not disproven; fair test is same event head + same metric, swap frontend only.

**Implication:** EXP-20260606-07 (ordered train) and EXP-20260606-08 (Hungarian train) show pre-processing helps; conv1d 0% F1 in EXP-07 was ordered-assignment collapse (NOTE-13), not raw-audio failure.

**Open:** Mel-in-graph vs cache-only — engineering vs learnability tradeoff.

---

## Session 2026-06-06 — Research documentation

---

### NOTE-20260606-07: `num_queries` vs `n_max_onsets`

| Field         | Value                                                      |
| ------------- | ---------------------------------------------------------- |
| **Timestamp** | 2026-06-06 09:18:00                                        |
| **Tags**      | architecture, config                                       |
| **Related**   | EXP-20260606-04, `configs/event/audio_baseline.json` |

**Context:** Whether K must equal tide’s 634 onsets for overfit to succeed.

**Discovery:**

- Need **K ≥ max onsets in batch**, not **K = N** per song.
- With GT refs + frozen deltas, K=1024 also reached F1=1.0; extra slots stayed quiet.
- Setting K=N for one song is a convenience, not a requirement for the metric pipeline.

**Implication:** Baseline config using 1024/1024 is fine; enforce `num_queries >= max_steps_per_chart`.

---

---

### NOTE-20260606-06: Overfit shortcuts (GT refs and frozen deltas)

| Field         | Value                                       |
| ------------- | ------------------------------------------- |
| **Timestamp** | 2026-06-06 09:15:00                         |
| **Tags**      | overfit, design-decision                    |
| **Related**   | EXP-20260606-03, `trainers.py`, `models.py` |

**Context:** What `overfit_one_song` mode with GT-aligned refs and frozen deltas is for.

**Discovery:**

- **`_query_ref_normalized_from_batch`:** initializes query time logits from sorted GT (overfit only).
- **`learn_time_delta=False`:** times are `sigmoid(ref)` only — model learns **confidence**, not timing from audio.
- Reaching F1 = 1.0 under this setup validates **loss + metrics + training loop**, not audio→timing.

**Implication:** Do not cite tide F1=1.0 as evidence the raw frontend works. Normal training uses uniform grid + learnable deltas.

---

---

### NOTE-20260606-05: Count from times, not a separate head

| Field         | Value                          |
| ------------- | ------------------------------ |
| **Timestamp** | 2026-06-06 09:12:00            |
| **Tags**      | design-decision, output-format |
| **Related**   | planning § problem reframed    |

**Context:** Whether to predict onset count separately from times.

**Discovery:**

- Ground truth count is `N = len(times)`; times are the primary object.
- Model should emit K slots; **count = number of slots with confidence above threshold** (after matching/dedup if needed).
- A separate count head adds redundancy and another failure mode.

**Implication:** Design reviews should treat confidence + time as the only outputs; count is derived at inference.

---

---

### NOTE-20260606-04: Gap and cluster charts break index alignment

| Field         | Value                          |
| ------------- | ------------------------------ |
| **Timestamp** | 2026-06-06 09:09:00            |
| **Tags**      | failure-modes, charts          |
| **Related**   | planning § gap/cluster example |

**Context:** Why strict slot-i → i-th-GT assignment is fragile for StepMania charts.

**Discovery:**

- Charts have **clusters** (many steps close together) and **gaps** (long silent spans).
- A uniform query grid spaced across the song does not align with “onset index” in time order when local density varies.
- Hungarian eval can still match by time, but **ordered training** forces wrong pairings when the grid doesn’t match GT density.

**Implication:** Prefer **time-based matching** for training, not index-based, for general charts. Tide is unusually regular on beats but still irregular in exact spacing.

---

---

### NOTE-20260606-03: Seq2seq goal vs DETR-style K slots

| Field         | Value                                               |
| ------------- | --------------------------------------------------- |
| **Timestamp** | 2026-06-06 09:06:00                                 |
| **Tags**      | architecture, product                               |
| **Related**   | planning § “Mental model: seq2seq vs what we built” |

**Context:** How the natural output (variable-length sorted times) relates to the current head.

**Discovery:**

- **Product framing:** audio → sorted list of onset times in seconds (variable N).
- **Implementation:** DETR-like — K learned queries each predict `(time, confidence)`; N inferred post-hoc from survivors above threshold.
- Alternative mental model: autoregressive seq2seq emitting one time per step until EOS — not what we built.

**Implication:** K must be ≥ max onsets per chart; matching strategy (ordered vs Hungarian) matters more for seq2seq-like correctness than for dense frames.

**Open:** Whether seq2seq is worth a parallel prototype vs improving query + matching.

---

---

### NOTE-20260606-02: Train ordered vs eval Hungarian

| Field         | Value                                         |
| ------------- | --------------------------------------------- |
| **Timestamp** | 2026-06-06 09:03:00                           |
| **Tags**      | training, evaluation, mismatch                |
| **Related**   | `losses.py`, EXP-20260606-02, EXP-20260606-03 |

**Context:** Why tide overfit with learnable deltas plateaued ~28% F1 despite partial learning.

**Discovery:**

- **Training loss** assigns query slot _i_ to the _i_-th sorted GT onset (ordered).
- **Eval** assigns queries to GT via **Hungarian** on times, then thresholds confidence.
- With a **uniform query time grid**, initial mean ordered L1 to GT was very large (~8.7 s) — slots start far from their ordered partners.
- High confidence on wrong slots hurts eval F1 even if some times move toward GT.

**Implication:** Train/eval mismatch **was** a major issue for conv1d (NOTE-20260606-13). Hungarian L1 training loss is now default (EXP-20260606-08).

**Open:** ~~Measure whether Hungarian training closes the gap on tide without GT refs.~~ Rerun logged in EXP-20260606-08.

---

---

### NOTE-20260606-01: How event eval works

| Field         | Value                                                          |
| ------------- | -------------------------------------------------------------- |
| **Timestamp** | 2026-06-06 09:00:00                                            |
| **Tags**      | evaluation, metrics                                            |
| **Related**   | `metrics.py`, `matching.py`, planning § evaluation walkthrough |

**Context:** Walkthrough of how validation/eval F1 is computed for the event onset model.

**Discovery:**

- Model outputs **K fixed query slots** `(time, confidence)`, not a variable-length list.
- GT is padded to `n_max_onsets`; mask marks real vs pad positions.
- **Hungarian matching** pairs predicted times to GT times (minimize total L1), then confidence thresholding yields TP/FP/FN.
- K can exceed GT count N — extra slots should stay low-confidence (FP if they fire above threshold).

**Implication:** Eval is set-matching in time, not “slot i must equal onset i.” Good for irregular spacing; training now uses Hungarian L1 assignment (see NOTE-20260606-12).

**Open:** ~~Should training use the same Hungarian matching as eval?~~ Resolved — Hungarian L1 in training (EXP-20260606-08).
