# Tide overfit — scheduled-sampling / decode experiments

Separate from the champion [`tide_overfit.json`](../../tide_overfit.json) track. These warm-start a teacher-pass checkpoint and ramp **scheduled sampling** toward free-run decode.

| Ver | File               | Was                   | Purpose                                              |
| --- | ------------------ | --------------------- | ---------------------------------------------------- |
| v1  | [v1.json](v1.json) | `decode/v2.json`      | SS warmup 15 + ramp 100 → p=1; full tide loss        |
| v2  | [v2.json](v2.json) | `decode/tide.json`    | SS ramp sketch (checkpoint `val_ar_decode_event_f1`) |
| v3  | [v3.json](v3.json) | `decode/perfect.json` | SS from perfect run1 checkpoint                      |

Not promoted to champion until a recipe passes **offline** free-run **634/634** ordered @ 20 ms without regressing teacher metrics.
