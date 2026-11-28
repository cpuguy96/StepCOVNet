# Autoresearch profiles

A **profile** binds the generic [autoresearch loop](../SKILL.md) to concrete paths: orient command, plan file, run command, logs, success metric, preflight.

| Profile | File |
| ------- | ---- |
| PyTorch ITGPT-MERT onset hill-climbing | [itgpt-onset.md](itgpt-onset.md) |
| General EXPERIMENT_LOG research | [experiment-log.md](experiment-log.md) |

## Add a profile

1. Copy `experiment-log.md` → `profiles/<name>.md`.
2. Fill: success criteria, preflight, orient, plan artifact, run, log, profile-specific anti-spam.
3. Add a row to the table in [../SKILL.md](../SKILL.md) § Choose a profile.

**Custom profile in one prompt:** user sets `Goal:` and `Metric:` in the message; agent runs the generic loop without a new file if the session is short-lived.
