# Antigravity run_workflow injects WORKSPACES/args/phase/log/parallel/agent and supports top-level await.
# ruff: noqa: F704, F821
"""Antigravity run_workflow harness for skill-local and rule-local agent evals.

Run via Antigravity's ``run_workflow`` tool with optional ``Args``:
    skill: Optional skill name (e.g. "onset-evaluation") to run only its local suite
    rule: Optional rule name/stem (e.g. "git-and-ci.md") to run only its local suite
    mode: "all" (default), "triggers", or "behavior"
    tag: Output run name under ``logs/agent_evals/`` (default: UTC timestamp)
    case_ids: Optional list of case IDs to run
    baseline: Optional repo-relative predictions JSON path for hill-climb diff
    model: Agent model tier (default: "flash")
"""

import datetime
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(WORKSPACES[0]) if WORKSPACES else pathlib.Path.cwd()
SKILL = str(args.get("skill") or "")
RULE = str(args.get("rule") or "")
MODE = str(args.get("mode") or "all").lower()
MODEL = str(args.get("model") or "flash")
BASELINE = str(args.get("baseline") or "")
TAG = str(
    args.get("tag")
    or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
)
FILTER_IDS = set(args.get("case_ids") or [])

EVAL_SCRIPT = ROOT / "scripts" / "eval_agent_setup.py"

list_cmd = [sys.executable, str(EVAL_SCRIPT), "list-suites", "--format", "json"]
if SKILL:
    list_cmd.extend(["--skill", SKILL])
if RULE:
    list_cmd.extend(["--rule", RULE])

SUITE_PATHS = [
    pathlib.Path(p)
    for p in json.loads(
        subprocess.run(
            list_cmd, cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    )
]
if not SUITE_PATHS:
    raise ValueError(f"No local eval suites matched skill={SKILL!r} rule={RULE!r}")

MERGED_TRIGGERS = []
MERGED_BEHAVIORS = []
for suite_path in SUITE_PATHS:
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    MERGED_TRIGGERS.extend(data.get("trigger_cases", []))
    MERGED_BEHAVIORS.extend(data.get("behavior_cases", []))

CATALOG = json.loads(
    subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "render-catalog", "--format", "json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
)
CATALOG_MD = subprocess.run(
    [sys.executable, str(EVAL_SCRIPT), "render-catalog", "--format", "markdown"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
).stdout

RULE_IDS = [item["id"] for item in CATALOG.get("rules", [])]
SKILL_IDS = [item["id"] for item in CATALOG.get("skills", [])]

TRIGGER_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["rules", "skills", "reason"],
}

BEHAVIOR_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "commands": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["response", "commands"],
}

GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["criterion", "passed", "reason"],
            },
        }
    },
    "required": ["results"],
}

trigger_predictions = {}
behavior_predictions = {}

if MODE in ("all", "triggers"):
    phase("Triggers (Layer 2)")
    trig_cases = [
        c for c in MERGED_TRIGGERS if not FILTER_IDS or c["id"] in FILTER_IDS
    ]
    log(
        f"Running {len(trig_cases)} trigger cases across {len(SUITE_PATHS)} suite(s) on model={MODEL}"
    )

    def _trigger_prompt(case):
        return (
            "You are routing a StepCOVNet user request strictly from the rule and skill "
            "descriptions below.\n"
            "Do NOT read workspace files or run tools.\n"
            "Decide which model_decision rules (if any) and which skills (if any) should "
            "activate for this user message.\n"
            "If the request is a simple code read, math explanation, or config lookup that "
            "does not need specialized playbooks or git/env procedures, return empty lists.\n\n"
            f"Valid rule IDs: {json.dumps(RULE_IDS)}\n"
            f"Valid skill IDs: {json.dumps(SKILL_IDS)}\n\n"
            f"{CATALOG_MD}\n"
            f"User message: {case['prompt']}\n\n"
            "Return only valid IDs from the lists above via finish."
        )

    trig_outputs = (
        await parallel(
            trig_cases,
            lambda c: agent(
                _trigger_prompt(c),
                schema=TRIGGER_SCHEMA,
                role=f"Trigger {c['id']}",
                type_name="research",
                model=MODEL,
            ),
        )
        if trig_cases
        else []
    )
    for case, out in zip(trig_cases, trig_outputs, strict=True):
        trigger_predictions[case["id"]] = out
    log(f"Completed {len(trigger_predictions)} trigger predictions.")

if MODE in ("all", "behavior"):
    phase("Behavior (Layer 3)")
    beh_cases = [
        c for c in MERGED_BEHAVIORS if not FILTER_IDS or c["id"] in FILTER_IDS
    ]
    log(
        f"Running {len(beh_cases)} behavior cases across {len(SUITE_PATHS)} suite(s) on model={MODEL}"
    )

    def _behavior_prompt(case):
        return (
            "You are operating inside the StepCOVNet repository.\n"
            "Consult AGENTS.md and any relevant files in .agents/rules/ and .agents/skills/ "
            "before answering so you follow exact repository conventions.\n"
            "Provide the exact answer you would give (or plan you would follow) plus any "
            "exact shell/python commands you would run under `commands`.\n\n"
            f"Scenario / User prompt:\n{case['prompt']}"
        )

    beh_outputs = (
        await parallel(
            beh_cases,
            lambda c: agent(
                _behavior_prompt(c),
                schema=BEHAVIOR_SCHEMA,
                role=f"Behavior {c['id']}",
                type_name="research",
                model=MODEL,
            ),
        )
        if beh_cases
        else []
    )

    rubric_pairs = [
        (case, out)
        for case, out in zip(beh_cases, beh_outputs, strict=True)
        if case.get("rubric")
    ]
    log(f"Grading {len(rubric_pairs)} rubric-bearing behavior cases...")

    def _grader_prompt(pair):
        case, out = pair
        return (
            "Grade the candidate agent output strictly against each criterion below.\n"
            "Mark passed=true ONLY if the response/commands clearly satisfy the criterion "
            "without violating it.\n"
            "Preserve each exact criterion string verbatim.\n\n"
            f"Prompt:\n{case['prompt']}\n\n"
            f"Criteria:\n{json.dumps(case['rubric'], indent=2)}\n\n"
            f"Candidate response:\n{out.get('response', '')}\n\n"
            f"Candidate commands:\n{json.dumps(out.get('commands', []), indent=2)}"
        )

    graded_outputs = (
        await parallel(
            rubric_pairs,
            lambda p: agent(
                _grader_prompt(p),
                schema=GRADER_SCHEMA,
                role=f"Grade {p[0]['id']}",
                type_name="research",
                model="flash",
            ),
        )
        if rubric_pairs
        else []
    )

    graded_by_id = {
        case["id"]: graded.get("results", [])
        for (case, _), graded in zip(rubric_pairs, graded_outputs, strict=True)
    }
    for case, out in zip(beh_cases, beh_outputs, strict=True):
        behavior_predictions[case["id"]] = {
            "response": out.get("response", ""),
            "commands": out.get("commands", []),
            "rubric_results": graded_by_id.get(case["id"], []),
        }
    log(f"Completed {len(behavior_predictions)} behavior evaluations.")

phase("Score & Hill-Climb Diff")
eval_dir = ROOT / "logs" / "agent_evals"
eval_dir.mkdir(parents=True, exist_ok=True)
pred_path = eval_dir / f"{TAG}.json"
report_path = eval_dir / f"{TAG}_report.json"

pred_payload = {
    "schema_version": 1,
    "tag": TAG,
    "model": MODEL,
    "trigger_predictions": trigger_predictions,
    "behavior_predictions": behavior_predictions,
}
pred_path.write_text(json.dumps(pred_payload, indent=2) + "\n", encoding="utf-8")

score_cmd = [
    sys.executable,
    str(EVAL_SCRIPT),
    "score",
    "--predictions",
    str(pred_path),
    "--out",
    str(report_path),
]
if SKILL:
    score_cmd.extend(["--skill", SKILL])
if RULE:
    score_cmd.extend(["--rule", RULE])
if BASELINE:
    score_cmd.extend(["--baseline", str(ROOT / BASELINE)])

score_proc = subprocess.run(score_cmd, cwd=ROOT, capture_output=True, text=True)
log(score_proc.stdout.strip())
if score_proc.stderr.strip():
    log(f"STDERR: {score_proc.stderr.strip()}")
if score_proc.returncode != 0:
    log(f"Scorer exited with code {score_proc.returncode}")

print(score_proc.stdout, end="")
