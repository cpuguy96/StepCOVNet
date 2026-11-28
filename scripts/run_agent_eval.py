#!/usr/bin/env python3
"""Harness-agnostic runner for StepCOVNet skill-local and rule-local agent evals.

Supports dumping structured prompt bundles for any coding-agent harness (Claude Code,
Codex, Cursor, Antigravity) or executing eval prompts directly via OpenAI,
Anthropic, Gemini REST APIs, or an arbitrary stdin/stdout CLI backend.

Usage (repo root, project venv):
    python scripts/run_agent_eval.py --rule git-and-ci.md --tag base --backend dump
    python scripts/run_agent_eval.py --skill onset-evaluation --tag cand --backend openai
    python scripts/run_agent_eval.py --tag cand --backend cmd --cmd "my-agent --json"
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_agent_setup  # noqa: E402

REPO = SCRIPTS_DIR.parent
JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\r?\n?(.*?)\r?\n?```",
    re.DOTALL | re.IGNORECASE,
)

TRIGGER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "rules": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reason", "rules", "skills"],
    "additionalProperties": False,
}

BEHAVIOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "commands": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["response", "commands"],
    "additionalProperties": False,
}

GRADER_SCHEMA: dict[str, Any] = {
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
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_trigger_prompt(
    case: dict[str, Any],
    rule_ids: list[str],
    skill_ids: list[str],
    catalog_md: str,
) -> str:
    """Format a Layer 2 routing prompt for a single trigger eval case.

    Args:
        case: Trigger eval case dict containing at least ``"prompt"``.
        rule_ids: Allowed ``model_decision`` rule IDs.
        skill_ids: Allowed skill directory names.
        catalog_md: Pre-rendered markdown catalog of rule and skill descriptions.

    Returns:
        Prompt asking the model to return JSON matching ``TRIGGER_SCHEMA``.
    """
    return (
        "You are routing a StepCOVNet user request strictly from the rule and skill "
        "descriptions below.\n"
        "Do NOT read workspace files or run tools.\n"
        "Decide which model_decision rules (if any) and which skills (if any) should "
        "activate for this user message.\n"
        "If the request is a simple code read, math explanation, or config lookup that "
        "does not need specialized playbooks or git/env procedures, return empty lists.\n\n"
        f"Valid rule IDs: {json.dumps(rule_ids)}\n"
        f"Valid skill IDs: {json.dumps(skill_ids)}\n\n"
        f"{catalog_md}\n"
        f"User message: {case['prompt']}\n\n"
        "Return only valid JSON matching the requested schema: "
        '{"reason": "...", "rules": [...], "skills": [...]}.'
    )


def build_behavior_prompt(case: dict[str, Any]) -> str:
    """Format a Layer 3 behavior prompt for a single behavior eval case.

    Args:
        case: Behavior eval case dict containing at least ``"prompt"``.

    Returns:
        Prompt asking the model to return JSON matching ``BEHAVIOR_SCHEMA``.
    """
    return (
        "You are operating inside the StepCOVNet repository.\n"
        "Consult AGENTS.md and any relevant files in .agents/rules/ and .agents/skills/ "
        "before answering so you follow exact repository conventions.\n"
        "Provide the exact answer you would give (or plan you would follow) plus any "
        "exact shell/python commands you would run under `commands`.\n\n"
        f"Scenario / User prompt:\n{case['prompt']}\n\n"
        "Return only valid JSON matching the requested schema: "
        '{"response": "...", "commands": [...]}.'
    )


def build_grader_prompt(
    case: dict[str, Any],
    candidate_output: dict[str, Any],
) -> str:
    """Format a rubric grading prompt for one behavior prediction.

    Args:
        case: Behavior eval case dict with ``"prompt"`` and ``"rubric"``.
        candidate_output: Candidate output dict containing ``response`` and ``commands``.

    Returns:
        Prompt asking the grader to return JSON matching ``GRADER_SCHEMA``.
    """
    return (
        "Grade the candidate agent output strictly against each criterion below.\n"
        "Mark passed=true ONLY if the response/commands clearly satisfy the criterion "
        "without violating it.\n"
        "Preserve each exact criterion string verbatim.\n\n"
        f"Prompt:\n{case['prompt']}\n\n"
        f"Criteria:\n{json.dumps(case.get('rubric', []), indent=2)}\n\n"
        f"Candidate response:\n{candidate_output.get('response', '')}\n\n"
        f"Candidate commands:\n{json.dumps(candidate_output.get('commands', []), indent=2)}\n\n"
        "Return only valid JSON matching the requested schema: "
        '{"results": [{"criterion": "...", "passed": true/false, "reason": "..."}]}.'
    )


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text or a markdown code fence.

    Args:
        raw_text: Raw model response string.

    Returns:
        Parsed JSON object dict.

    Raises:
        ValueError: If no valid JSON object can be parsed from ``raw_text``.
    """
    candidate = raw_text.strip()
    fence = JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(
                f"Expected JSON object in response: {raw_text!r}"
            ) from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON object in response: {raw_text!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Send an HTTP POST request with a JSON body and parse the JSON response.

    Args:
        url: Destination HTTPS endpoint URL.
        headers: HTTP request headers.
        payload: Request body dictionary serialized as UTF-8 JSON.
        timeout_sec: Socket timeout in seconds.

    Returns:
        Parsed JSON response object.

    Raises:
        RuntimeError: If the HTTP request fails or returns non-JSON text.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc.reason}") from exc
    return extract_json_object(raw)


def run_cmd_backend(
    prompt: str,
    schema: dict[str, Any],
    cmd: str,
    *,
    cwd: pathlib.Path,
) -> dict[str, Any]:
    """Invoke a shell/CLI backend passing ``prompt`` on stdin and schema in env.

    Args:
        prompt: Prompt text sent to the subprocess stdin.
        schema: JSON Schema passed to the subprocess in ``EVAL_SCHEMA_JSON``.
        cmd: Shell command template or CLI executable invoked via ``subprocess.run``.
        cwd: Working directory for the command.

    Returns:
        Parsed JSON object emitted by the command on stdout.

    Raises:
        RuntimeError: If the command exits non-zero.
    """
    env = dict(os.environ)
    env["EVAL_SCHEMA_JSON"] = json.dumps(schema)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        input=prompt,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Backend command exited with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return extract_json_object(completed.stdout)


def run_openai_backend(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Call an OpenAI-compatible Chat Completions endpoint for JSON output.

    Args:
        prompt: User prompt text.
        schema: JSON Schema enforced via ``response_format``.
        model: Model name (for example ``gpt-4o-mini``).

    Returns:
        Parsed JSON object from the first completion choice.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is unset or the API fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip(
        "/"
    )
    response = _post_json(
        f"{base_url}/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_eval",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    )
    content = response["choices"][0]["message"]["content"]
    return extract_json_object(content)


def run_anthropic_backend(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Call the Anthropic Messages API requesting structured JSON output.

    Args:
        prompt: User prompt text.
        schema: JSON Schema included in the prompt instruction.
        model: Claude model identifier.

    Returns:
        Parsed JSON object from the assistant text response.

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is unset or the API fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --backend anthropic")
    full_prompt = (
        f"{prompt}\n\nOutput ONLY raw JSON conforming to schema:\n{json.dumps(schema)}"
    )
    response = _post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
        },
    )
    blocks = response.get("content", [])
    text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
    return extract_json_object(text)


def run_gemini_backend(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Call the Gemini ``generateContent`` REST API requesting JSON output.

    Args:
        prompt: User prompt text.
        schema: JSON Schema passed in ``generationConfig.responseJsonSchema``.
        model: Gemini model identifier (for example ``gemini-2.5-flash``).

    Returns:
        Parsed JSON object from the first candidate text part.

    Raises:
        RuntimeError: If ``GEMINI_API_KEY`` is unset or the API fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for --backend gemini")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    response = _post_json(
        url,
        {"Content-Type": "application/json"},
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        },
    )
    parts = response["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    return extract_json_object(text)


def build_prompt_bundle(
    suite_paths: list[pathlib.Path],
    catalog: dict[str, list[dict[str, str]]],
    catalog_md: str,
    *,
    mode: str = "all",
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Assemble harness-agnostic trigger and behavior prompts for the target suites.

    Args:
        suite_paths: Discovered local suite JSON paths.
        catalog: Live catalog dict returned by ``eval_agent_setup.load_live_catalog()``.
        catalog_md: Markdown-rendered catalog string.
        mode: Evaluation slice (``"all"``, ``"triggers"``, or ``"behavior"``).
        case_ids: Optional filter keeping only listed case IDs.

    Returns:
        Dict with ``trigger_prompts``, ``behavior_prompts``, and ``schemas``.
    """
    rule_ids = [r["id"] for r in catalog.get("rules", [])]
    skill_ids = [s["id"] for s in catalog.get("skills", [])]

    trigger_cases: list[dict[str, Any]] = []
    behavior_cases: list[dict[str, Any]] = []
    for path in suite_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        trigger_cases.extend(data.get("trigger_cases", []))
        behavior_cases.extend(data.get("behavior_cases", []))

    filter_ids = case_ids or set()
    trigger_prompts: list[dict[str, Any]] = []
    if mode in ("all", "triggers"):
        for case in trigger_cases:
            if filter_ids and case["id"] not in filter_ids:
                continue
            trigger_prompts.append(
                {
                    "id": case["id"],
                    "prompt": build_trigger_prompt(
                        case, rule_ids, skill_ids, catalog_md
                    ),
                }
            )

    behavior_prompts: list[dict[str, Any]] = []
    if mode in ("all", "behavior"):
        for case in behavior_cases:
            if filter_ids and case["id"] not in filter_ids:
                continue
            behavior_prompts.append(
                {
                    "id": case["id"],
                    "case": case,
                    "prompt": build_behavior_prompt(case),
                }
            )

    return {
        "schema_version": 1,
        "suite_paths": [str(p) for p in suite_paths],
        "trigger_prompts": trigger_prompts,
        "behavior_prompts": behavior_prompts,
        "schemas": {
            "trigger": TRIGGER_SCHEMA,
            "behavior": BEHAVIOR_SCHEMA,
            "grader": GRADER_SCHEMA,
        },
    }


def _make_backend_caller(
    *,
    backend: str,
    model: str,
    cmd: str,
    root: pathlib.Path,
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Build a uniform ``(prompt, schema) -> dict`` callable for the chosen backend.

    Args:
        backend: One of ``"cmd"``, ``"openai"``, ``"anthropic"``, ``"gemini"``.
        model: Model identifier passed to API backends.
        cmd: Command string passed to ``subprocess.run`` for ``"cmd"``.
        root: Repository root working directory.

    Returns:
        Callable accepting ``(prompt, schema)`` and returning a parsed JSON dict.

    Raises:
        ValueError: If ``backend == "cmd"`` without ``cmd``, or ``backend`` is unknown.
    """
    if backend == "cmd":
        if not cmd.strip():
            raise ValueError("--cmd is required when --backend cmd is selected")
        return lambda prompt, schema: run_cmd_backend(prompt, schema, cmd, cwd=root)
    if backend == "openai":
        return lambda prompt, schema: run_openai_backend(prompt, schema, model=model)
    if backend == "anthropic":
        return lambda prompt, schema: run_anthropic_backend(prompt, schema, model=model)
    if backend == "gemini":
        return lambda prompt, schema: run_gemini_backend(prompt, schema, model=model)
    raise ValueError(f"Unsupported backend: {backend!r}")


def _default_model_for_backend(backend: str) -> str:
    """Return a sensible default model name when ``--model`` is omitted.

    Args:
        backend: Selected backend identifier.

    Returns:
        Default model string for the backend.
    """
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-latest",
        "gemini": "gemini-2.5-flash",
        "cmd": "default",
        "dump": "dump",
    }
    return defaults.get(backend, "default")


def _execute_bundle(
    bundle: dict[str, Any],
    *,
    call_candidate: Callable[[str, dict[str, Any]], dict[str, Any]],
    call_grader: Callable[[str, dict[str, Any]], dict[str, Any]],
    tag: str,
    model: str,
) -> dict[str, Any]:
    """Execute trigger, behavior, and rubric grading prompts cleanly.

    Args:
        bundle: Prompt bundle returned by ``build_prompt_bundle()``.
        call_candidate: Backend callable used for trigger and behavior prompts.
        call_grader: Backend callable used for rubric grading prompts.
        tag: Output run tag stored in the predictions payload.
        model: Candidate model identifier stored in the predictions payload.

    Returns:
        Predictions payload compatible with ``scripts/eval_agent_setup.py score``.
    """
    trigger_predictions = {
        item["id"]: call_candidate(item["prompt"], TRIGGER_SCHEMA)
        for item in bundle["trigger_prompts"]
    }
    behavior_predictions: dict[str, Any] = {}
    for item in bundle["behavior_prompts"]:
        out = call_candidate(item["prompt"], BEHAVIOR_SCHEMA)
        case = item["case"]
        rubric_results: list[dict[str, Any]] = []
        if case.get("rubric"):
            graded = call_grader(build_grader_prompt(case, out), GRADER_SCHEMA)
            rubric_results = list(graded.get("results", []))
        behavior_predictions[item["id"]] = {
            "response": out.get("response", ""),
            "commands": out.get("commands", []),
            "rubric_results": rubric_results,
        }
    return {
        "schema_version": 1,
        "tag": tag,
        "model": model,
        "trigger_predictions": trigger_predictions,
        "behavior_predictions": behavior_predictions,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the harness-agnostic agent eval runner.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Run or export StepCOVNet skill-local and rule-local agent evals."
    )
    parser.add_argument("--skill", default="", help="Optional skill directory name.")
    parser.add_argument("--rule", default="", help="Optional rule filename or stem.")
    parser.add_argument(
        "--mode",
        choices=("all", "triggers", "behavior"),
        default="all",
        help="Evaluation slice to execute (default: all).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Output tag under logs/agent_evals/ (default: UTC timestamp).",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional list of specific case IDs to include.",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="Optional baseline predictions JSON for hill-climb diff.",
    )
    parser.add_argument(
        "--backend",
        choices=("dump", "cmd", "openai", "anthropic", "gemini"),
        default="dump",
        help="Execution backend (default: dump prompt bundle without API calls).",
    )
    parser.add_argument(
        "--cmd",
        default="",
        help="Shell command invoked for each prompt when --backend cmd is used.",
    )
    parser.add_argument("--model", default="", help="Model name for API backends.")
    parser.add_argument(
        "--grader-model",
        default="",
        help="Optional grader model override (defaults to --model).",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO),
        help="Optional repository root override (primarily for tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Discover local eval suites, export or execute prompts, and score results.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code (``0`` on success, non-zero if scoring fails).

    Raises:
        ValueError: If no local eval suites match or ``--backend cmd`` lacks ``--cmd``.
    """
    args = _parse_args(argv)
    root = pathlib.Path(args.repo_root).resolve()
    tag = args.tag or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    rules_dir = root / ".agents" / "rules"
    skills_dir = root / ".agents" / "skills"
    suite_paths = [
        p
        for p in eval_agent_setup.discover_suites(
            skill=args.skill,
            rule=args.rule,
            rule_evals_dir=rules_dir / "evals",
            skills_dir=skills_dir,
        )
        if p.is_file()
    ]
    if not suite_paths:
        raise ValueError(
            f"No local eval suites matched skill={args.skill!r} rule={args.rule!r}"
        )
    catalog = eval_agent_setup.load_live_catalog(
        rules_dir=rules_dir,
        skills_dir=skills_dir,
    )
    catalog_md = eval_agent_setup.format_catalog_markdown(catalog)
    bundle = build_prompt_bundle(
        suite_paths,
        catalog,
        catalog_md,
        mode=args.mode,
        case_ids=set(args.case_ids or []),
    )
    eval_dir = root / "logs" / "agent_evals"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "dump":
        bundle_path = eval_dir / f"{tag}_prompts.json"
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        print(
            f"Wrote {len(bundle['trigger_prompts'])} trigger and "
            f"{len(bundle['behavior_prompts'])} behavior prompts -> {bundle_path}"
        )
        return 0

    model = args.model or _default_model_for_backend(args.backend)
    grader_model = args.grader_model or model
    call_candidate = _make_backend_caller(
        backend=args.backend,
        model=model,
        cmd=args.cmd,
        root=root,
    )
    call_grader = (
        call_candidate
        if grader_model == model
        else _make_backend_caller(
            backend=args.backend,
            model=grader_model,
            cmd=args.cmd,
            root=root,
        )
    )
    predictions = _execute_bundle(
        bundle,
        call_candidate=call_candidate,
        call_grader=call_grader,
        tag=tag,
        model=model,
    )
    pred_path = eval_dir / f"{tag}.json"
    report_path = eval_dir / f"{tag}_report.json"
    pred_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")

    score_args = ["--predictions", str(pred_path), "--out", str(report_path)]
    if args.skill:
        score_args.extend(["--skill", args.skill])
    if args.rule:
        score_args.extend(["--rule", args.rule])
    if args.baseline:
        score_args.extend(["--baseline", str(root / args.baseline)])
    score_rc = eval_agent_setup.main(["score", *score_args])
    return int(score_rc)


if __name__ == "__main__":
    raise SystemExit(main())
