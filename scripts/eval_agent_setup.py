#!/usr/bin/env python3
"""Validate, render, score, and diff StepCOVNet agent-setup evaluations.

Usage (repo root, project venv):
    python scripts/eval_agent_setup.py validate
    python scripts/eval_agent_setup.py render-catalog --format json
    python scripts/eval_agent_setup.py score --predictions logs/agent_evals/cand.json --baseline logs/agent_evals/base.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
RULES_DIR = REPO / ".agents" / "rules"
SKILLS_DIR = REPO / ".agents" / "skills"
DEFAULT_SUITE = REPO / "evals" / "agent_setup" / "suite.json"
SCHEMA_VERSION = 1

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract YAML frontmatter fields including folded/literal scalar blocks.

    Args:
        text: Raw markdown file contents.

    Returns:
        Mapping of frontmatter keys to stripped string values.
    """
    text = text.lstrip("\ufeff")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}

    lines = match.group(1).splitlines()
    out: dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if ":" not in line or line.startswith((" ", "\t")):
            idx += 1
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip()
        val = raw_val.strip()
        if val in {">", ">-", "|", "|-"}:
            parts: list[str] = []
            idx += 1
            while idx < len(lines) and (
                lines[idx].startswith((" ", "\t")) or not lines[idx].strip()
            ):
                stripped = lines[idx].strip()
                if stripped:
                    parts.append(stripped)
                idx += 1
            out[key] = " ".join(parts)
            continue
        out[key] = val.strip('"').strip("'")
        idx += 1
    return out


def load_live_catalog(
    rules_dir: pathlib.Path | None = None,
    skills_dir: pathlib.Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Load live ``model_decision`` rules and auto-discoverable skills from disk.

    Args:
        rules_dir: Override directory for ``.agents/rules``.
        skills_dir: Override directory for ``.agents/skills``.

    Returns:
        Dict with sorted ``rules`` and ``skills`` metadata entry lists.
    """
    r_dir = rules_dir or RULES_DIR
    s_dir = skills_dir or SKILLS_DIR

    rules: list[dict[str, str]] = []
    if r_dir.is_dir():
        for path in sorted(r_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
            if meta.get("trigger", "").lower() == "model_decision":
                rules.append(
                    {
                        "id": path.name,
                        "description": meta.get("description", ""),
                    }
                )

    skills: list[dict[str, str]] = []
    if s_dir.is_dir():
        for skill_md in sorted(s_dir.glob("*/SKILL.md")):
            meta = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            skills.append(
                {
                    "id": meta.get("name", "") or skill_md.parent.name,
                    "description": meta.get("description", ""),
                }
            )

    return {"rules": rules, "skills": skills}


def format_catalog_markdown(catalog: dict[str, list[dict[str, str]]]) -> str:
    """Format a live rule/skill catalog as compact markdown for prompt injection.

    Args:
        catalog: Catalog dict returned by ``load_live_catalog()``.

    Returns:
        Markdown text listing available rules and skills with descriptions.
    """
    lines = ["## Model-decision rules"]
    for rule in catalog.get("rules", []):
        lines.append(f"- `{rule['id']}`: {rule['description']}")
    lines.append("")
    lines.append("## Available skills")
    for skill in catalog.get("skills", []):
        lines.append(f"- `{skill['id']}`: {skill['description']}")
    lines.append("")
    return "\n".join(lines)


def load_json(path: pathlib.Path | str) -> dict[str, Any]:
    """Read a JSON object from disk.

    Args:
        path: Input JSON file path.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the root JSON value is not an object.
    """
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _validate_target_names(
    case_id: str,
    names: Any,
    known: set[str],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(names, list):
        return [f"{case_id}: {label} must be a list"]
    for item in names:
        if not isinstance(item, str) or item not in known:
            errors.append(f"{case_id}: unknown {label} entry {item!r}")
    return errors


def validate_suite(
    suite: dict[str, Any],
    catalog: dict[str, list[dict[str, str]]] | None = None,
) -> list[str]:
    """Validate an agent-setup evaluation suite against disk rules and skills.

    Args:
        suite: Parsed ``suite.json`` dictionary.
        catalog: Optional pre-loaded live catalog; loaded from disk when omitted.

    Returns:
        List of human-readable validation error strings (empty when valid).
    """
    cat = catalog or load_live_catalog()
    known_rules = {item["id"] for item in cat.get("rules", [])}
    known_skills = {item["id"] for item in cat.get("skills", [])}
    all_targets = known_rules | known_skills

    errors: list[str] = []
    if suite.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    trigger_cases = suite.get("trigger_cases")
    behavior_cases = suite.get("behavior_cases")
    if not isinstance(trigger_cases, list) or not trigger_cases:
        errors.append("trigger_cases must be a non-empty list")
        trigger_cases = []
    if not isinstance(behavior_cases, list) or not behavior_cases:
        errors.append("behavior_cases must be a non-empty list")
        behavior_cases = []

    seen_ids: set[str] = set()
    covered_rules: set[str] = set()
    covered_skills: set[str] = set()
    negative_controls = 0

    for idx, case in enumerate(trigger_cases):
        if not isinstance(case, dict):
            errors.append(f"trigger_cases[{idx}] must be an object")
            continue
        cid = str(case.get("id", "")).strip()
        if not cid:
            errors.append(f"trigger_cases[{idx}] missing non-empty id")
            continue
        if cid in seen_ids:
            errors.append(f"Duplicate case id {cid!r}")
        seen_ids.add(cid)

        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            errors.append(f"{cid}: prompt must be non-empty")

        exp_rules = case.get("expected_rules", [])
        opt_rules = case.get("optional_rules", [])
        forb_rules = case.get("forbidden_rules", [])
        exp_skills = case.get("expected_skills", [])
        opt_skills = case.get("optional_skills", [])
        forb_skills = case.get("forbidden_skills", [])

        errors.extend(
            _validate_target_names(cid, exp_rules, known_rules, "expected_rules")
        )
        errors.extend(
            _validate_target_names(cid, opt_rules, known_rules, "optional_rules")
        )
        errors.extend(
            _validate_target_names(cid, forb_rules, known_rules, "forbidden_rules")
        )
        errors.extend(
            _validate_target_names(cid, exp_skills, known_skills, "expected_skills")
        )
        errors.extend(
            _validate_target_names(cid, opt_skills, known_skills, "optional_skills")
        )
        errors.extend(
            _validate_target_names(cid, forb_skills, known_skills, "forbidden_skills")
        )

        if isinstance(exp_rules, list) and isinstance(forb_rules, list):
            overlap_r = set(exp_rules) & set(forb_rules)
            if overlap_r:
                errors.append(
                    f"{cid}: expected_rules and forbidden_rules overlap: {sorted(overlap_r)}"
                )
            covered_rules.update(r for r in exp_rules if isinstance(r, str))
        if isinstance(exp_skills, list) and isinstance(forb_skills, list):
            overlap_s = set(exp_skills) & set(forb_skills)
            if overlap_s:
                errors.append(
                    f"{cid}: expected_skills and forbidden_skills overlap: {sorted(overlap_s)}"
                )
            covered_skills.update(s for s in exp_skills if isinstance(s, str))

        if (
            isinstance(exp_rules, list)
            and isinstance(exp_skills, list)
            and not exp_rules
            and not exp_skills
        ):
            if (not isinstance(forb_rules, list) or not forb_rules) and (
                not isinstance(forb_skills, list) or not forb_skills
            ):
                errors.append(
                    f"{cid}: negative control trigger case must set forbidden_rules or forbidden_skills"
                )
            negative_controls += 1

    for missing_rule in sorted(known_rules - covered_rules):
        errors.append(f"Uncovered model_decision rule in trigger_cases: {missing_rule}")
    for missing_skill in sorted(known_skills - covered_skills):
        errors.append(f"Uncovered skill in trigger_cases: {missing_skill}")
    if negative_controls == 0:
        errors.append("trigger_cases must include at least one negative control case")

    for idx, case in enumerate(behavior_cases):
        if not isinstance(case, dict):
            errors.append(f"behavior_cases[{idx}] must be an object")
            continue
        cid = str(case.get("id", "")).strip()
        if not cid:
            errors.append(f"behavior_cases[{idx}] missing non-empty id")
            continue
        if cid in seen_ids:
            errors.append(f"Duplicate case id {cid!r}")
        seen_ids.add(cid)

        target = str(case.get("target", "")).strip()
        # Scoped glob rules may also be targets; check RULES_DIR too.
        glob_rules = (
            {p.name for p in RULES_DIR.glob("*.md")} if RULES_DIR.is_dir() else set()
        )
        if target not in (all_targets | glob_rules):
            errors.append(f"{cid}: unknown target {target!r}")
        if not str(case.get("prompt", "")).strip():
            errors.append(f"{cid}: prompt must be non-empty")

        must_any = case.get("must_contain_any", [])
        must_re = case.get("must_contain_regex", [])
        forb_re = case.get("must_not_contain_regex", [])
        rubric = case.get("rubric", [])
        if not must_any and not must_re and not forb_re and not rubric:
            errors.append(
                f"{cid}: must define at least one deterministic check or rubric"
            )

        for field_name, patterns in (
            ("must_contain_regex", must_re),
            ("must_not_contain_regex", forb_re),
        ):
            if not isinstance(patterns, list):
                errors.append(f"{cid}: {field_name} must be a list")
                continue
            for pat in patterns:
                try:
                    re.compile(str(pat))
                except re.error as exc:
                    errors.append(
                        f"{cid}: invalid regex {pat!r} in {field_name}: {exc}"
                    )

    return errors


def _score_trigger_case(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    exp_rules = set(case.get("expected_rules", []))
    opt_rules = set(case.get("optional_rules", []))
    forb_rules = set(case.get("forbidden_rules", []))
    exp_skills = set(case.get("expected_skills", []))
    opt_skills = set(case.get("optional_skills", []))
    forb_skills = set(case.get("forbidden_skills", []))

    pred_rules = set(pred.get("rules", []))
    pred_skills = set(pred.get("skills", []))

    tp_rules = sorted(exp_rules & pred_rules)
    fn_rules = sorted(exp_rules - pred_rules)
    fp_rules = sorted(pred_rules - (exp_rules | opt_rules))
    forb_hit_rules = sorted(forb_rules & pred_rules)

    tp_skills = sorted(exp_skills & pred_skills)
    fn_skills = sorted(exp_skills - pred_skills)
    fp_skills = sorted(pred_skills - (exp_skills | opt_skills))
    forb_hit_skills = sorted(forb_skills & pred_skills)

    passed = not fn_rules and not fp_rules and not fn_skills and not fp_skills
    is_negative = not exp_rules and not exp_skills
    return {
        "id": case["id"],
        "is_negative": is_negative,
        "passed": passed,
        "tp": len(tp_rules) + len(tp_skills),
        "fp": len(fp_rules) + len(fp_skills),
        "fn": len(fn_rules) + len(fn_skills),
        "tp_rules": tp_rules,
        "fn_rules": fn_rules,
        "fp_rules": fp_rules,
        "forbidden_hit_rules": forb_hit_rules,
        "tp_skills": tp_skills,
        "fn_skills": fn_skills,
        "fp_skills": fp_skills,
        "forbidden_hit_skills": forb_hit_skills,
    }


def _score_behavior_case(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    response = str(pred.get("response", ""))
    commands = [str(c) for c in pred.get("commands", [])]
    combined = response + "\n" + "\n".join(commands)
    combined_lower = combined.lower()

    failures: list[str] = []

    for group in case.get("must_contain_any", []):
        options = [str(opt) for opt in group]
        if not any(opt.lower() in combined_lower for opt in options):
            failures.append(f"missing any of {options}")

    for pat in case.get("must_contain_regex", []):
        if not re.search(str(pat), combined, re.IGNORECASE):
            failures.append(f"missing regex /{pat}/")

    for pat in case.get("must_not_contain_regex", []):
        if re.search(str(pat), combined, re.IGNORECASE):
            failures.append(f"matched forbidden regex /{pat}/")

    rubric = [str(item) for item in case.get("rubric", [])]
    raw_results = pred.get("rubric_results", [])
    by_criterion: dict[str, dict[str, Any]] = {}
    for entry in raw_results:
        if isinstance(entry, dict) and "criterion" in entry:
            key = str(entry["criterion"]).strip().lower()
            by_criterion[key] = entry

    rubric_passed = 0
    for idx, criterion in enumerate(rubric):
        match = by_criterion.get(criterion.strip().lower())
        if (
            match is None
            and idx < len(raw_results)
            and isinstance(raw_results[idx], dict)
        ):
            match = raw_results[idx]
        if match is None:
            failures.append(f"rubric missing result: {criterion}")
        elif not bool(match.get("passed", False)):
            reason = str(match.get("reason", "failed")).strip()
            failures.append(f"rubric failed ({criterion}): {reason}")
        else:
            rubric_passed += 1

    return {
        "id": case["id"],
        "target": case.get("target", ""),
        "passed": len(failures) == 0,
        "rubric_total": len(rubric),
        "rubric_passed": rubric_passed,
        "failures": failures,
    }


def score_predictions(
    suite: dict[str, Any], predictions: dict[str, Any]
) -> dict[str, Any]:
    """Score trigger and behavior predictions against a validated eval suite.

    Args:
        suite: Loaded ``suite.json`` dictionary.
        predictions: Predictions payload from ``run_workflow_eval.py`` or fixture.

    Returns:
        Structured report with summary metrics, per-target stats, and case rows.
    """
    trig_preds = predictions.get("trigger_predictions", {})
    beh_preds = predictions.get("behavior_predictions", {})

    trig_results: list[dict[str, Any]] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    neg_total = 0
    neg_fp_cases = 0
    per_target: dict[str, dict[str, int]] = {}

    def _bump(name: str, key: str) -> None:
        bucket = per_target.setdefault(name, {"tp": 0, "fn": 0, "fp": 0})
        bucket[key] += 1

    for case in suite.get("trigger_cases", []):
        cid = case["id"]
        if cid not in trig_preds:
            continue
        res = _score_trigger_case(case, trig_preds[cid])
        trig_results.append(res)
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]
        if res["is_negative"]:
            neg_total += 1
            if res["fp"] > 0:
                neg_fp_cases += 1
        for name in res["tp_rules"] + res["tp_skills"]:
            _bump(name, "tp")
        for name in res["fn_rules"] + res["fn_skills"]:
            _bump(name, "fn")
        for name in res["fp_rules"] + res["fp_skills"]:
            _bump(name, "fp")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    trig_passed = sum(1 for r in trig_results if r["passed"])
    trig_pass_rate = trig_passed / len(trig_results) if trig_results else 0.0
    neg_fp_rate = neg_fp_cases / neg_total if neg_total else 0.0

    beh_results: list[dict[str, Any]] = []
    for case in suite.get("behavior_cases", []):
        cid = case["id"]
        if cid not in beh_preds:
            continue
        beh_results.append(_score_behavior_case(case, beh_preds[cid]))

    beh_passed = sum(1 for r in beh_results if r["passed"])
    beh_pass_rate = beh_passed / len(beh_results) if beh_results else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "tag": str(predictions.get("tag", "")),
        "model": str(predictions.get("model", "")),
        "triggers": {
            "evaluated": len(trig_results),
            "passed": trig_passed,
            "pass_rate": round(trig_pass_rate, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "negative_total": neg_total,
            "negative_fp_cases": neg_fp_cases,
            "negative_fp_rate": round(neg_fp_rate, 4),
            "per_target": dict(sorted(per_target.items())),
            "cases": trig_results,
        },
        "behavior": {
            "evaluated": len(beh_results),
            "passed": beh_passed,
            "pass_rate": round(beh_pass_rate, 4),
            "cases": beh_results,
        },
    }


def diff_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare baseline and candidate eval reports for TDD hill-climbing.

    Args:
        baseline: Baseline report produced by ``score_predictions()``.
        candidate: Candidate report produced by ``score_predictions()``.

    Returns:
        Diff dictionary with ``fixed``, ``regressed``, ``still_failing``,
        ``new_cases``, and metric deltas in percentage points.
    """
    base_cases: dict[str, tuple[str, bool]] = {}
    cand_cases: dict[str, tuple[str, bool]] = {}

    for layer in ("triggers", "behavior"):
        for row in baseline[layer]["cases"]:
            base_cases[row["id"]] = (layer, bool(row["passed"]))
        for row in candidate[layer]["cases"]:
            cand_cases[row["id"]] = (layer, bool(row["passed"]))

    fixed: list[str] = []
    regressed: list[str] = []
    still_failing: list[str] = []
    still_passing: list[str] = []
    new_passed: list[str] = []
    new_failed: list[str] = []

    for cid, (layer, passed_now) in sorted(cand_cases.items()):
        label = f"{layer}:{cid}"
        if cid not in base_cases:
            (new_passed if passed_now else new_failed).append(label)
            continue
        passed_before = base_cases[cid][1]
        if not passed_before and passed_now:
            fixed.append(label)
        elif passed_before and not passed_now:
            regressed.append(label)
        elif not passed_before and not passed_now:
            still_failing.append(label)
        else:
            still_passing.append(label)

    return {
        "baseline_tag": baseline.get("tag", ""),
        "candidate_tag": candidate.get("tag", ""),
        "trigger_f1_delta_pp": round(
            (candidate["triggers"]["f1"] - baseline["triggers"]["f1"]) * 100.0, 2
        ),
        "trigger_pass_delta_pp": round(
            (candidate["triggers"]["pass_rate"] - baseline["triggers"]["pass_rate"])
            * 100.0,
            2,
        ),
        "behavior_pass_delta_pp": round(
            (candidate["behavior"]["pass_rate"] - baseline["behavior"]["pass_rate"])
            * 100.0,
            2,
        ),
        "fixed": fixed,
        "regressed": regressed,
        "still_failing": still_failing,
        "still_passing": still_passing,
        "new_passed": new_passed,
        "new_failed": new_failed,
    }


def format_report_markdown(
    report: dict[str, Any], diff: dict[str, Any] | None = None
) -> str:
    """Format an eval report (and optional hill-climb diff) as markdown.

    Args:
        report: Candidate report dictionary.
        diff: Optional output from ``diff_reports()``.

    Returns:
        Markdown summary report string.
    """
    trig = report["triggers"]
    beh = report["behavior"]
    tag = report["tag"] or "(untagged)"
    lines = [
        f"# Agent setup eval — {tag}",
        "",
        "| Layer | Evaluated | Passed | Pass rate | Key metric |",
        "| ----- | --------- | ------ | --------- | ---------- |",
        (
            f"| Triggers (Layer 2) | {trig['evaluated']} | {trig['passed']} | "
            f"{trig['pass_rate'] * 100:.1f}% | F1 {trig['f1'] * 100:.1f}% "
            f"(P {trig['precision'] * 100:.1f}% / R {trig['recall'] * 100:.1f}%, "
            f"Neg-FP {trig['negative_fp_cases']}/{trig['negative_total']}) |"
        ),
        (
            f"| Behavior (Layer 3) | {beh['evaluated']} | {beh['passed']} | "
            f"{beh['pass_rate'] * 100:.1f}% | — |"
        ),
        "",
    ]

    failed_trig = [c for c in trig["cases"] if not c["passed"]]
    if failed_trig:
        lines.append("## Failed trigger cases")
        for c in failed_trig:
            lines.append(
                f"- `{c['id']}`: FN_rules={c['fn_rules']} FP_rules={c['fp_rules']} "
                f"FN_skills={c['fn_skills']} FP_skills={c['fp_skills']}"
            )
        lines.append("")

    failed_beh = [c for c in beh["cases"] if not c["passed"]]
    if failed_beh:
        lines.append("## Failed behavior cases")
        for c in failed_beh:
            joined = "; ".join(c["failures"])
            lines.append(f"- `{c['id']}` (`{c['target']}`): {joined}")
        lines.append("")

    if diff is not None:
        lines.extend(
            [
                f"## Hill-climb diff ({diff['baseline_tag'] or 'baseline'} → {diff['candidate_tag'] or 'candidate'})",
                "",
                (
                    f"- Trigger F1 delta: {diff['trigger_f1_delta_pp']:+.2f} pp | "
                    f"Trigger pass delta: {diff['trigger_pass_delta_pp']:+.2f} pp | "
                    f"Behavior pass delta: {diff['behavior_pass_delta_pp']:+.2f} pp"
                ),
                f"- FIXED (FAIL → PASS): {diff['fixed'] or 'none'}",
                f"- REGRESSED (PASS → FAIL): {diff['regressed'] or 'none'}",
                f"- STILL FAILING: {diff['still_failing'] or 'none'}",
                f"- NEW PASS: {diff['new_passed'] or 'none'} | NEW FAIL: {diff['new_failed'] or 'none'}",
                "",
            ]
        )

    return "\n".join(lines)


def _cmd_render_catalog(args: argparse.Namespace) -> int:
    catalog = load_live_catalog()
    if args.format == "json":
        print(json.dumps(catalog, indent=2))
    else:
        print(format_catalog_markdown(catalog))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    suite_path = pathlib.Path(args.suite)
    suite = load_json(suite_path)
    errors = validate_suite(suite)
    n_trig = len(suite.get("trigger_cases", []))
    n_beh = len(suite.get("behavior_cases", []))
    if errors:
        print(f"INVALID {suite_path} ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"OK - {suite_path} valid ({n_trig} trigger cases, {n_beh} behavior cases).")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    suite = load_json(args.suite)
    errors = validate_suite(suite)
    if errors:
        print("Suite validation errors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    cand_preds = load_json(args.predictions)
    cand_report = score_predictions(suite, cand_preds)

    diff: dict[str, Any] | None = None
    if args.baseline:
        base_preds = load_json(args.baseline)
        base_report = score_predictions(suite, base_preds)
        diff = diff_reports(base_report, cand_report)

    print(format_report_markdown(cand_report, diff=diff))

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"report": cand_report}
        if diff is not None:
            payload["diff"] = diff
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if diff is not None and diff["regressed"]:
        return 1
    if args.strict:
        trig_ev = cand_report["triggers"]["evaluated"]
        beh_ev = cand_report["behavior"]["evaluated"]
        if (trig_ev and cand_report["triggers"]["passed"] < trig_ev) or (
            beh_ev and cand_report["behavior"]["passed"] < beh_ev
        ):
            return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, render, score, and diff StepCOVNet agent evals."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("render-catalog", help="Print live rules and skills.")
    p_cat.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )

    p_val = sub.add_parser("validate", help="Validate eval suite against disk.")
    p_val.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE),
        help="Path to suite.json.",
    )

    p_score = sub.add_parser("score", help="Score predictions and optional diff.")
    p_score.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE),
        help="Path to suite.json.",
    )
    p_score.add_argument(
        "--predictions",
        required=True,
        help="Path to candidate predictions JSON.",
    )
    p_score.add_argument(
        "--baseline",
        default="",
        help="Optional baseline predictions JSON for hill-climb diff.",
    )
    p_score.add_argument(
        "--out",
        default="",
        help="Optional output path for JSON report + diff.",
    )
    p_score.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 unless all evaluated cases pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for agent-setup evaluation utilities.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "render-catalog":
        return _cmd_render_catalog(args)
    if args.command == "validate":
        return _cmd_validate(args)
    return _cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
