#!/usr/bin/env python3
"""Validate, discover, render, score, and diff skill-local/rule-local agent evals.

Usage (repo root, project venv):
    python scripts/eval_agent_setup.py validate
    python scripts/eval_agent_setup.py validate --skill onset-evaluation
    python scripts/eval_agent_setup.py list-suites --format json
    python scripts/eval_agent_setup.py render-catalog --format json
    python scripts/eval_agent_setup.py score --skill onset-evaluation --predictions cand.json --baseline base.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
RULES_DIR = REPO / ".agents" / "rules"
RULE_EVALS_DIR = RULES_DIR / "evals"
SKILLS_DIR = REPO / ".agents" / "skills"
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


def discover_suites(
    skill: str = "",
    rule: str = "",
    suite_path: str = "",
    *,
    skills_dir: pathlib.Path | None = None,
    rule_evals_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Resolve local skill and rule eval suite JSON files.

    Args:
        skill: Optional single skill directory name.
        rule: Optional single rule filename or stem (e.g. ``git-and-ci.md``).
        suite_path: Optional explicit suite JSON file path.
        skills_dir: Override ``.agents/skills`` directory.
        rule_evals_dir: Override ``.agents/rules/evals`` directory.

    Returns:
        Sorted list of resolved suite paths.
    """
    if suite_path:
        return [pathlib.Path(suite_path)]
    s_dir = skills_dir or SKILLS_DIR
    re_dir = rule_evals_dir or RULE_EVALS_DIR
    if skill:
        return [s_dir / skill / "evals" / "suite.json"]
    if rule:
        stem = rule.removesuffix(".md")
        return [re_dir / f"{stem}.json"]

    paths: list[pathlib.Path] = []
    if s_dir.is_dir():
        paths.extend(sorted(s_dir.glob("*/evals/suite.json")))
    if re_dir.is_dir():
        paths.extend(sorted(re_dir.glob("*.json")))
    return paths


def load_merged_suite(paths: list[pathlib.Path]) -> dict[str, Any]:
    """Merge local skill and rule eval JSON files into one combined suite dict.

    Args:
        paths: Ordered list of local suite JSON paths to read and combine.

    Returns:
        Merged suite dict with ``trigger_cases`` and ``behavior_cases`` lists.
    """
    merged_triggers: list[dict[str, Any]] = []
    merged_behaviors: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        for case in data.get("trigger_cases", []):
            merged_triggers.append(case)
        for case in data.get("behavior_cases", []):
            merged_behaviors.append(case)
    return {
        "schema_version": SCHEMA_VERSION,
        "suites": [str(p) for p in paths],
        "trigger_cases": merged_triggers,
        "behavior_cases": merged_behaviors,
    }


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


def validate_local_suites(
    paths: list[pathlib.Path],
    catalog: dict[str, list[dict[str, str]]] | None = None,
    *,
    rules_dir: pathlib.Path | None = None,
    skills_dir: pathlib.Path | None = None,
    rule_evals_dir: pathlib.Path | None = None,
    check_all_coverage: bool = True,
) -> list[str]:
    """Validate skill-local and rule-local eval suites against disk definitions.

    Args:
        paths: Local suite JSON paths to validate.
        catalog: Optional pre-loaded live catalog; loaded from disk when omitted.
        rules_dir: Override ``.agents/rules`` directory.
        skills_dir: Override ``.agents/skills`` directory.
        rule_evals_dir: Override ``.agents/rules/evals`` directory.
        check_all_coverage: Whether to require every skill and ``model_decision``
            rule on disk to have a local eval suite file.

    Returns:
        List of human-readable validation errors (empty when valid).
    """
    r_dir = rules_dir or RULES_DIR
    s_dir = skills_dir or SKILLS_DIR
    re_dir = rule_evals_dir or (r_dir / "evals")
    cat = catalog or load_live_catalog(rules_dir=r_dir, skills_dir=s_dir)

    known_model_rules = {item["id"] for item in cat.get("rules", [])}
    known_skills = {item["id"] for item in cat.get("skills", [])}
    all_rule_files = (
        {p.name for p in r_dir.glob("*.md") if p.name.lower() != "readme.md"}
        if r_dir.is_dir()
        else set()
    )
    all_targets = known_skills | all_rule_files

    errors: list[str] = []
    if not paths:
        return ["No local eval suite files found"]

    seen_ids: set[str] = set()

    if check_all_coverage:
        for skill_id in sorted(known_skills):
            expected_path = s_dir / skill_id / "evals" / "suite.json"
            if not expected_path.is_file():
                errors.append(f"Missing local skill eval suite: {expected_path}")
        for rule_id in sorted(known_model_rules):
            stem = rule_id.removesuffix(".md")
            expected_path = re_dir / f"{stem}.json"
            if not expected_path.is_file():
                errors.append(
                    f"Missing local model_decision rule eval suite: {expected_path}"
                )

    for path in paths:
        if not path.is_file():
            errors.append(f"Missing eval suite file: {path}")
            continue
        suite = load_json(path)
        label = path.as_posix()

        if suite.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {SCHEMA_VERSION}")

        owner = str(suite.get("owner", "")).strip()
        owner_type = str(suite.get("owner_type", "")).strip()
        if owner_type not in {"skill", "rule"}:
            errors.append(f"{label}: owner_type must be 'skill' or 'rule'")
        elif owner_type == "skill":
            if owner not in known_skills:
                errors.append(f"{label}: unknown owner skill {owner!r}")
            elif path.parent.parent.name != owner:
                errors.append(
                    f"{label}: owner {owner!r} does not match skill directory {path.parent.parent.name!r}"
                )
        elif owner_type == "rule":
            if owner not in all_rule_files:
                errors.append(f"{label}: unknown owner rule {owner!r}")
            elif f"{path.stem}.md" != owner:
                errors.append(
                    f"{label}: owner {owner!r} does not match filename {path.name!r}"
                )

        trigger_cases = suite.get("trigger_cases", [])
        behavior_cases = suite.get("behavior_cases", [])
        if not isinstance(trigger_cases, list):
            errors.append(f"{label}: trigger_cases must be a list")
            trigger_cases = []
        if not isinstance(behavior_cases, list):
            errors.append(f"{label}: behavior_cases must be a list")
            behavior_cases = []
        if not trigger_cases and not behavior_cases:
            errors.append(
                f"{label}: suite must define at least one trigger or behavior case"
            )

        pos_for_owner = 0
        neg_for_owner = 0

        for idx, case in enumerate(trigger_cases):
            if not isinstance(case, dict):
                errors.append(f"{label}: trigger_cases[{idx}] must be an object")
                continue
            cid = str(case.get("id", "")).strip()
            if not cid:
                errors.append(f"{label}: trigger_cases[{idx}] missing non-empty id")
                continue
            if cid in seen_ids:
                errors.append(f"Duplicate case id {cid!r}")
            seen_ids.add(cid)

            if not str(case.get("prompt", "")).strip():
                errors.append(f"{cid}: prompt must be non-empty")

            exp_rules = case.get("expected_rules", [])
            opt_rules = case.get("optional_rules", [])
            forb_rules = case.get("forbidden_rules", [])
            exp_skills = case.get("expected_skills", [])
            opt_skills = case.get("optional_skills", [])
            forb_skills = case.get("forbidden_skills", [])

            errors.extend(
                _validate_target_names(
                    cid, exp_rules, known_model_rules, "expected_rules"
                )
            )
            errors.extend(
                _validate_target_names(
                    cid, opt_rules, known_model_rules, "optional_rules"
                )
            )
            errors.extend(
                _validate_target_names(
                    cid, forb_rules, known_model_rules, "forbidden_rules"
                )
            )
            errors.extend(
                _validate_target_names(cid, exp_skills, known_skills, "expected_skills")
            )
            errors.extend(
                _validate_target_names(cid, opt_skills, known_skills, "optional_skills")
            )
            errors.extend(
                _validate_target_names(
                    cid, forb_skills, known_skills, "forbidden_skills"
                )
            )

            if isinstance(exp_rules, list) and isinstance(forb_rules, list):
                overlap_r = set(exp_rules) & set(forb_rules)
                if overlap_r:
                    errors.append(
                        f"{cid}: expected_rules and forbidden_rules overlap: {sorted(overlap_r)}"
                    )
            if isinstance(exp_skills, list) and isinstance(forb_skills, list):
                overlap_s = set(exp_skills) & set(forb_skills)
                if overlap_s:
                    errors.append(
                        f"{cid}: expected_skills and forbidden_skills overlap: {sorted(overlap_s)}"
                    )

            if (
                owner_type == "skill"
                and isinstance(exp_skills, list)
                and owner in exp_skills
            ):
                pos_for_owner += 1
            if (
                owner_type == "skill"
                and isinstance(forb_skills, list)
                and owner in forb_skills
            ):
                neg_for_owner += 1
            if (
                owner_type == "rule"
                and owner in known_model_rules
                and isinstance(exp_rules, list)
                and owner in exp_rules
            ):
                pos_for_owner += 1

        if owner_type == "skill":
            if pos_for_owner == 0:
                errors.append(
                    f"{label}: skill suite must include >=1 trigger case expecting {owner!r}"
                )
            if neg_for_owner == 0:
                errors.append(
                    f"{label}: skill suite must include >=1 negative trigger case forbidding {owner!r}"
                )
        elif owner_type == "rule" and owner in known_model_rules and pos_for_owner == 0:
            errors.append(
                f"{label}: model_decision rule suite must include >=1 trigger case expecting {owner!r}"
            )

        for idx, case in enumerate(behavior_cases):
            if not isinstance(case, dict):
                errors.append(f"{label}: behavior_cases[{idx}] must be an object")
                continue
            cid = str(case.get("id", "")).strip()
            if not cid:
                errors.append(f"{label}: behavior_cases[{idx}] missing non-empty id")
                continue
            if cid in seen_ids:
                errors.append(f"Duplicate case id {cid!r}")
            seen_ids.add(cid)

            target = str(case.get("target", "")).strip()
            if target not in all_targets:
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
    """Score trigger and behavior predictions against a merged or local suite.

    Args:
        suite: Merged suite dictionary from ``load_merged_suite()``.
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
        ``new_passed``, ``new_failed``, and metric deltas in percentage points.
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


def _cmd_list_suites(args: argparse.Namespace) -> int:
    paths = discover_suites(skill=args.skill, rule=args.rule)
    if args.format == "json":
        print(json.dumps([str(p) for p in paths], indent=2))
    else:
        for p in paths:
            print(p)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    paths = discover_suites(skill=args.skill, rule=args.rule, suite_path=args.suite)
    check_all = not (args.skill or args.rule or args.suite)
    errors = validate_local_suites(paths, check_all_coverage=check_all)
    if errors:
        print(f"INVALID local eval suites ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        return 1
    merged = load_merged_suite(paths)
    n_trig = len(merged["trigger_cases"])
    n_beh = len(merged["behavior_cases"])
    print(
        f"OK - {len(paths)} local suite(s) valid "
        f"({n_trig} trigger cases, {n_beh} behavior cases)."
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    paths = discover_suites(skill=args.skill, rule=args.rule, suite_path=args.suite)
    check_all = not (args.skill or args.rule or args.suite)
    errors = validate_local_suites(paths, check_all_coverage=check_all)
    if errors:
        print("Suite validation errors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    merged_suite = load_merged_suite(paths)
    cand_preds = load_json(args.predictions)
    cand_report = score_predictions(merged_suite, cand_preds)

    diff: dict[str, Any] | None = None
    if args.baseline:
        base_preds = load_json(args.baseline)
        base_report = score_predictions(merged_suite, base_preds)
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
        description="Validate, render, score, and diff local agent eval suites."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("render-catalog", help="Print live rules and skills.")
    p_cat.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )

    p_list = sub.add_parser(
        "list-suites", help="List discovered skill-local and rule-local eval suites."
    )
    p_list.add_argument("--skill", default="", help="Filter to one skill name.")
    p_list.add_argument("--rule", default="", help="Filter to one rule name/stem.")
    p_list.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )

    p_val = sub.add_parser("validate", help="Validate local eval suites against disk.")
    p_val.add_argument("--skill", default="", help="Validate only one skill suite.")
    p_val.add_argument("--rule", default="", help="Validate only one rule suite.")
    p_val.add_argument("--suite", default="", help="Validate an explicit suite path.")

    p_score = sub.add_parser("score", help="Score predictions and optional diff.")
    p_score.add_argument("--skill", default="", help="Score only one skill suite.")
    p_score.add_argument("--rule", default="", help="Score only one rule suite.")
    p_score.add_argument("--suite", default="", help="Score an explicit suite path.")
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
    if args.command == "list-suites":
        return _cmd_list_suites(args)
    if args.command == "validate":
        return _cmd_validate(args)
    return _cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
