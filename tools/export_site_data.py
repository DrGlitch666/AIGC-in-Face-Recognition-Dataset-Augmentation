#!/usr/bin/env python3
"""Export frozen experiment results for an offline showcase website.

Run from the repository: python tools/export_site_data.py
Requires scipy. No model inference, training, or GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


PROTOCOL_KEYS = {"official": "lfw", "filtered": "lfw_filtered", "heldout": "lfw_heldout"}


def optional_object(value, label):
    """A missing or null optional block has no recorded fields, not zero values."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object or null, got {type(value).__name__}")
    return value


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return float(value)


def close(actual, expected, label, tolerance=0.000002):
    if abs(number(actual, label) - number(expected, label)) > tolerance:
        raise ValueError(f"{label}: inconsistent values {actual} vs {expected}")


def load_json(root, relative, sources):
    path = root / relative
    raw = path.read_bytes()
    result = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"{relative}: expected a JSON object")
    sources.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
    return result


def benchmark_view(metrics, protocol):
    """Export TAR on the full held-out set, accuracy on its balanced subset."""
    key = PROTOCOL_KEYS[protocol]
    benchmarks = optional_object(metrics.get("benchmarks"), f"{metrics['exp_id']}.benchmarks")
    block = benchmarks.get(key)
    if block is None:
        return None
    block = optional_object(block, f"{metrics['exp_id']}.benchmarks.{key}")
    tar_key = "tar_at_far_full" if protocol == "heldout" else "tar_at_far"
    if tar_key not in block:
        raise ValueError(f"{metrics['exp_id']}/{key}: missing {tar_key}; no subset TAR fallback")
    far_results = optional_object(block[tar_key], f"{metrics['exp_id']}/{key}/{tar_key}")
    far_block = far_results.get("1e-03")
    if far_block is None:
        raise ValueError(f"{metrics['exp_id']}/{key}: missing FAR=1e-03 result")
    far_block = optional_object(far_block, f"{metrics['exp_id']}/{key}/{tar_key}/1e-03")
    return {
        "accuracy": number(block["value"], key + "/value"),
        "tar": number(far_block["tar"], key + "/" + tar_key),
        "tar_threshold": far_block.get("threshold"),
        "actual_far": far_block.get("actual_far"),
        "accuracy_pairs_used": block.get("n_pairs_used"),
        "accuracy_pairs_missing": block.get("n_pairs_missing"),
        "tar_pairs_used": block.get("n_pairs_full") if protocol == "heldout" else block.get("n_pairs_used"),
        "protocol": block.get("protocol"),
        "accuracy_note": block.get("accuracy_note"),
        "accuracy_source_key": f"benchmarks.{key}.value",
        "tar_source_key": f"benchmarks.{key}.{tar_key}.1e-03.tar",
    }


def build(root):
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise ValueError("Missing scipy. Run: python -m pip install scipy") from exc

    sources = []
    generation = load_json(root, "results/runs/w3-generator-comparison/comparison.json", sources)
    comparison = load_json(root, "results/runs/w4-e4-comparison/comparison.json", sources)
    arms = {a["name"]: a for a in comparison["arms"]}
    if "E1" not in arms or "E4a" not in arms:
        raise ValueError("E4 comparison must contain E1 and E4a")
    close(comparison["far"], 0.001, "comparison.far", tolerance=1e-12)
    n_seeds = int(comparison["n_seeds"])
    if n_seeds < 2:
        raise ValueError("At least two seeds are required for uncertainty estimates")

    for arm in comparison["arms"]:
        if len(arm["seeds"]) != n_seeds:
            raise ValueError(f"{arm['name']}: seed count differs from n_seeds")
        for protocol in PROTOCOL_KEYS:
            summary = arm["protocols"][protocol]
            for metric in ("accuracy", "tar"):
                values = summary[f"per_seed_{metric}"]
                if len(values) != n_seeds:
                    raise ValueError(f"{arm['name']}/{protocol}/{metric}: incorrect seed count")
                values = [number(x, metric) for x in values]
                close(statistics.mean(values), summary[f"{metric}_mean"], f"{arm['name']}/{protocol}/{metric}_mean")
                close(statistics.stdev(values), summary[f"{metric}_std"], f"{arm['name']}/{protocol}/{metric}_std")
                for seed, value in zip(arm["seeds"], values):
                    close(seed[protocol][metric], value, f"{seed['exp_id']}/{protocol}/{metric}")

    derived_tests = {}
    for protocol in PROTOCOL_KEYS:
        derived_tests[protocol] = {}
        for metric in ("accuracy", "tar"):
            test = comparison["tests"][protocol][f"welch_{metric}"]
            t_value = number(test["t"], "Welch t")
            df = number(test["df"], "Welch df")
            se = number(test["se"], "Welch se")
            delta = number(test["delta"], "Welch delta")
            if df <= 0 or se <= 0:
                raise ValueError("Welch df and se must be positive")
            close(test["mean_a"], arms["E1"]["protocols"][protocol][f"{metric}_mean"], "Welch mean_a")
            close(test["mean_b"], arms["E4a"]["protocols"][protocol][f"{metric}_mean"], "Welch mean_b")
            close(delta, test["mean_b"] - test["mean_a"], "Welch delta")
            p_two_sided = float(2 * student_t.sf(abs(t_value), df))
            critical = float(student_t.ppf(0.975, df))
            derived_tests[protocol][metric] = {
                "delta": delta,
                "ci95_low": delta - critical * se,
                "ci95_high": delta + critical * se,
                "p_two_sided": p_two_sided,
                "significant_at_0_05": p_two_sided < 0.05,
                "critical_t_two_sided_0_05": critical,
                "t": t_value,
                "df": df,
                "method": "Welch; p=2*t.sf(abs(t),df); CI=delta +/- t.ppf(0.975,df)*se",
                "source": f"results/runs/w4-e4-comparison/comparison.json#/tests/{protocol}/welch_{metric}",
                "rounding_note": "Derived from the rounded frozen t, df, se and delta; p and CI are approximate.",
            }

    runs = {}
    warnings = []
    # Scan metrics if present; publish only experiment fields, no personal machine paths.
    for path in sorted((root / "results/runs").glob("*/metrics.json")):
        relative = path.relative_to(root).as_posix()
        metrics = load_json(root, relative, sources)
        exp_id = metrics.get("exp_id")
        if not isinstance(exp_id, str) or exp_id != path.parent.name:
            raise ValueError(f"{relative}: exp_id missing or different from directory name")
        train_set = optional_object(metrics.get("train_set"), f"{exp_id}.train_set")
        model = optional_object(metrics.get("model"), f"{exp_id}.model")
        protocol_detail = optional_object(metrics.get("protocol_detail"), f"{exp_id}.protocol_detail")
        protocols = {}
        for protocol in PROTOCOL_KEYS:
            view = benchmark_view(metrics, protocol)
            if view is not None:
                protocols[protocol] = view
        runs[exp_id] = {
            "source": relative,
            "train_set": {k: train_set.get(k) for k in (
                "real_identities", "real_images", "synth_images", "total_images", "synth_ratio")},
            "model": {k: model.get(k) for k in ("arch", "loss", "epochs", "params_total")},
            "protocols": protocols,
            "protocol_detail": {k: v for k, v in protocol_detail.items() if k in ("official", "filtered")},
        }
        detail = optional_object(protocol_detail.get("lfw_heldout"), f"{exp_id}.protocol_detail.lfw_heldout")
        runs[exp_id]["heldout_manifest"] = {k: detail.get(k) for k in ("n_pairs", "n_same", "n_diff", "n_images_needed", "n_aligned")}

    for arm in comparison["arms"]:
        for seed in arm["seeds"]:
            exp_id = seed["exp_id"]
            if exp_id not in runs:
                warnings.append(f"{exp_id}/metrics.json absent: using frozen comparison summary only")
                continue
            for protocol in PROTOCOL_KEYS:
                view = runs[exp_id]["protocols"].get(protocol)
                if view is None:
                    raise ValueError(f"{exp_id}: missing benchmark for {protocol}")
                for metric in ("accuracy", "tar"):
                    close(view[metric], seed[protocol][metric], f"{exp_id}/{protocol}/{metric}: metrics vs comparison")

    baseline_id = arms["E1"]["seeds"][0]["exp_id"]
    if baseline_id not in runs:
        raise ValueError(f"Required representative metrics missing: {baseline_id}/metrics.json")
    # The frozen comparison stores these counts in text; parse it rather than hand-copy numbers.
    match = re.fullmatch(r"real\s+(\d+)\s*\+\s*synth\s+(\d+)", arms["E4a"]["train_set"])
    if match is None:
        raise ValueError("E4a.train_set: expected 'real <count> + synth <count>'")
    real, synth = map(int, match.groups())
    close(real, runs[baseline_id]["train_set"]["real_images"], "E1 real image count", tolerance=0)
    data = {
        "schema_version": 1,
        "generation": generation,
        "recognition": comparison,
        "derived_tests": derived_tests,
        "training_counts": {"real": real, "synth": synth, "total": real + synth,
                            "synth_fraction": synth / (real + synth),
                            "source": "recognition.arms[E4a].train_set (parsed)"},
        "runs": runs,
        "sources": sources,
        "warnings": warnings,
        "display_notes": {
            "official": "Official pair manifest contains training identities. Accuracy here is accuracy@best, not a literature-equivalent cross-validated LFW score.",
            "heldout": "Custom held-out protocol; identities disjoint from training. Not the official LFW number. Accuracy uses a balanced subset; TAR uses full pairs.",
            "uncertainty": "Across-seed standard deviation and confidence interval are different. Do not label SD as CI.",
            "statistics": "Use each Welch test's df; critical_values_n3 is not a universal critical value. Derived p and CI do not overwrite frozen source results.",
            "generation": "The recipe comparison's small sample and the screened training corpus have different populations; their id_sim means must not be conflated.",
        },
    }
    # Validate serialization before writing either output.
    json.dumps(data, ensure_ascii=False, allow_nan=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: <root>/site/data")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "site/data"
    try:
        data = build(root)
        formatted = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        js_payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        output.mkdir(parents=True, exist_ok=True)
        (output / "site_data.json").write_text(formatted, encoding="utf-8")
        (output / "site_data.js").write_text("window.SITE_DATA = " + js_payload + ";\n", encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print("Export OK")
    print(f"JSON: {output / 'site_data.json'}")
    print(f"JS:   {output / 'site_data.js'}")
    print(f"Checked source files: {len(data['sources'])}")
    print("Held-out TAR: benchmarks.lfw_heldout.tar_at_far_full['1e-03'].tar")
    for protocol, tests in data["derived_tests"].items():
        for metric, test in tests.items():
            verdict = "significant" if test["significant_at_0_05"] else "not significant"
            print(f"{protocol}/{metric}: p={test['p_two_sided']:.6f} ({verdict}; two-sided Welch)")
    for warning in data["warnings"]:
        print("WARNING: " + warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
