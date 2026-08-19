"""Finalize A4.2 artifact labels without changing registered scientific results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return cast(dict[str, Any], payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir

    results_path = output_dir / "results.json"
    results = load(results_path)
    old_key = "total_estimated_provider_cost_usd_all_a42_calls"
    new_key = "total_estimated_provider_cost_usd_recorded_scored_and_timed_calls"
    if old_key not in results:
        raise RuntimeError("Expected pre-finalization A4.2 cost subtotal is missing.")
    results[new_key] = results.pop(old_key)
    limitation = (
        "The aggregate provider-cost subtotal covers quality, adversarial, repeatability, "
        "and timed-latency records. Compatibility-probe and latency-warmup costs are not "
        "captured in that subtotal; the registered adoption cost ceiling is evaluated "
        "separately from the maximum quality-pass request cost."
    )
    limitations = cast(list[str], results["limitations"])
    limitations.append(limitation)
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit_path = output_dir / "post_audit.json"
    audit = load(audit_path)
    audit_limitations = cast(list[str], audit["limitations"])
    audit_limitations.append(limitation)
    audit["cost_reporting_scope_corrected_before_execution"] = True
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\nCost note: the reported aggregate provider-cost subtotal excludes the "
            "compatibility probe and latency warmups. The registered adoption ceiling uses "
            "the maximum estimated cost of the 240 quality-pass requests.\n"
        )
    with (output_dir / "post_audit.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\nCost-reporting scope was corrected before execution: aggregate diagnostic "
            "cost does not claim to include compatibility-probe or latency-warmup calls.\n"
        )


if __name__ == "__main__":
    main()
