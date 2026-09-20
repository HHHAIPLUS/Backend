from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    report_path = Path(os.getenv("PHASE2_REPORT_PATH", "artifacts/phase2_oos_report.json"))
    roadmap_path = Path("REAL_MONEY_READINESS_ROADMAP.md")
    report = json.loads(report_path.read_text())

    if report.get("status") != "PROMOTED":
        raise SystemExit("Refusing to mark Phase 2 complete: validation report is not PROMOTED.")

    metrics = report.get("metrics", {})
    gate = metrics.get("absolute_gate", {})
    required = {"enough_samples": True, "accuracy_ok": True, "balanced_accuracy_ok": True, "positive_trade_expectancy": True, "positive_total_net_return": True, "drawdown_ok": True}
    if any(gate.get(k) is not v for k, v in required.items()):
        raise SystemExit("Refusing to mark Phase 2 complete: one or more absolute OOS gates are not PASS.")

    text = roadmap_path.read_text()
    start = text.index("# PHASE 2 —")
    end = text.index("# PHASE 3 —", start)
    section = text[start:end]
    section = section.replace("**Status: [ ] IN PROGRESS**", "**Status: [✓] COMPLETE**", 1)
    section = section.replace("- [ ] ", "- [✓] ")
    evidence = (
        "\n### Final Phase 2 evidence\n"
        + f"- Validation report status: **PROMOTED**.\n"
        + f"- Commit: `{os.getenv('GITHUB_SHA', 'unknown')}`.\n"
        + f"- Workflow run: `{os.getenv('GITHUB_RUN_ID', 'unknown')}`.\n"
        + f"- Symbol/interval/provider: `{report.get('symbol')}` / `{report.get('interval')}` / `{report.get('provider')}`.\n"
        + f"- Candles / dataset rows: `{report.get('candles')}` / `{report.get('rows')}`.\n"
        + f"- OOS samples/trades: `{metrics.get('candidate_oos', {}).get('samples')}` / `{metrics.get('candidate_oos', {}).get('trades')}`.\n"
        + f"- Accuracy / balanced accuracy: `{metrics.get('candidate_oos', {}).get('accuracy')}` / `{metrics.get('candidate_oos', {}).get('balanced_accuracy')}`.\n"
        + f"- Average trade net return / total net return: `{metrics.get('candidate_oos', {}).get('avg_trade_net_return')}` / `{metrics.get('candidate_oos', {}).get('total_net_return')}`.\n"
        + f"- Maximum drawdown: `{metrics.get('candidate_oos', {}).get('max_drawdown')}`.\n"
        + f"- Statistical evidence: `{metrics.get('promotion', {}).get('paired_bootstrap')}`.\n"
        + "- Real-money execution remains disabled; Phase 2 does not unlock live trading.\n"
    )
    marker = "### Phase 2 completion gate"
    pos = section.index(marker)
    section = section[:pos] + evidence + "\n" + section[pos:]
    text = text[:start] + section + text[end:]
    text = text.replace("| Phase 2 — Predictive Brain / OOS Validation | **[ ] IN PROGRESS** |", "| Phase 2 — Predictive Brain / OOS Validation | **[✓] COMPLETE** |", 1)
    text = text.replace("**Current phase: PHASE 2 — PREDICTIVE BRAIN / UNTOUCHED OOS VALIDATION.**", "**Current phase: PHASE 3 — ROBUSTNESS / WALK-FORWARD / STRESS VALIDATION.**", 1)
    roadmap_path.write_text(text)
    print("PHASE2_ROADMAP_MARKED_COMPLETE")


if __name__ == "__main__":
    main()