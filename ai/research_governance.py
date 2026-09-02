from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class PromotionApproval:
    candidate_id: str
    approved_by: str
    rollback_target: str
    approved_at: str
    production_mutation_performed: bool = False


class ResearchGovernance:
    """Final human approval/rollback contract; never mutates production itself."""

    def __init__(self):
        self.approvals: dict[str, PromotionApproval] = {}

    def approve(self, candidate: Any, approved_by: str, rollback_target: str) -> PromotionApproval:
        if candidate.status != "promotion_eligible":
            raise ValueError("candidate must pass research gates before approval")
        if not approved_by.strip() or not rollback_target.strip():
            raise ValueError("approver and known-good rollback target are required")
        approval = PromotionApproval(str(candidate.candidate_id), approved_by.strip(), rollback_target.strip(), datetime.now(timezone.utc).isoformat())
        self.approvals[approval.candidate_id] = approval
        candidate.evidence = {**candidate.evidence, "explicit_approval": asdict(approval)}
        return approval

    def rollback_target(self, candidate_id: str) -> str | None:
        approval = self.approvals.get(candidate_id)
        return approval.rollback_target if approval else None
