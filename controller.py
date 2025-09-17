from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import time, json

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIM_LOG = LOG_DIR / "sim_actions.jsonl"

DEP_MIN, DEP_MAX = 0, 4
PTSD_MIN, PTSD_MAX = 0, 2

def compute_risk_score(dep_class: int, ptsd_class: int) -> float:
    dep_score = dep_class / 4.0
    ptsd_score = ptsd_class / 2.0
    return round((0.6 * dep_score + 0.4 * ptsd_score) * 100.0, 2)

@dataclass
class Case:
    dataset: str                 # "EDAIC" | "DAIC-WOZ"
    participant_id: int
    pred_dep: int
    pred_ptsd: int
    actual_dep: Optional[int] = None
    actual_ptsd: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def risk(self) -> float:
        return compute_risk_score(int(self.pred_dep), int(self.pred_ptsd))

@dataclass
class Action:
    decision: str   # "confirm" | "override_up" | "override_down" | "defer"
    notes: str = ""

@dataclass
class Outcome:
    accepted_dep: int
    accepted_ptsd: int
    was_override: bool
    new_risk: float

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def apply_action(case: Case, action: Action) -> Outcome:
    d, p = int(case.pred_dep), int(case.pred_ptsd)
    was_override = False
    if action.decision == "override_up":
        d, p = _clamp(d+1, DEP_MIN, DEP_MAX), _clamp(p+1, PTSD_MIN, PTSD_MAX); was_override = True
    elif action.decision == "override_down":
        d, p = _clamp(d-1, DEP_MIN, DEP_MAX), _clamp(p-1, PTSD_MIN, PTSD_MAX); was_override = True
    elif action.decision in ("confirm", "defer"):
        pass
    else:
        raise ValueError(f"Unknown decision: {action.decision}")
    return Outcome(d, p, was_override, compute_risk_score(d, p))

def log_event(event: Dict[str, Any], path: Path = SIM_LOG):
    event = {**event, "ts": time.time()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
