"""Small, immutable qualification scenarios; no provider-specific semantics.

The paths name synthetic inputs for structural PLAN_ONLY admission. They are
not biological datasets and do not establish scientific input readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from agent.schemas import AgentRequest, RunMode
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from agent.application.session_state import digest


SURFACES = frozenset(("interpreter", "stage_a", "stage_b", "answer", "guidance", "layer2"))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    surface: str
    utterance: str
    fixture: str = "empty"
    request: AgentRequest | None = None
    expected: Mapping = field(default_factory=dict)
    fixed_capability_ids: tuple[str, ...] = ()
    preflight: bool = False
    human_review_required: bool = False

    def __post_init__(self):
        if not self.scenario_id or self.surface not in SURFACES or not self.utterance.strip():
            raise ValueError("Invalid qualification scenario identity or surface.")
        if not self.fixture or (self.request is not None and not isinstance(self.request, AgentRequest)):
            raise ValueError("Invalid qualification request or fixture.")
        if type(self.preflight) is not bool or type(self.human_review_required) is not bool:
            raise ValueError("Qualification flags must be booleans.")
        if (type(self.fixed_capability_ids) is not tuple
                or any(type(v) is not str or not v for v in self.fixed_capability_ids)
                or len(set(self.fixed_capability_ids)) != len(self.fixed_capability_ids)):
            raise ValueError("Fixed capabilities must be a unique tuple.")
        object.__setattr__(self, "expected", freeze_json_mapping(self.expected, "scenario.expected"))

    def to_dict(self):
        return dict(scenario_id=self.scenario_id, surface=self.surface, utterance=self.utterance,
            fixture=self.fixture, request=None if self.request is None else self.request.to_dict(),
            expected=_serialize(self.expected), fixed_capability_ids=list(self.fixed_capability_ids),
            preflight=self.preflight, human_review_required=self.human_review_required)

    def fingerprint(self):
        return digest(self.to_dict())


def canonical_scenarios(input_path="/synthetic/qualification-pbmc.h5ad") -> tuple[Scenario, ...]:
    """Fourteen ready-to-replay cases, not a request to execute or live-run them."""
    inspect = "Inspect the selected scATAC-seq dataset."
    def request(name, text, inputs):
        return AgentRequest("qualification-" + name, text, inputs, RunMode.PLAN_ONLY)
    inspection = request("inspection", inspect, {"input_path": input_path})
    chain_text = "Compute mouse EpiZoo embeddings on CPU and build their cell-neighbor graph."
    chain = request("producer-consumer", chain_text, {
        "input_path": input_path, "output_dir": "/synthetic/qualification-output",
        "species": "mouse", "device": "cpu"})
    unrelated_text = "Inspect only the selected input dataset; the supplied reference and query inputs are unrelated to this request."
    unrelated = request("unrelated-inputs", unrelated_text, {
        "input_path": input_path, "reference_input_path": "/synthetic/reference.h5ad",
        "query_input_path": "/synthetic/query.h5ad", "species": "mouse"})
    unsupported_text = "Predict protein folding structures from the supplied scATAC-seq dataset."
    unsupported = request("unsupported", unsupported_text, {"input_path": input_path})
    return (
        Scenario("I01", "layer2", inspect, request=inspection,
                 expected={"kind": "plan", "tool": "inspect_scATAC", "tools": ("inspect_scATAC",)}, preflight=True),
        Scenario("I02", "stage_a", inspect, request=inspection,
                 expected={"kind": "select", "capabilities": ("processed_inspection",)}),
        Scenario("I03", "stage_b", inspect, request=inspection,
                 expected={"kind": "plan", "tool": "inspect_scATAC", "tools": ("inspect_scATAC",),
                           "request_sources": {"inspect_scATAC.path": "input_path"}},
                 fixed_capability_ids=("processed_inspection",), preflight=True),
        Scenario("I04", "stage_b", chain_text, request=chain,
                 expected={"kind": "plan", "tool": "build_cell_neighbors", "tools": ("epizoo_embed_cells", "build_cell_neighbors")},
                 fixed_capability_ids=("embedding_analysis",), preflight=True),
        Scenario("I05", "stage_b", unrelated_text, request=unrelated,
                 expected={"kind": "plan", "tool": "inspect_scATAC", "tools": ("inspect_scATAC",),
                           "request_sources": {"inspect_scATAC.path": "input_path"}},
                 fixed_capability_ids=("processed_inspection", "reference_annotation"), preflight=True),
        Scenario("I06", "stage_a", unsupported_text, request=unsupported,
                 expected={"kind": "unsupported"}),
        Scenario("I07", "interpreter", "Make the TSS threshold stricter.", fixture="selection",
                 expected={"kind": "clarify", "reason": "missing_parameter_value"}),
        Scenario("I08", "answer", "What does the non-zero entry count tell us about this matrix?", fixture="matrix",
                 expected={"kind": "answer", "referent": "@current_result", "output_name": "matrix"},
                 human_review_required=True),
        Scenario("I09", "answer", "What does the accepted annotation show for cluster 3?", fixture="annotation",
                 expected={"kind": "answer", "referent": "@current_result", "subject": "3", "output_name": "annotation"},
                 human_review_required=True),
        Scenario("I10", "interpreter", "Explain the current result.", fixture="rollback",
                 expected={"kind": "answer", "referent": "@current_result", "target_turn": "older", "output_name": "inspection"}),
        Scenario("I11", "interpreter", "Explain the most recently created result.", fixture="rollback",
                 expected={"kind": "answer", "referent": "@most_recently_created", "target_turn": "newer", "output_name": "inspection"}),
        Scenario("I12", "interpreter", "Explain the result discussed in the immediately previous turn.", fixture="previous_answer",
                 expected={"kind": "answer", "referent": "@previous_turn_result", "subject": "3", "target_turn": "accepted", "output_name": "annotation"}),
        Scenario("I13", "guidance", "What analyses could help investigate the accepted annotation for cluster 3?", fixture="annotation",
                 expected={"kind": "answer", "referent": "@current_result", "subject": "3", "output_name": "annotation"},
                 human_review_required=True),
        Scenario("I14", "interpreter", "Explain this result.", fixture="ambiguous",
                 expected={"kind": "clarify", "reason": "ambiguous_subject"}),
    )


def scenario_by_id(scenario_id: str) -> Scenario:
    return next(s for s in canonical_scenarios() if s.scenario_id == scenario_id)
