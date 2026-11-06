"""Read-only metadata for accepted retired raw-tool records.

No callable, recovery hook, registration, provider offer or execution capability.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


@dataclass(frozen=True)
class HistoricalTool:
    name: str
    recovery_policy_version: str
    result_contract: object
    required_arguments: object
    optional_arguments: object
    semantic_planning: object
    native_functions: object = None
    scientific_delivery: object = None
    scientific_authority: object = None
    planning: object = None


@lru_cache(maxsize=7)
def tool(name):
    from .registry import (UnknownToolError, ArgumentSpec, ArgumentPlanningSemantics, ResultContract,
        ArtifactSemanticKind, PlanningSourceEligibility, PlanningProvenanceRole, SemanticProducerPortSpec,
        SemanticPortMember, _semantic_argument_port)
    rows = json.loads(Path(__file__).with_name('raw_history_contracts.json').read_text())
    if name not in rows: raise UnknownToolError('No reviewed historical raw-tool reader')
    row = rows[name]; types = dict(str=str, int=int, float=float, dict=dict, list=list, tuple=tuple, bool=bool, Path=Path, NoneType=type(None))
    def arguments(values):
        result = {}
        for key, value in values.items():
            planning = value['planning']
            if planning:
                planning['source_eligibility'] = PlanningSourceEligibility(planning['source_eligibility'])
                planning['accepted_artifact_kinds'] = tuple(ArtifactSemanticKind(k) for k in planning['accepted_artifact_kinds'])
                planning['provenance_role'] = None if planning['provenance_role'] is None else PlanningProvenanceRole(planning['provenance_role'])
                planning['conversational_members'] = tuple(tuple(m) for m in planning['conversational_members'])
                planning = ArgumentPlanningSemantics(**planning)
            result[key] = ArgumentSpec(tuple(types[t] for t in value['types']), choices=tuple(value['choices']), planning=planning)
        return MappingProxyType(result)
    ports = tuple(SemanticProducerPortSpec(**dict(p, members=tuple(SemanticPortMember(**m) for m in p['members']))) for p in row['ports'])
    required, optional = arguments(row['required_arguments']), arguments(row['optional_arguments'])
    scalars = tuple(_semantic_argument_port(key, required=key in required) for key, value in {**required, **optional}.items()
                    if value.planning is not None and value.planning.scientific_parameter)
    return HistoricalTool(name, row['policy'], ResultContract(row['result_name'], {k:tuple(types[t] for t in ts) for k,ts in row['fields'].items()}),
        required, optional, SimpleNamespace(producer_ports=ports, consumer_ports=scalars))


class ReplayRegistry:
    """Scoped to pure saved-declaration replay, never passed to execution."""
    def __init__(self, registry): self.registry = registry
    def get(self, name): return self.registry.accepted(name)
    def names(self):
        rows = json.loads(Path(__file__).with_name('raw_history_contracts.json').read_text())
        return (*self.registry.names(), *(name for name in rows if not self.registry.contains(name)))
    def contains(self, name): return name in self.names()
