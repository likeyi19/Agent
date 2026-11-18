"""Typed scientific refusals; unexpected programming errors stay distinct."""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ScientificInputRequirement:
    """An input owner's unresolved operand, before task/step addressing.

    Owners supply exact operand identities even when there are no offers. The
    coordinator must not recover these identities from labels or option order.
    """
    port: str
    argument: str
    kind: str = 'resource'
    member: str | None = None
    dependencies: tuple[str, ...] = ()
    applicability: Mapping = field(default_factory=dict)
    offers: tuple = ()
    messages: tuple[str, ...] = ()
    choices: tuple = ()


class ScientificInputRequired(ValueError):
    def __init__(self, *requirements, bindings=None):
        if not requirements or any(not isinstance(r, ScientificInputRequirement) for r in requirements):
            raise TypeError('Scientific prerequisite requires typed owner operands.')
        self.requirements = tuple(requirements)
        self.bindings = None if bindings is None else dict(bindings)
        super().__init__(' '.join(self.questions))

    @property
    def questions(self):
        return tuple(message for r in self.requirements for message in r.messages)

    @property
    def options(self):
        return tuple(option for r in self.requirements for option in r.offers)


class ScientificDiagnostic(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class OperationPreparationContext:
    """Selected consumer context; no execution or new lifecycle authority."""
    tool: object
    step_id: str
    registry: object
    selected: Mapping
    admitted: Mapping
    answers: Mapping
    receipts: list
    sessions: object
    interaction: object
    record: Mapping
    model_catalogs: tuple = ()
    observe_existing: bool = True

    def planned(self, value, *, seen=()):
        """Only the registered producer may promise properties of a future file."""
        from dataclasses import replace
        from agent.schemas import StepOutputRef
        if not isinstance(value, StepOutputRef) or value.step_id in seen or value.step_id not in self.selected:
            return {}
        name, arguments = self.selected[value.step_id]
        tool = self.registry.get(name)
        ports = [p for p in tool.semantic_planning.producer_ports if any(m.field_name==value.output_key for m in p.members)]
        if len(ports)!=1 or tool.planned_output_properties is None:
            return {}
        context = replace(self, tool=tool, step_id=value.step_id)
        return tool.planned_output_properties(arguments, ports[0].name, context, seen=(*seen,value.step_id))

    def planned_parameter_value(self, arguments, metadata):
        """An owner's exact input-port property, with observations disabled."""
        from dataclasses import replace
        from agent.schemas import StepOutputRef
        rule=metadata.get('value_from_planned_input')
        if not isinstance(rule,Mapping) or set(rule)!={'port','property'}:return None
        ports=[p for p in self.tool.semantic_planning.consumer_ports if p.name==rule['port']]
        if len(ports)!=1 or len(ports[0].members)!=1:return None
        source=arguments.get(ports[0].members[0].field_name)
        if not isinstance(source,StepOutputRef):return None
        return replace(self,observe_existing=False).planned(source).get(rule['property'])

    def accepted_input(self, port, arguments):
        """Exact accepted-result options through the existing captured context."""
        from agent.application.source_observations import accepted_input
        return accepted_input(self,port,arguments)

    def existing(self, value):
        if not self.observe_existing:return None
        from agent.application.source_observations import existing_object
        return existing_object(value, self.selected, self.registry,
            store=None if self.sessions is None else self.sessions._application.run_store)


def parameter_requirements(tool, names, *, offers=None):
    """Project owner-selected missing operands, preserving their typed domains."""
    offers = offers or {}
    rows = []
    for name in names:
        spec = tool.required_arguments.get(name, tool.optional_arguments.get(name))
        port = next(p for p in tool.semantic_planning.consumer_ports if any(m.field_name == name for m in p.members))
        physical = tool.source_input_types.get(port.name)
        rows.append(ScientificInputRequirement(port.name, name,
            'external_input' if physical else 'scalar', dependencies=('source:bytes',),
            applicability=dict(expected_types=[t.__name__ for t in spec.accepted_types],
                source_input_types=physical or ()),
            offers=tuple(offers.get(name, ())), choices=tuple(value for value in spec.choices if value is not None),
            messages=('Supply ' + name + ' for this selected operation.',)))
    return tuple(rows)
