"""Typed scientific refusals; unexpected programming errors stay distinct."""

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class ScientificContract:
    """One selected owner's facts; omitted properties are unknown, not false."""
    inputs: Mapping = field(default_factory=dict)
    parameters: Mapping = field(default_factory=dict)
    outputs: Mapping = field(default_factory=dict)
    effects: Mapping = field(default_factory=dict)
    selection: Mapping = field(default_factory=dict)
    resolve_output: Callable | None = field(default=None, repr=False, compare=False)
    resolve_selection: Callable | None = field(default=None, repr=False, compare=False)

    def projection(self):
        return {name: value for name in ('inputs', 'parameters', 'outputs', 'effects', 'selection')
                if (value := getattr(self, name))}


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

    def planned(self, value, *, seen=(), names=()):
        """Only the registered producer may promise properties of a future file."""
        from dataclasses import replace
        from agent.schemas import StepOutputRef
        if not isinstance(value, StepOutputRef) or value.step_id in seen or value.step_id not in self.selected:
            return {}
        name, arguments = self.selected[value.step_id]
        tool = self.registry.get(name)
        if tool.semantic_planning is None:
            return {}
        ports = [p for p in tool.semantic_planning.producer_ports if any(m.field_name==value.output_key for m in p.members)]
        if len(ports)!=1:
            return {}
        context = replace(self, tool=tool, step_id=value.step_id)
        port=ports[0]
        properties={}
        if port.identity_from_port is not None:
            source=next(p for p in tool.semantic_planning.consumer_ports if p.name==port.identity_from_port)
            properties.update(context.properties(arguments.get(source.members[0].field_name), names,
                                                 seen=(*seen,value.step_id)))
        contract = tool.selected_scientific_contract(arguments)
        if contract.resolve_output is not None:
            properties.update(contract.resolve_output(port.name, context, seen=(*seen,value.step_id), names=names))
        properties.update(contract.outputs.get(port.name, {}))
        return properties

    def coordinate_context(self, source, *, observe=True):
        """Use the task's exact source scope and the selected producer's declarations."""
        from agent.schemas import PriorOutputRef, StepOutputRef
        from agent.application.source_observations import identity_parent
        planned = {}
        seen = set()
        while isinstance(source, StepOutputRef) and source.step_id not in seen:
            parent = identity_parent(source, self.selected, self.registry, coordinate_lineage=True)
            if parent is None:
                break
            seen.add(source.step_id)
            _, arguments = self.selected[source.step_id]
            for key in ('species', 'assembly'):
                if arguments.get(key) is not None:
                    planned.setdefault(key, arguments[key])
            source = parent
        values = self.admitted.get('source_coordinates', {}) | planned
        if not observe or values.get('assembly') is not None:
            return values
        from pathlib import Path
        import h5py
        if (isinstance(source, PriorOutputRef) or
                isinstance(source, (str, Path)) and h5py.is_hdf5(source)):
            observed = self.existing(source)
            if observed is None or not h5py.is_hdf5(observed.path):
                return values
            from agent.tools.analysis.sequence_reference import assembly_observations
            claims = assembly_observations(observed.path, observed.sha256)['observations']
            labels = {label for row in claims for label in row['values']}
            if len(labels) == 1 and any(row['complete'] for row in claims):
                return dict(values, assembly=labels.pop())
        return values

    def coordinate_arguments(self, arguments, source, *, observe=True):
        keys = self.tool.application_inputs.intersection(('species', 'assembly')) - arguments.keys()
        if keys:
            values = self.coordinate_context(source, observe=observe)
            return {**arguments, **{key: values[key] for key in keys if values.get(key) is not None}}
        return arguments

    def properties(self, value, names, *, seen=()):
        """Future guarantees or checked existing storage, never inferred biology."""
        properties=self.planned(value, seen=seen, names=names)
        if not names:
            return properties
        observed=self.existing(value)
        if observed is not None:
            from agent.tools.analysis.peak_h5ad import scientific_properties
            from agent.tools._file_identity import checked_file
            checked_file(observed.path,observed.sha256)
            file_properties = tuple(name for name in names if name not in ('table_format','object_type'))
            if file_properties:
                properties.update(scientific_properties(observed.path, file_properties, graph_inputs=observed.graph_inputs))
            if 'table_format' in names:
                from agent.tools.analysis.native_tables import observe_table_format
                table_format=observed.table_format or observe_table_format(observed.path)
                if table_format is not None:properties['table_format']=table_format
            if 'object_type' in names and observed.object_type is not None:
                properties['object_type']=observed.object_type
            checked_file(observed.path,observed.sha256)
        return properties

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


def missing_contract_ports(tool, arguments):
    if tool.semantic_planning is None or tool.scientific_contract is None:
        return ()
    required=tool.selected_scientific_contract(arguments).parameters.get('required_ports', ())
    return tuple(port.members[0].field_name for port in tool.semantic_planning.consumer_ports
                 if port.name in required and arguments.get(port.members[0].field_name) is None)


def validate_input_properties(arguments, context):
    """Match only owner-declared properties; unknown future facts stay unknown."""
    tool=context.tool
    if tool.semantic_planning is None:
        # Explicit low-level compiler contracts can describe scalar/file tools
        # without Planner-visible semantic ports. They make no content promises.
        return
    contract = tool.selected_scientific_contract(arguments)
    if context.observe_existing:
        missing = missing_contract_ports(tool, arguments)
        if missing:
            raise ScientificInputRequired(*parameter_requirements(tool, missing), bindings=arguments)
    for port in tool.semantic_planning.consumer_ports:
        expected = contract.inputs.get(port.name, {})
        if not expected:continue
        source=next((arguments[m.field_name] for m in port.members
                     if m.name in ('dataset','path','manifest_path') and m.field_name in arguments),None)
        if source is None:continue
        properties=context.properties(source, expected)
        properties.update(dict.fromkeys(contract.parameters.get('invalidated_input_properties',{}).get(port.name,()),False))
        absent=sorted(name for name, value in expected.items() if name in properties and properties[name] != value
                      and not (value is True and properties[name] in ('numeric','boolean','dataframe'))
                      and not (name == 'format' and properties[name] == 'non_hdf5' and value not in ('h5ad','bam')))
        if absent:
            raise ScientificDiagnostic('SCIENTIFIC_INPUT_PROPERTY_REQUIRED',
                tool.name+'.'+port.name+' requires '+', '.join(absent)+
                '; the selected source does not provide it. Select a compatible scientific producer or input.')
        unknown=sorted(name for name in expected if '.' in name and name not in properties
                       and not name.startswith(('measurement.','count_values.')))
        if context.observe_existing and unknown:
            raise ScientificDiagnostic('SCIENTIFIC_INPUT_PROPERTY_UNVERIFIED',
                tool.name+'.'+port.name+' requires evidence for '+', '.join(unknown)+
                '; select a producer that guarantees it or an input on which it can be checked.')
