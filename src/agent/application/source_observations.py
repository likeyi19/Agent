"""Exact existing source/artifact views; future outputs remain symbolic.

This adapter preserves identity and delegates bounded metadata reads to the
existing field/assembly owners. It never registers accepted results as uploads.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from agent.schemas import StepOutputRef, PriorOutputRef
from agent.schemas.verification_authority import digest
from agent.tools._file_identity import file_sha256, checked_file


def checksum_field(port):
    """The registered digest field or an exact atomic manifest-port member."""
    field=port.observation_sha256_field or port.accepted_file_sha256_field
    if field is not None:return field
    members=[m.field_name for m in port.members if m.name in ('sha256','manifest_sha256')]
    return members[0] if len(members)==1 else None


@dataclass(frozen=True)
class ExistingObject:
    path: str
    sha256: str
    accepted_binding: object = None
    table_format: str | None = None
    object_type: str | None = None
    graph_inputs: tuple = ()
    publication_parameters: Mapping = field(default_factory=dict)

    def matches_sha256(self, value, registry):
        """Compare a literal digest or its exact accepted atomic-port member."""
        if isinstance(value,str):return value==self.sha256
        if not isinstance(value,PriorOutputRef) or self.accepted_binding is None or value.binding!=self.accepted_binding:
            return False
        from agent.orchestration.prior_outputs import source_port
        port=source_port(registry,value.binding.tool_name,value.binding.source_port)
        field=checksum_field(port)
        return field is not None and value.output_key==field

    def fields(self, axis):
        from agent.tools.analysis.categorical import fields
        from .dialogue_evidence import _unsafe_fact_field, _unsafe_fact_value
        checked_file(self.path, self.sha256)
        names = fields(self.path, axis=axis)
        checked_file(self.path, self.sha256)
        return tuple(name for name in names[:64] if len(name) <= 80 and name.strip()
            and not _unsafe_fact_field(name) and not _unsafe_fact_value(name))

    def read_bytes(self, *, max_bytes=4*1024**2):
        """Bounded exact manifest bytes, retaining the selected file identity."""
        checked_file(self.path,self.sha256)
        with open(self.path,'rb') as stream:raw=stream.read(max_bytes+1)
        if len(raw)>max_bytes:raise ValueError('Selected manifest exceeds the preparation metadata bound.')
        checked_file(self.path,self.sha256)
        return raw

    def slots(self, section):
        """Shallow physical slot names; no array values or scientific eligibility."""
        if section not in ('layers','obsm','obsp'):raise ValueError('Unknown slot section.')
        import h5py
        from agent.tools.analysis.peak_h5ad import metadata_item
        checked_file(self.path,self.sha256)
        with h5py.File(self.path,'r') as handle:
            group=metadata_item(handle,section)
            from itertools import islice
            names=tuple(islice(group,64)) if isinstance(group,h5py.Group) else ()
            names=tuple(name for name in names[:64] if len(name)<=80 and metadata_item(handle,section+'/'+name) is not None)
        checked_file(self.path,self.sha256)
        return names

    def identity(self):
        if self.accepted_binding is not None:
            return dict(accepted_binding=self.accepted_binding.to_dict(), source_sha256=self.sha256)
        return dict(source_path=self.path, source_sha256=self.sha256)


def identity_parent(value, selected, registry):
    if value.step_id not in selected:return None
    tool_name,arguments=selected[value.step_id]
    semantic=registry.get(tool_name).semantic_planning
    if semantic is None:return None
    ports=[port for port in semantic.producer_ports if any(member.field_name==value.output_key for member in port.members)]
    if len(ports)!=1 or ports[0].identity_from_port is None:return None
    consumer=next(port for port in semantic.consumer_ports if port.name==ports[0].identity_from_port)
    return arguments.get(consumer.members[0].field_name)


def identity_source(value, selected, registry):
    """Follow only an explicit byte-transparent producer contract."""
    seen = set()
    while isinstance(value, StepOutputRef):
        if value.step_id in seen:return None
        seen.add(value.step_id)
        value=identity_parent(value,selected,registry)
    return value


def existing_object(value, selected, registry, *, store=None):
    value = identity_source(value, selected, registry)
    if isinstance(value, PriorOutputRef):
        from agent.orchestration.prior_outputs import validate_binding, source_port
        from agent.schemas.verification_authority import AuthorityError
        stored = validate_binding(value.binding, store=store, registry=registry, integrity=True)
        port = source_port(registry, value.binding.tool_name, value.binding.source_port)
        if value.output_key not in {m.field_name for m in port.members}:
            raise AuthorityError('Observation member is outside the accepted source port.')
        path = stored.result[value.output_key]
        sha_field = checksum_field(port)
        if sha_field is None:
            raise AuthorityError('Accepted source port has no exact observable file identity.')
        sha = stored.result[sha_field]
        if not isinstance(path,(str,Path)) or not isinstance(sha,str):
            raise AuthorityError('Selected operation did not publish this observable file and digest.')
        checked_file(path, sha)
        table_format=stored.result.get('result_format') if port.semantic_type=='native_genomic_table.v1' else None
        graph_inputs=stored.result.get('parameters', {}).get('graph_inputs', ()) if port.semantic_type=='native_matrix_h5ad.v1' else ()
        parameters=stored.result.get('parameters',{}) if port.semantic_type=='native_matrix_h5ad.v1' else {}
        return ExistingObject(str(path), sha, value.binding, table_format, stored.result.get('object_type'), tuple(graph_inputs), parameters)
    if isinstance(value, (str, Path)):
        return ExistingObject(str(value), file_sha256(value))
    return None


def source_offers(interaction, preceding, tool, port_name):
    """Captured, registered file choices for one exact source-consuming port."""
    from agent.schemas.orchestration import _serialize
    from .dialogue_execution import captured_input_group
    binding = captured_input_group(interaction, preceding)
    types = tool.source_input_types.get(port_name, ())
    if not types or binding is None:
        return ()
    return tuple(dict(handle='source.'+str(index), label=member['label'], source_input=True,
        component=dict(path=member['source_path'],sha256=member['source_sha256'],member=_serialize(member)))
        for index, member in enumerate(binding.members)
        if member['input_type'] in types or binding.attachment_candidates and member['input_type']=='attachment')


def apply_source_choices(admitted, answers, requirements, registry, *, captured_group=None):
    """Add explicit missing source roles through the existing registration binder.

    The reducer has already checked the exact captured offer; this pure replay
    never re-pins source bytes or replaces an already supplied source.
    """
    from agent.schemas.orchestration import _serialize
    from .local_resources import LocalResourceAdmission, RegisteredInputGroup, _group_port, group_role_inputs
    source_answers = [a for a in answers if a['kind'] == 'resource'
        and isinstance(a['value'], Mapping) and a['value'].get('source_input')]
    if not source_answers:
        return admitted
    values = dict(admitted['inputs'])
    metadata = admitted.get('registered_input')
    if metadata is None and captured_group is not None:
        # A captured offer catalog is not execution attribution until a selected
        # source Answer below binds its exact member to a registered consumer.
        captured = LocalResourceAdmission._submission_binding(dict(
            registered_input=captured_group, execution_inputs=values))
        if type(captured) is not RegisteredInputGroup:
            raise ValueError('Source answers require a captured registered group.')
        metadata = captured.attribution()
    grouped = metadata is not None and 'members' in metadata
    members = [] if not grouped else _serialize(metadata['members'])
    roles = [] if not grouped else _serialize(metadata['roles'])
    changed = False
    for answer in source_answers:
        option = answer['value']
        req = requirements[answer['requirement_id']]
        tool = registry.get(req['tool_name'])
        port, source, _ = _group_port(registry, tool.name, req['port'])
        member = option['component']['member']
        for field in source.members:
            value = member['source_sha256'] if field.name=='sha256' else member['source_path']
            if field.input_name in values and values[field.input_name] != value:
                raise ValueError('A source answer cannot replace a captured source.')
            values[field.input_name] = value
        if grouped:
            indices = [i for i,m in enumerate(members) if m['resource_id']==member['resource_id']]
            if not indices:
                members.append(member); indices=[len(members)-1]
            elif members[indices[0]] != member:
                raise ValueError('A selected source registration changed.')
            role = dict(tool=tool.name,port=port.name,source='source.'+str(indices[0]))
            if role not in roles: roles.append(role)
        changed = True
    if not changed:
        return admitted
    admitted = dict(admitted, inputs=values)
    if grouped:
        binding = RegisteredInputGroup(members, values, roles=roles, attachment_candidates=True,
            resource_choice=metadata.get('resource_choice'))
        values=group_role_inputs(binding, roles, registry)
        admitted.update(inputs=values, registered_input=RegisteredInputGroup(members, values,
            roles=roles,attachment_candidates=True,resource_choice=binding.resource_choice).attribution(), roles=roles)
    return admitted


def accepted_input(context, port_name, arguments):
    """Fill an absent publication port from an exact canonical captured choice.

    Registry channels and the Session's captured current Revision own eligibility.
    Uploaded lookalikes and arbitrary filesystem paths never enter these offers.
    """
    from agent.orchestration.prior_outputs import channel_for
    from agent.schemas.verification_authority import AuthorityError
    from agent.orchestration.scientific_inputs import ScientificInputRequired,ScientificInputRequirement,ScientificDiagnostic
    port=next(p for p in context.tool.semantic_planning.consumer_ports if p.name==port_name)
    if all(arguments.get(m.field_name) is not None for m in port.members):return arguments
    offers=[];bindings={}
    if context.sessions is not None:
        current,_=context.sessions.planning_context_for_interaction(context.sessions._interaction_session_id,
            context.interaction.turn_id,integrity=True,reuse=True,admitted=context.admitted)
        for item in () if current is None else current.items:
            try:channel=channel_for(item.binding,context.tool.name,port_name,context.registry)
            except AuthorityError:continue
            token='accepted.'+digest(item.binding.to_dict())
            offers.append(dict(handle=token,label=item.binding.tool_name+' · '+item.binding.step_id,
                accepted_binding=item.binding.to_dict()))
            bindings[token]=(item.binding,channel)
    first=port.members[0].field_name
    chosen=context.answers.get(first)
    if chosen is not None:
        if chosen not in offers:raise ScientificDiagnostic('ACCEPTED_INPUT_CHANGED','The chosen publication is outside this captured consumer context.')
        binding,channel=bindings[chosen['handle']];values=dict(arguments)
        for member in channel.members:
            value=PriorOutputRef(binding,member.output_key)
            if member.argument_name in values and values[member.argument_name]!=value:
                raise ScientificDiagnostic('ACCEPTED_INPUT_CONFLICT','An accepted-result answer cannot replace an existing source operand.')
            values[member.argument_name]=value
        return values
    raise ScientificInputRequired(ScientificInputRequirement(port_name,first,'resource',
        dependencies=('source:bytes',),offers=tuple(offers),
        messages=('Select the exact accepted '+port_name+' result. An uploaded lookalike cannot supply publication authority.',)))
