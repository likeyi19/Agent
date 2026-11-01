"""Attach reviewed owner publications to shared delivery through ToolRegistry."""
from .registry import ScientificDeliverySpec


def delivery_spec(owner, artifact_type, *, h5ad_roles=(), figure=None):
    def publication(args, output, authority):
        value, sha = owner.load(output['manifest_path'], output['manifest_sha256'])
        if (authority['artifact_contract'] != owner.CONTRACT or authority['science_profile'] != owner.PROFILE
                or authority['verifier'] != owner.VERIFIER or value['arguments'] != owner.arguments(args)
                or owner.result(value, output['manifest_path'], sha) != output):
            raise ValueError('Scientific delivery differs from its exact producing request')
        return value, sha

    def prepare(stream, role):
        if role not in owner.DELIVERY_FILES:
            raise ValueError('Unreviewed scientific file')
        from agent.application.native_delivery import prepare as shared_prepare
        if role.endswith('.xls'):
            return b''.join(line for line in stream.read().splitlines(keepends=True) if not line.startswith(b'#'))
        return shared_prepare(stream, role)

    return ScientificDeliverySpec(owner.CONTRACT, artifact_type, owner.DELIVERY_FILES, publication, prepare,
        projected_roles=tuple(name for name in owner.DELIVERY_FILES if name.endswith(('.json','.tsv','.xls'))),
        h5ad_roles=h5ad_roles, figure=figure)
