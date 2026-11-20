"""Attach reviewed owner publications to shared delivery through ToolRegistry."""
from .registry import ScientificDeliverySpec


def delivery_spec(owner, artifact_type, *, h5ad_roles=(), figure=None):
    def publication(args, output, authority):
        value, sha = owner.load(output['manifest_path'], output['manifest_sha256'])
        # The owner has admitted this exact current or historical profile.
        # Accepted authority must match the publication, not today's catalog.
        if (authority['artifact_contract'] != value['contract'] or authority['science_profile'] != value['profile']
                or authority['verifier'] != owner.VERIFIER or value['arguments'] != owner.arguments(args)
                or owner.result(value, output['manifest_path'], sha) != output):
            raise ValueError('Scientific delivery differs from its exact producing request')
        return value, sha

    def prepare(stream, role, *, destination=None):
        if role not in owner.DELIVERY_FILES:
            raise ValueError('Unreviewed scientific file')
        from agent.application.native_delivery import prepare as shared_prepare
        return shared_prepare(stream, role, destination=destination)

    return ScientificDeliverySpec(owner.CONTRACT, artifact_type, owner.DELIVERY_FILES, publication, prepare,
        projected_roles=tuple(name for name in owner.DELIVERY_FILES if name.endswith(('.json','.tsv','.xls'))),
        h5ad_roles=h5ad_roles, figure=figure, produced_files=getattr(owner, 'delivery_files', None))
