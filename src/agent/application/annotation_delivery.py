"""Deliver the single accepted hierarchy through shared native download owners."""
from agent.tools.analysis.hierarchical_annotation import CONTRACT
from .native_delivery import accepted_artifacts

ROLE = 'annotated.h5ad'


def accepted_annotations(app, session_id, revision, origin, output, *, step_id=None):
    return accepted_artifacts(app, session_id, revision, origin, output, step_id=step_id)


def accepted_annotation(app, session_id, revision, origin, output, *, step_id=None):
    return next(a for a in accepted_annotations(app, session_id, revision, origin, output, step_id=step_id)
                if a.role == ROLE)
