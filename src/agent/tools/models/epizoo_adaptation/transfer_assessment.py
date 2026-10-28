"""Observe native correspondence and apply a reproducible transfer preference.

Coverage and overlap are method-selection evidence, not a biological score.
The mapping returned by EpiZoo is reused unchanged for parameter transfer.
"""
from collections import Counter
from pathlib import Path
import numpy as np
from . import contract as c

POLICY = dict(id='epizoo-transfer-preference.v1', usable_mapping_fraction=0.20,
              median_lifted_overlap_fraction=0.50,
              denominator='all-target-features', min_overlap_bp=1,
              liftover_min_match='native-tool-default-0.95',
              interpretation='operational-preference;not-biological-performance-threshold')


def observe_mapping(mapping, n_target, n_reference, lifted, overlaps, *, min_overlap):
    lifted_ids = set()
    with Path(lifted).open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 4: raise ValueError('Unexpected native lifted BED.')
            lifted_ids.add(int(fields[3]))
    candidates = Counter(); best = {}; ties = Counter(); chosen = {}
    with Path(overlaps).open() as stream:
        for line in stream:
            f = line.split()
            if len(f) != 9: raise ValueError('Unexpected native overlap BED.')
            target, bp = int(f[3]), int(f[8])
            if bp < min_overlap: continue
            reference=int(f[7])
            candidates[target] += 1
            if bp > best.get(target, -1): best[target], ties[target] = bp, 1
            elif bp == best[target]: ties[target] += 1
            if mapping.get(target) == reference:
                chosen[target] = (bp, bp / (int(f[2])-int(f[1])), bp / (int(f[6])-int(f[5])))
    if (set(chosen) != set(mapping) or not set(mapping) <= lifted_ids
            or any(not 0 <= k < n_target or not 0 <= v < n_reference for k,v in mapping.items())):
        raise ValueError('Native mapping evidence does not match its returned indices.')
    def summary(values):
        return None if not values else dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values)))
    assignments = Counter(mapping.values())
    return dict(total_target_ccres=n_target, retained_reference_ccres=n_reference,
        lifted_target_ccres=len(lifted_ids), usable_mapped_target_ccres=len(mapping),
        usable_mapping_fraction=len(mapping)/n_target if n_target else 0.0,
        overlap_bp=summary([v[0] for v in chosen.values()]),
        lifted_overlap_fraction=summary([v[1] for v in chosen.values()]),
        reference_overlap_fraction=summary([v[2] for v in chosen.values()]),
        multiple_overlap_targets=sum(v>1 for v in candidates.values()),
        tied_best_overlap_targets=sum(v>1 for v in ties.values()),
        repeated_reference_assignments=sum(v-1 for v in assignments.values()),
        unique_reference_assignments=len(assignments), min_overlap_bp=min_overlap)


def select(requested, candidates):
    if requested not in ('auto','de_novo','mapped_reference'):
        raise ValueError('Unknown transfer preference.')
    if requested == 'de_novo':
        return dict(policy=POLICY, requested=requested, selected='de_novo',
                    reason='explicit-preference;mapping-not-requested', selected_candidate=None, candidates=[])
    if requested == 'mapped_reference':
        if len(candidates) != 1:
            raise ValueError('Explicit mapped_reference requires one selected compatible source reference and chain.')
        # Low coverage is not an admission veto. An empty map inherits no
        # reference-specific rows and cannot fulfil an explicit mapped request.
        if not candidates[0]['statistics']['usable_mapped_target_ccres']:
            raise ValueError('No usable retained-reference correspondence for the requested mapped transfer.')
        selected = candidates[0]
        reason = 'explicit-preference;automatic-threshold-not-applied'
    else:
        eligible = [v for v in candidates if
            v['statistics']['usable_mapping_fraction'] >= POLICY['usable_mapping_fraction']
            and v['statistics']['lifted_overlap_fraction'] is not None
            and v['statistics']['lifted_overlap_fraction']['median'] >= POLICY['median_lifted_overlap_fraction']]
        selected = min(eligible, key=lambda v:(-v['statistics']['usable_mapping_fraction'],
            -v['statistics']['lifted_overlap_fraction']['median'], v['identity'])) if eligible else None
        reason = ('qualified-coverage-and-overlap' if selected else
                  'evaluated-below-preference-criteria' if candidates else 'mapping-unavailable;not-evaluated')
    return dict(policy=POLICY,requested=requested,selected='mapped_reference' if selected else 'de_novo',
        reason=reason, selected_candidate=selected['identity'] if selected else None,candidates=candidates)


def assess(requested, mappings, source, reference, features, root):
    from .backend import mapping
    if requested == 'de_novo':
        decision=select(requested, [])
        c.write_json(root/'transfer-decision.json',decision)
        return decision, None, None
    candidates=[]; maps={}; specs={}
    for spec in mappings:
        identity=c.digest(spec)
        if identity in maps: continue
        directory=root/('mapping-'+identity); directory.mkdir()
        result=mapping(spec,source,reference,features,directory,evidence=True)
        stats=c.read_json(directory/'mapping-evidence.json')
        candidates.append(dict(identity=identity,source_species=spec['source_species'],
            source_assembly='hg38' if spec['source_species']=='human' else 'mm10',
            chain=spec['chain'],settings=dict(min_overlap_bp=spec['min_overlap'],
                liftover_min_match=0.95,tool_versions=spec['tool_versions']),statistics=stats))
        maps[identity]=result; specs[identity]=spec
    decision=select(requested,candidates)
    c.write_json(root/'transfer-decision.json',decision)
    selected=decision['selected_candidate']
    return decision,maps.get(selected),specs.get(selected)
