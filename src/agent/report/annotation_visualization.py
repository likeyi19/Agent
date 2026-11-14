"""Native annotation projections for the existing verified figure bundle.

No fitting, marker tests or native scientific execution. Activity
summaries come from the annotation owner and exact accepted Gene Activity X.
"""
from pathlib import Path
from textwrap import fill

import h5py
from anndata.io import read_elem
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from agent.tools.analysis import hierarchical_annotation as annotation
from agent.tools.analysis.hierarchical_annotation import presentation_data
from . import visualization as v

CAPTIONS = {
    'annotation_composition': 'Final cell_type_final labels may be inherited from an ancestor. Current-round annotation_status retains ambiguous or unresolved evidence; inherited labels are not independent confident subtype assignments. Branch IDs identify computational populations, not biological truth. Display omissions are explicit.',
    'marker_activity_dot': 'Genuine per-cell Gene Activity: color is log1p(mean Gene Activity), size is fraction activity > 0 in each recorded group. Representative genes follow positive native marker order. Accessibility-derived activity is not measured RNA expression; markers are dependent annotation evidence.',
    'marker_activity_heatmap': 'log1p of mean accepted Gene Activity X per displayed branch and representative marker. Display transformation only; no normalization or new marker ranking. Accessibility-derived activity is not measured RNA expression.',
    'annotation_hierarchy': 'Accepted branch relationships and cell counts; at most 60 branches. Nodes show assigned current-round candidates or the recorded evidence status. Small ID prefixes trace the complete hierarchy in Downloads & reports. Descendants may inherit an ancestor label; this is not independent subtype confidence.',
    'annotation_umap': 'Exact accepted UMAP coordinates colored by hierarchical annotations, joined through exact original-source ancestry and ordered cells. Final labels may be inherited from an ancestor; annotation_status retains the current-round evidence state. Gray denotes unassigned cells. No UMAP refit or biological validation.',
    'marker_activity_umap': 'Exact accepted UMAP colored by log1p(raw native Gene Activity), without additional normalization. All cells, including zero activity, are retained. Accessibility-derived activity is not measured RNA expression. Patterns may help inspect candidate annotations and within-cluster heterogeneity; markers used in annotation are dependent evidence, not independent biological validation. Raw Gene Activity is sensitive to library depth.',
}


def _projection(step_id, kind, sources, value, metadata, *, width=10, height=6):
    digest = v._domain_digest('agent.annotation-figure.'+kind+'.v1', value)
    caption = CAPTIONS[kind]
    if metadata.get('ambiguous_marker_names'):
        caption += ' Repeated selected feature names omitted because native ranks do not identify a unique column: ' + ', '.join(metadata['ambiguous_marker_names']) + '.'
    return v._FigureProjection(kind+'-'+step_id,kind,kind+'-'+step_id+'.png',(step_id,),tuple(sources),digest,
        int(width*v._DPI),int(height*v._DPI),v._DPI,
        dict(metadata,caption=caption,projection_version=2),v._ScientificPanelData(kind,value))


def _sha(source):
    return source["integrity"].get("sha256") or (source["integrity"].get("authoritative_digest") or {}).get("value") or v._sha256_file(Path(source["artifact_path"]))


def _pin(source):
    path=Path(source['artifact_path'])
    if path.resolve()!=path or path.is_symlink() or v._sha256_file(path)!=_sha(source):
        raise ValueError('Figure source differs from pinned publication')
    return path


def marker_activity_input(data, ids, xy, sources, *, basis):
    """Align the bounded native marker columns by exact, unique cell identity."""
    from agent.tools.analysis.categorical import ordered_ids, require_same_cells
    from agent.tools.analysis.embedding_analysis import _cell_order_digest
    activity_ids = ordered_ids(data['cells'], source='Gene Activity')
    require_same_cells(tuple(sorted(ids)), tuple(sorted(activity_ids)))
    order = pd.Index(activity_ids).get_indexer(ids)
    # Prefer one already selected positive marker per recorded group, then fill
    # from the same published native selection. Never inspect UMAP agreement.
    genes, groups = [], set()
    for row in data['selected_marker_sources']:
        if row['group'] not in groups:
            genes.append(row['gene']); groups.add(row['group'])
    genes = list(dict.fromkeys([*genes, *data['genes']]))[:6]
    if not genes:
        raise ValueError('Accepted annotation has no representative Gene Activity markers.')
    columns = [data['genes'].index(gene) for gene in genes]
    values = np.asarray(data['per_cell_activity'])[order][:, columns]
    return dict(cell_ids=list(ids), coordinates=np.asarray(xy).tolist(), genes=genes, values=values.tolist(),
        sources=sources, measurement=data['measurement'],
        marker_selection='one positive marker per recorded group from the existing native representative selection, then remaining selected markers in published order; deduplicated; at most six genes; no UMAP-based ranking',
        selected_marker_sources=[row for row in data['selected_marker_sources'] if row['gene'] in genes],
        ambiguous_marker_names=data.get('ambiguous_marker_names', []),
        correspondence=dict(basis=basis, n_cells=len(ids), cell_order_sha256=_cell_order_digest(ids)))


def marker_activity_projection(read_sources):
    value = read_sources()
    data = {k:value[k] for k in ('cell_ids', 'coordinates', 'genes', 'values')}
    digest = v._domain_digest('agent.marker-activity-umap.v1', dict(data, sources=value['sources']))
    return (v._FigureProjection('marker-activity-'+digest[:16], 'marker_activity_umap',
        'marker-activity-'+digest[:16]+'.png',
        tuple(dict.fromkeys(s['producing_step_id'] for s in value['sources'])), tuple(value['sources']),
        digest, 12*v._DPI, 8*v._DPI, v._DPI,
        dict(caption=CAPTIONS['marker_activity_umap'], projection_version=1,
            delivery='accepted_marker_activity_umap', coordinate_reused=True, correspondence=value['correspondence'],
            marker_selection=value['marker_selection'], selected_marker_sources=value['selected_marker_sources'],
            measurement=value['measurement'], ambiguous_marker_names=value.get('ambiguous_marker_names', []), transformation='log1p(Gene Activity)', normalization='none added',
            n_cells=len(value['cell_ids']), genes=value['genes'], dependent_evidence=True),
        v._ScientificPanelData('marker_activity_umap', data)),)


def projections(snapshot, *, marker_activity=True):
    artifacts = snapshot.payload['artifacts'];result=[]
    for step in snapshot.payload['steps']:
        tool, sid = step['tool_name'],step['step_id']
        if tool != 'annotate_scATAC_cell_types':
            continue
        source=v._source_artifact(artifacts,step_id=sid,kind='scatac_cell_type_annotation_manifest_json',tool_name=tool)
        path=_pin(source)
        data=presentation_data(path,_sha(source))
        data_source_path=next(p['path'] for p in data['sources'] if Path(p['path']).name=='annotated.h5ad')
        sources=[source]+[dict(producing_step_id=sid,tool_name=tool,artifact_kind='native_annotation_sidecar',
            artifact_path=p['path'],integrity=dict(sha256=p['sha256'])) for p in data.pop('sources') if p['path']!=str(path)]
        counts=dict(groups=data['group_order'],counts=data['group_counts'],composition=data['composition'],
            population_summary=data.get('population_summary', []), inherited_label_cells=data.get('inherited_label_cells', 0))
        result.append(_projection(sid,'annotation_composition',sources,counts,
            dict(n_cells=len(data['cells']),groups_omitted=data['groups_omitted']),width=13,height=max(7,min(12,.26*len(counts['composition'])+5))))
        if data['genes']:
            dot={k:data[k] for k in ('genes','group_order','mean_activity','fraction_positive','groups_omitted')}
            result.append(_projection(sid,'marker_activity_dot',sources,dot,
                dict(marker_selection=data['marker_selection'],selected_marker_sources=data['selected_marker_sources'],groups_omitted=data['groups_omitted'],measurement=data['measurement'],dependent_evidence=True,ambiguous_marker_names=data.get('ambiguous_marker_names', [])),
                height=max(6,min(14,.27*len(data['group_order'])+2))))
            result.append(_projection(sid,'marker_activity_heatmap',sources,dot,
                dict(marker_selection=data['marker_selection'],groups_omitted=data['groups_omitted'],measurement=data['measurement'],ambiguous_marker_names=data.get('ambiguous_marker_names', [])),width=12,height=8))
        result.append(_projection(sid,'annotation_hierarchy',sources,
            dict(branches=data['branches'],branches_omitted=data['branches_omitted']),
            dict(branches_omitted=data['branches_omitted']),width=11,
            height=max(4.5,min(12,.4*len(data['group_order'])+1.8))))
        # Native coordinates in the current analysis scope, or original input
        # coordinates retained exactly by the accepted annotation publication.
        coordinate_sources=[]
        for coordinate in artifacts:
            capability = snapshot.cell_results.get(coordinate['producing_step_id'])
            native = (coordinate['artifact_kind'] == 'native_matrix_h5ad' and capability is not None
                      and capability['coordinates'])
            if not native:continue
            if capability.get('cell_source_sha256',capability['source_sha256']) in data.get('coordinate_source_sha256s',[data['source_sha256']]):
                coordinate_sources.append((coordinate, capability, 'accepted_native_operation'))
        if not coordinate_sources:
            with h5py.File(data_source_path,'r') as handle:
                slot=handle.get('obsm/X_umap')
                supplied=(isinstance(slot,h5py.Dataset) and slot.shape==(len(data['cells']),2)
                    and slot.dtype.kind in 'fi' and np.isfinite(slot[()]).all())
            if supplied:
                coordinate_sources.append((next(s for s in sources if s['artifact_path']==data_source_path),
                    dict(coordinates='X_umap',source_sha256=data['source_sha256']), 'retained_original_input'))
        for coordinate, capability, coordinate_origin in coordinate_sources:
            cp=_pin(coordinate)
            from agent.tools.analysis.categorical import coordinates
            ids,xy=coordinates(cp,_sha(coordinate), capability['coordinates'])
            source_identity = capability.get('cell_source_sha256',capability['source_sha256'])
            if source_identity not in data.get('coordinate_source_sha256s',[data['source_sha256']]):
                continue  # No relationship: do not assume it or refit.
            if marker_activity and source_identity is not None and data['genes'] and set(ids)==set(data['cells']):
                inputs=marker_activity_input(data,ids,xy,sources+[coordinate],basis='exact_original_source')
                result.extend(marker_activity_projection(lambda:inputs))
            if list(ids)!=data['cells']:
                continue  # Existing categorical figures retain their ordered-axis contract.
            for level in [None, *range(1,min(data['levels_reached'],5)+1)]:
                with h5py.File(data_source_path, 'r') as handle:
                    obs=read_elem(handle['obs'])
                field='cell_type_final' if level is None else 'cell_type_level_'+str(level)
                labels=[None if pd.isna(x) else str(x) for x in obs[field]]
                value=dict(cell_ids=list(ids),coordinates=xy.tolist(),labels=labels)
                suffix=coordinate['producing_step_id']+'-'+('final' if level is None else 'level-'+str(level))
                from dataclasses import replace
                projection=_projection(sid,'annotation_umap',sources+[coordinate],value,
                    dict(coordinate_reused=True,coordinate_origin=coordinate_origin,
                         correspondence='exact_original_source_and_ordered_cells',n_cells=len(ids),
                         annotation_level='final' if level is None else level,levels_omitted=max(0,data['levels_reached']-5)))
                result.append(replace(projection,figure_id=projection.figure_id+'-'+suffix,filename=projection.figure_id+'-'+suffix+'.png'))
    from dataclasses import replace
    result = [replace(p, presentation=dict(p.presentation, compatible_umap_available=any(
        q.figure_kind=='annotation_umap' and q.source_step_ids==p.source_step_ids for q in result)))
        if p.figure_kind=='annotation_composition' else p for p in result]
    return result


def _palette(labels):
    categories=sorted({x for x in labels if x is not None})
    palette=dict(zip(categories,v._categorical_palette(len(categories)))) if categories else {}
    palette[None]='#bdbdbd'
    return palette


def _bars(ax,labels,counts,title,colors):
    n=min(40,len(labels)); y=np.arange(n)
    ax.barh(y,counts[:n],color=colors[:n]);ax.set_yticks(y,labels[:n],fontsize=7)
    ax.invert_yaxis();ax.set_xlabel('Cell count');ax.set_title(title+(f' (showing {n}/{len(labels)})' if len(labels)>n else ''))
    for i,count in enumerate(counts[:n]):ax.text(count,i,' '+str(count),va='center',fontsize=7)
    ax.margins(x=.2)


def _status_color(status):
    return {'assigned':'#377f89', 'ambiguous':'#c59b49', 'unresolved':'#88929f',
            'insufficient_evidence':'#afb7be', 'unassigned':'#bdbdbd'}.get(status, '#4477aa')


def _scatter(ax,xy,labels,palette,title):
    # Missing data are a structural category, never an invented biological label.
    ax.scatter(xy[:,0],xy[:,1],c=[palette[x] for x in labels],s=max(.5,min(12,20000/len(xy))),alpha=.8,linewidths=0)
    ax.set_title(title);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.set_xticks([]);ax.set_yticks([])


def _hierarchy_positions(rows):
    """Place each subtree together, with its parent centered over its children."""
    by_id = {row['branch_id']: row for row in rows}
    children = {key: [] for key in by_id}
    for row in rows:
        if row['parent'] in children:
            children[row['parent']].append(row['branch_id'])
    positions, visited = {}, set()
    cursor = 0
    def place(key):
        nonlocal cursor
        if key in visited:
            raise ValueError('Cyclic hierarchy figure source')
        visited.add(key)
        ys = [place(child) for child in children[key]]
        y = sum(ys)/len(ys) if ys else -cursor
        if not ys:
            cursor += .9
        positions[key] = (by_id[key]['level'], y)
        return y
    for row in rows:
        if row['parent'] not in by_id:
            place(row['branch_id'])
            cursor += .45
    if len(positions) != len(rows):
        raise ValueError('Incomplete hierarchy figure source')
    return positions


def render(figure,data):
    kind,value=data.kind,data.value
    if kind=='annotation_composition':
        grid=figure.add_gridspec(2,2,height_ratios=[3.5,1],left=.22,right=.96,bottom=.17,top=.88,wspace=.8,hspace=.65)
        axes=[figure.add_subplot(grid[0,0]),figure.add_subplot(grid[0,1]),figure.add_subplot(grid[1,:])]
        palette=_palette([r['label'] for r in value['composition']])
        _bars(axes[0],[fill(r['label'],26) if r['label'] is not None else 'Unassigned (missing)' for r in value['composition']],
            [r['n_cells'] for r in value['composition']],'Final labels (may be inherited)',[palette[r['label']] for r in value['composition']])
        statuses=value.get('population_summary', [])
        visible=[r for r in statuses if r['cells']]
        _bars(axes[1],[r['status'].replace('_',' ') for r in visible],[r['cells'] for r in visible],
            'Current-round evidence',[_status_color(r['status']) for r in visible])
        zeros=[r['status'].replace('_',' ') for r in statuses if not r['cells']]
        if zeros:axes[1].text(0,-.20,'Zero cells: '+', '.join(zeros),transform=axes[1].transAxes,fontsize=7,color='#78848c')
        x=np.arange(len(value['groups']));axes[2].bar(x,value['counts'],color='#a5b3bd',width=.65)
        axes[2].set_xticks(x,[g.removeprefix('branch-')[:8] for g in value['groups']],fontsize=6)
        axes[2].set_ylabel('Cells',fontsize=7);axes[2].set_title('Terminal populations · branch ID prefixes',fontsize=8,color='#78848c')
        axes[2].tick_params(axis='y',labelsize=6)
        for i,n in enumerate(value['counts']):axes[2].text(i,n,str(n),ha='center',va='bottom',fontsize=6,color='#667580')
        for ax in axes:
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(length=0,pad=6)
        figure.text(.5,.035,f"{value.get('inherited_label_cells', 0)} cells inherit ancestor labels; current-round ambiguous or unresolved evidence remains unchanged.",ha='center',fontsize=9)
    elif kind=='marker_activity_dot':
        ax=figure.add_axes((.20,.24,.53,.65)); means=np.asarray(value['mean_activity']);fraction=np.asarray(value['fraction_positive'])
        x,y=np.meshgrid(np.arange(len(value['genes'])),np.arange(len(value['group_order'])))
        dots=ax.scatter(x.ravel(),y.ravel(),s=200*fraction.ravel(),c=np.log1p(means).ravel(),cmap='viridis',vmin=0)
        ax.set_xticks(range(len(value['genes'])),value['genes'],rotation=50,ha='right',fontsize=8)
        ax.set_yticks(range(len(value['group_order'])),[g[:16] for g in value['group_order']],fontsize=7);ax.invert_yaxis();ax.set_ylabel('Branch ID prefix')
        ax.set_title('Native markers: accessibility-derived Gene Activity' + (f" (showing {len(value['group_order'])}/{len(value['group_order'])+value['groups_omitted']} groups)" if value['groups_omitted'] else ''))
        figure.colorbar(dots,ax=ax,label='log1p(mean Gene Activity)',fraction=.04)
        handles=[ax.scatter([],[],s=200*f,color='#555555',label=f'{f:.0%}') for f in (.1,.5,1)]
        ax.legend(handles=handles,title='Fraction activity > 0',loc='upper left',bbox_to_anchor=(1.30,1),fontsize=6,title_fontsize=6)
    elif kind=='marker_activity_heatmap':
        ax=figure.add_axes((.2,.25,.65,.65))
        values=np.log1p(np.asarray(value['mean_activity']))
        mesh=ax.imshow(values,aspect='auto',cmap='viridis',vmin=0)
        ax.set_xticks(range(len(value['genes'])),value['genes'],rotation=50,ha='right',fontsize=8)
        ax.set_yticks(range(len(value['group_order'])),[g[:16] for g in value['group_order']],fontsize=7)
        ax.set_title('Representative marker Gene Activity')
        figure.colorbar(mesh,ax=ax,label='log1p(mean Gene Activity X)')
    elif kind=='annotation_hierarchy':
        ax=figure.add_axes((.06,.16,.88,.72)); rows=value['branches'];positions=_hierarchy_positions(rows)
        for row in rows:
            x,y=positions[row['branch_id']]
            if row['parent'] in positions:
                px,py=positions[row['parent']];mid=(px+x)/2
                ax.plot([px,mid,mid,x],[py,py,y,y],color='#bbbbbb',linewidth=1,zorder=1)
            ax.scatter([x],[y],s=30,color=_status_color(row.get('status')),zorder=2)
            label=row.get('label') or (row.get('status') or 'Branch').replace('_',' ').capitalize()
            ax.text(x+.04,y+.08,fill(label,28),fontsize=9,va='bottom',
                bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=3)
            ax.text(x+.04,y-.12,str(row['n_cells'])+' cells · '+row['branch_id'].removeprefix('branch-')[:8],fontsize=7,color='#78848c',va='top',
                bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=3)
        ax.set_xticks(sorted({row['level'] for row in rows}))
        ax.set_yticks([]);ax.set_xlabel('Hierarchy level');ax.set_title('Cell-type annotation hierarchy'+
            (f" ({value['branches_omitted']} branches omitted; see hierarchy.json)" if value['branches_omitted'] else ''))
        if rows:ax.set_xlim(min(r['level'] for r in rows)-.12,max(r['level'] for r in rows)+.55)
        ax.margins(y=.12)
        ax.spines[:].set_visible(False);ax.tick_params(length=0)
        figure.text(.5,.035,'Current-round labels / evidence status · descendants may inherit an ancestor label\nSmall prefixes identify branches; complete identities and evidence are in hierarchy.json.',ha='center',fontsize=8,color='#667580')
    elif kind=='marker_activity_umap':
        xy=np.asarray(value['coordinates']); activity=np.log1p(np.asarray(value['values']))
        axes=figure.subplots(2,3,squeeze=False)
        figure.subplots_adjust(left=.06,right=.95,bottom=.12,top=.88,hspace=.45,wspace=.55)
        figure.suptitle('Marker activity on accepted UMAP — accessibility-derived Gene Activity',fontsize=12)
        figure.text(.5,.03,'log1p(Gene Activity); no additional normalization. Not measured RNA expression or independent validation.',ha='center',fontsize=9)
        for j,ax in enumerate(axes.flat):
            if j>=len(value['genes']):ax.set_visible(False);continue
            dots=ax.scatter(xy[:,0],xy[:,1],c=activity[:,j],cmap='viridis',vmin=0,
                s=max(.5,min(12,20000/len(xy))),alpha=.8,linewidths=0)
            ax.set(title=value['genes'][j],xlabel='UMAP 1',ylabel='UMAP 2',xticks=[],yticks=[])
            figure.colorbar(dots,ax=ax,label='log1p(Gene Activity)',fraction=.05,pad=.03)
    elif kind=='annotation_umap':
        xy=np.asarray(value['coordinates']);palette=_palette(value['labels'])
        ax=figure.add_axes((.08,.12,.6,.78));_scatter(ax,xy,value['labels'],palette,'Hierarchical cell types')
        legend=list(palette.items());shown=legend[:30]+([(None,palette[None])] if len(legend)>30 else [])
        handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l if l is not None else 'Missing / unassigned') for l,c in shown]
        if len(legend)>30:
            figure.text(.04,.96,f'Legend shows 30/{len(legend)-1} labels plus missing coverage; all labels remain in downloads.',fontsize=8)
        figure.legend(handles=handles,loc='center left',bbox_to_anchor=(.7,.5),fontsize=7,frameon=False)
    else:raise ValueError('Unknown annotation figure kind')
