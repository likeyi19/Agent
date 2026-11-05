"""Annotation/comparison projections for the existing verified figure bundle.

No fitting, marker tests, label mapping or native scientific execution. Activity
summaries come from the annotation owner and exact published per-cell RP.
"""
from pathlib import Path

import h5py
from anndata.io import read_elem
import numpy as np
from matplotlib.lines import Line2D

from agent.tools.analysis import cell_label_comparison as comparison
from agent.tools.analysis import scatac_annotation as annotation
from . import visualization as v

CAPTIONS = {
    'annotation_composition': 'Native MAESTRO group and predicted cell-type composition, including unassigned cells. Candidate labels are computational annotations, not biological validation. Display omissions are explicit.',
    'marker_activity_dot': 'Genuine per-cell MAESTRO Enhanced regulatory potential (RP): color is log1p(mean raw RP), size is fraction RP > 0 in each recorded group. Representative genes follow positive native marker order. Accessibility-derived activity is not measured RNA expression; markers are dependent annotation evidence.',
    'marker_activity_distribution': 'Per-cell log1p(raw native MAESTRO RP) distributions for a bounded subset of the displayed representative markers and groups. All cells in each displayed group are retained. RP is accessibility-derived activity, not measured RNA expression or independent validation; raw RP is sensitive to library depth.',
    'annotation_umap': 'Exact accepted UMAP coordinates colored by native MAESTRO predictions, joined through exact original-source ancestry and ordered cells. Gray denotes unassigned cells. No UMAP refit or biological validation.',
    'reference_confusion': 'Exact library/cell-matched annotation counts against supplied reference labels, without label mapping. Missing predictions count as errors; missing reference labels are excluded from metrics and reported in coverage. Reference annotations are not infallible biological truth.',
    'reference_composition': 'Full-source predicted and reference compositions; populations can differ. Matched/unmatched cell and label coverage is reported separately. Gray denotes missing labels.',
    'reference_umap': 'The same accepted UMAP coordinates show fixed predictions, uploaded reference labels and agreement. Identical labels share colors across views. Gray means missing prediction/reference or unmatched coverage; no coordinate refit. Reference agreement is not independent biological validation.',
}


def _projection(step_id, kind, sources, value, metadata, *, width=10, height=6):
    digest = v._domain_digest('agent.annotation-figure.'+kind+'.v1', value)
    return v._FigureProjection(kind+'-'+step_id,kind,kind+'-'+step_id+'.png',(step_id,),tuple(sources),digest,
        int(width*v._DPI),int(height*v._DPI),v._DPI,
        dict(metadata,caption=CAPTIONS[kind],projection_version=2),v._ScientificPanelData(kind,value))


def _sha(source):
    return source["integrity"].get("sha256") or (source["integrity"].get("authoritative_digest") or {}).get("value") or v._sha256_file(Path(source["artifact_path"]))


def _pin(source):
    path=Path(source['artifact_path'])
    if path.resolve()!=path or path.is_symlink() or v._sha256_file(path)!=_sha(source):
        raise ValueError('Figure source differs from pinned publication')
    return path


def projections(snapshot):
    artifacts = snapshot.payload['artifacts'];result=[]
    for step in snapshot.payload['steps']:
        tool, sid = step['tool_name'],step['step_id']
        if tool not in ('annotate_scATAC_cell_types',comparison.TOOL):
            continue
        kind = 'scatac_cell_type_annotation_manifest_json' if tool=='annotate_scATAC_cell_types' else 'cell_label_comparison_manifest'
        source=v._source_artifact(artifacts,step_id=sid,kind=kind,tool_name=tool)
        path=_pin(source)
        if tool=='annotate_scATAC_cell_types':
            data=annotation.presentation_data(path,_sha(source))
            sources=[source]+[dict(producing_step_id=sid,tool_name=tool,artifact_kind='native_annotation_sidecar',
                artifact_path=p['path'],integrity=dict(sha256=p['sha256'])) for p in data.pop('sources') if p['path']!=str(path)]
            counts=dict(groups=data['group_order'],counts=data['group_counts'],composition=data['composition'])
            result.append(_projection(sid,'annotation_composition',sources,counts,
                dict(n_cells=len(data['cells']),groups_omitted=data['groups_omitted']),height=max(6,min(14,.26*max(len(counts['groups']),len(counts['composition']))+2))))
            if data['genes']:
                dot={k:data[k] for k in ('genes','group_order','mean_rp','fraction_positive','groups_omitted')}
                result.append(_projection(sid,'marker_activity_dot',sources,dot,
                    dict(marker_selection=data['marker_selection'],selected_marker_sources=data['selected_marker_sources'],groups_omitted=data['groups_omitted'],measurement=data['measurement'],dependent_evidence=True),
                    height=max(6,min(14,.27*len(data['group_order'])+2))))
                # Bounded dense payload is the actual native per-cell RP subset.
                distribution=dict(genes=data['genes'][:4],group_order=data['group_order'][:20],groups=data['groups'],
                    values=np.asarray(data['per_cell_rp'])[:,:4].tolist(),total_groups=len(set(data['groups'])),total_genes=len(data['genes']))
                result.append(_projection(sid,'marker_activity_distribution',sources,distribution,
                    dict(groups_omitted=max(0,len(set(data['groups']))-20),genes_omitted=max(0,len(data['genes'])-4),measurement=data['measurement'],dependent_evidence=True),width=12,height=8))
            # Only already accepted coordinates in this run. Later explicit
            # cross-run recoloring remains with the Interaction overlay owner.
            for coordinate in artifacts:
                if coordinate['artifact_kind']!='cell_umap_h5ad':continue
                cp=_pin(coordinate)
                from agent.tools.analysis.categorical import coordinates
                ids,xy=coordinates(cp,_sha(coordinate))
                from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY
                with h5py.File(cp,'r') as handle:metadata=read_elem(handle['uns'][PROVENANCE_KEY])
                if metadata.get('embedding',{}).get('source_sha256')!=data['source_sha256'] or list(ids)!=data['cells']:
                    continue  # No relationship: do not assume it or refit.
                value=dict(cell_ids=list(ids),coordinates=xy.tolist(),labels=data['labels'])
                suffix=coordinate['producing_step_id']
                from dataclasses import replace
                projection=_projection(sid,'annotation_umap',sources+[coordinate],value,
                    dict(coordinate_reused=True,correspondence='exact_original_source_and_ordered_cells',n_cells=len(ids)))
                result.append(replace(projection,figure_id=projection.figure_id+'-'+suffix,filename=projection.figure_id+'-'+suffix+'.png'))
        else:
            manifest,sha=comparison.load(path,_sha(source))
            pins={Path(p['path']).name:p for p in manifest['files']}
            report=comparison.read(pins['comparison.json']['path'],pins['comparison.json']['sha256'])
            overlay=comparison.read(pins['overlay.json']['path'],pins['overlay.json']['sha256'])
            sources=[source]+[dict(producing_step_id=sid,tool_name=tool,artifact_kind='cell_label_comparison_data',
                artifact_path=p['path'],integrity=dict(sha256=p['sha256'])) for p in (pins['comparison.json'],pins['overlay.json'])]
            if report['confusion']['rows']:
                result.append(_projection(sid,'reference_confusion',sources,report['confusion'],
                    dict(evaluated_cells=report['summary']['evaluated_cells'],raw_counts=True),
                    width=min(20,max(8,3+.4*len(report['confusion']['columns']))),height=min(20,max(7,3+.4*len(report['confusion']['rows'])))))
            result.append(_projection(sid,'reference_composition',sources,report['composition'],
                dict(matched_cells=report['summary']['matched_cells']),height=min(16,max(6,.25*len(report['composition'])+2))))
            if overlay['available']:
                result.append(_projection(sid,'reference_umap',sources,overlay,
                    dict(coordinate_reused=True,matched_cells=report['summary']['matched_cells'],
                        evaluated_cells=report['summary']['evaluated_cells'],n_cells=report['summary']['n_prediction_cells']),width=16,height=6))
    return result


def _palette(labels):
    categories=sorted({x for x in labels if x is not None})
    palette=dict(zip(categories,v._categorical_palette(len(categories))))
    palette[None]='#bdbdbd'
    return palette


def _bars(ax,labels,counts,title,colors):
    n=min(40,len(labels)); y=np.arange(n)
    ax.barh(y,counts[:n],color=colors[:n]);ax.set_yticks(y,labels[:n],fontsize=7)
    ax.invert_yaxis();ax.set_xlabel('Cell count');ax.set_title(title+(f' (showing {n}/{len(labels)})' if len(labels)>n else ''))
    for i,count in enumerate(counts[:n]):ax.text(count,i,' '+str(count),va='center',fontsize=7)
    ax.margins(x=.2)


def _scatter(ax,xy,labels,palette,title):
    # Missing data are a structural category, never an invented biological label.
    ax.scatter(xy[:,0],xy[:,1],c=[palette[x] for x in labels],s=max(.5,min(12,20000/len(xy))),alpha=.8,linewidths=0)
    ax.set_title(title);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.set_xticks([]);ax.set_yticks([])


def render(figure,data):
    kind,value=data.kind,data.value
    if kind=='annotation_composition':
        axes=figure.subplots(1,2);figure.subplots_adjust(left=.2,right=.94,wspace=.9,bottom=.12,top=.88)
        _bars(axes[0],value['groups'],value['counts'],'Recorded groups',['#4477aa']*len(value['groups']))
        palette=_palette([r['label'] for r in value['composition']])
        _bars(axes[1],[r['label'] if r['label'] is not None else 'Unassigned (missing)' for r in value['composition']],
            [r['n_cells'] for r in value['composition']],'Predicted cell types',[palette[r['label']] for r in value['composition']])
    elif kind=='marker_activity_dot':
        ax=figure.add_axes((.14,.24,.69,.65)); means=np.asarray(value['mean_rp']);fraction=np.asarray(value['fraction_positive'])
        x,y=np.meshgrid(np.arange(len(value['genes'])),np.arange(len(value['group_order'])))
        dots=ax.scatter(x.ravel(),y.ravel(),s=200*fraction.ravel(),c=np.log1p(means).ravel(),cmap='viridis',vmin=0)
        ax.set_xticks(range(len(value['genes'])),value['genes'],rotation=50,ha='right',fontsize=8)
        ax.set_yticks(range(len(value['group_order'])),value['group_order']);ax.invert_yaxis();ax.set_ylabel('Recorded group')
        ax.set_title('Native marker activity: accessibility-derived RP' + (f" (showing {len(value['group_order'])}/{len(value['group_order'])+value['groups_omitted']} groups)" if value['groups_omitted'] else ''))
        figure.colorbar(dots,ax=ax,label='log1p(mean raw RP)',fraction=.04)
        handles=[ax.scatter([],[],s=200*f,color='#555555',label=f'{f:.0%}') for f in (.1,.5,1)]
        ax.legend(handles=handles,title='Fraction RP > 0',loc='upper left',bbox_to_anchor=(1.18,1),fontsize=7)
    elif kind=='marker_activity_distribution':
        axes=figure.subplots(2,2,squeeze=False);figure.suptitle(f"Per-cell native RP: {len(value['genes'])}/{value['total_genes']} selected genes; {len(value['group_order'])}/{value['total_groups']} groups",fontsize=11);figure.subplots_adjust(left=.08,right=.97,bottom=.15,top=.9,hspace=.7,wspace=.3)
        values=np.log1p(np.asarray(value['values']));groups=np.asarray(value['groups'])
        for j,ax in enumerate(axes.flat):
            if j>=len(value['genes']):ax.set_visible(False);continue
            chunks=[values[groups==g,j] for g in value['group_order']]
            # Matplotlib KDE is undefined on constant/singleton data; retain its
            # exact observed point instead of fabricating a distribution.
            for i,chunk in enumerate(chunks):
                if len(chunk)>1 and np.ptp(chunk)>0:
                    ax.violinplot([chunk],positions=[i],showextrema=True,showmedians=True)
                else:ax.scatter([i]*len(chunk),chunk,s=8,color='#4477aa')
            ax.set_xticks(range(len(chunks)),value['group_order'],rotation=60,fontsize=6)
            ax.set_ylabel('log1p(raw native RP)');ax.set_title(value['genes'][j])
    elif kind in ('annotation_umap','reference_umap'):
        xy=np.asarray(value['coordinates']);combined=value.get('predicted',value.get('labels'))+value.get('reference',[]);palette=_palette(value.get('category_order',combined))
        if kind=='annotation_umap':
            ax=figure.add_axes((.08,.12,.6,.78));_scatter(ax,xy,value['labels'],palette,'MAESTRO predicted cell types')
        else:
            axes=figure.subplots(1,3);figure.subplots_adjust(left=.04,right=.99,bottom=.29,top=.87,wspace=.25)
            _scatter(axes[0],xy,value['predicted'],palette,'Fixed predictions')
            _scatter(axes[1],xy,value['reference'],palette,'Uploaded reference labels')
            labels=[None if p is None or r is None or not m else 'Agreement' if p==r else 'Disagreement'
                for p,r,m in zip(value['predicted'],value['reference'],value['matched'],strict=True)]
            agreement={'Agreement':'#228833','Disagreement':'#cc6677',None:'#bdbdbd'}
            _scatter(axes[2],xy,labels,agreement,'Agreement / missing coverage')
            axes[2].legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l or 'Missing / unmatched') for l,c in agreement.items()],fontsize=7)
            figure.text(.04,.02,f"Same coordinates; {sum(value['matched'])}/{len(xy)} cells matched; reference labeled {sum(r is not None for r in value['reference'])}; prediction labeled {sum(p is not None for p in value['predicted'])}.",fontsize=8)
        legend=list(palette.items());shown=legend[:30]+([(None,palette[None])] if len(legend)>30 else [])
        handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l if l is not None else 'Missing / unassigned') for l,c in shown]
        if len(legend)>30:
            figure.text(.04,.96,f'Legend shows 30/{len(legend)-1} labels plus missing coverage; all labels remain in downloads.',fontsize=8)
        figure.legend(handles=handles,loc='center left' if kind=='annotation_umap' else 'lower center',
            bbox_to_anchor=(.7,.5) if kind=='annotation_umap' else (.5,.06),ncol=1 if kind=='annotation_umap' else min(6,max(1,len(handles))),fontsize=7,frameon=False)
    elif kind=='reference_confusion':
        counts=np.asarray(value['counts']);rows=value['rows'];columns=value['columns']
        visible_rows=min(40,len(rows));visible_columns=min(40,len(columns)-1)
        selected=list(range(visible_columns))+[len(columns)-1]
        preview=counts[:visible_rows,selected]
        v._render_confusion(figure,v._ConfusionData(tuple(rows[:visible_rows]),
            tuple(columns[i] if columns[i] is not None else 'Unassigned (missing)' for i in selected),
            preview,preview.size<=400),row_axis="Reference annotation",title="Fixed-label comparison (raw counts)")
        if visible_rows<len(rows) or visible_columns<len(columns)-1:
            figure.text(.04,.02,f'Showing {visible_rows}/{len(rows)} reference and {visible_columns}/{len(columns)-1} predicted classes plus missing; {int(preview.sum())}/{int(counts.sum())} evaluated cells. Full confusion in downloads.',fontsize=8)
    elif kind=='reference_composition':
        axes=figure.subplots(1,2);figure.subplots_adjust(left=.22,right=.96,bottom=.12,top=.88,wspace=1.1)
        palette=_palette([r['label'] for r in value])
        for ax,source in zip(axes,('prediction','reference')):
            rows=[r for r in value if r['source']==source]
            _bars(ax,[r['label'] if r['label'] is not None else 'Missing / unassigned' for r in rows],
                [r['n_cells'] for r in rows],source.capitalize()+' full source',[palette[r['label']] for r in rows])
    else:raise ValueError('Unknown annotation figure kind')
