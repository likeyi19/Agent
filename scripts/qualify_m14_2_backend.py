"""Explicit non-biological qualification using the real EpiZoo and SEAM weights.

Run with --output pointing outside Git to a new directory. No downloads,
production schedule, biological vocabulary truncation, or fallback weights.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch

from agent.tools.models.epizoo_adaptation import resources,contract as c,publication
from agent.tools.data.regulatory_feature_reference import build_regulatory_feature_reference,publish_regulatory_feature_reference
from agent.tools.data.scatac_reference import SourceProvenance
from agent.tools.data.regulatory_matrix_adoption import adopt_cell_by_features
from agent.tools.data.authority_context import VerificationContext,authority_operation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epizoo',type=Path,default=Path('/home/likeyi/program/EpiZoo'))
    parser.add_argument('--models',type=Path,default=Path('/home/likeyi/program/model_checkpoints/EpiZoo'))
    parser.add_argument('--steps',type=int,default=2)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    source=resources.qualify_source_bundle(checkpoint_path=args.models/'pretrained_EpiZoo.pth',
        resources_dir=args.epizoo/'data',output_path=root/'source-bundle.json')
    seam=resources.qualify_seam_bundle(checkpoint_path=args.models/'SEAM.pth',config_dir=args.epizoo/'config/SEAM_config',
        source_url=resources.SEAM_URL,output_path=root/'seam-bundle.json')
    species=dict(scientific_name='Danio rerio',taxonomy_id=7955)
    # Artificial sequence and complete artificial vocabulary: no biological claim.
    genome=''.join(np.random.default_rng(0).choice(list('ACGT'),8192))
    fasta=root/'fixture.fa';fasta.write_text('>fixture\n'+genome+'\n')
    fai=root/'fixture.fa.fai';fai.write_text('fixture\t8192\t9\t8192\t8193\n')
    coordinates=[(i*192,i*192+128) for i in reversed(range(32))]
    bed=root/'fixture.bed';bed.write_text(''.join(f'fixture\t{a}\t{b}\n' for a,b in coordinates))
    reference=build_regulatory_feature_reference(species=species,target_assembly='M14.2-technical-fixture.v1',
        fasta_path=fasta,fai_path=fai,feature_bed_path=bed,feature_category='regulatory_regions',
        genome_provenance=SourceProvenance('caller_supplied',source='Artificial DNA; technical qualification only'),
        feature_provenance=SourceProvenance('caller_supplied',source='Complete 32-feature artificial vocabulary'))
    ref=publish_regulatory_feature_reference(reference,root/'reference.json')
    rows=np.repeat(np.arange(4),3);cols=np.array([0,3,8,1,10,20,2,9,31,5,11,26])
    matrix=sp.csr_matrix((np.tile([5,1,2],4).astype(np.int64),(rows,cols)),shape=(4,32))
    data=ad.AnnData(matrix,obs=pd.DataFrame(index=['fixture-cell-'+str(i) for i in range(4)]),
        var=pd.DataFrame(index=[f'fixture:{a}-{b}' for a,b in coordinates]))
    path=root/'fixture.h5ad';data.write_h5ad(path)
    adopted=adopt_cell_by_features(source_path=str(path),source_sha256=c.file_record(path)['sha256'],
        reference_manifest_path=ref['manifest_path'],reference_manifest_sha256=ref['manifest_sha256'],
        species=species,assembly=reference.target_assembly,matrix_semantics='fragment_counts',output_dir=str(root/'matrix'))
    binding=lambda path: {k:v for k,v in c.file_record(path).items() if k!='size_bytes'}
    arguments=dict(matrices=[{k:adopted[k] for k in ('manifest_path','manifest_sha256')}],
        source_bundle=binding(root/'source-bundle.json'),seam_bundle=binding(root/'seam-bundle.json'),
        strategy='de_novo',mapping=None,device='cuda:0',output_dir=str(root/'adaptation'),
        profile=c.execution_profile(purpose='qualification',seed=0,batch_size=1,sequence_batch_size=4,
            max_steps=args.steps,save_steps=args.steps,log_steps=1))
    c.write_json(root/'arguments.json',arguments)
    execution=c.digest(dict(qualification='m14.2-real-backend-artificial-vocabulary',arguments=arguments))
    with authority_operation(VerificationContext()) as context:
        output=publication.adapt_epizoo_species(**arguments,execution_identity=execution)
        # Exercise proof reuse, with no second training or target reload.
        publication.verify_public_result(arguments,output['result'])
        recovered=publication.recover_adaptation(arguments,execution)
        assert recovered==output['result']
        c.write_json(root/'accepted.json',dict(execution_identity=execution,**output))
    print('Completed technical qualification:',root)


if __name__=='__main__':main()
