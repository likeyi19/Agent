"""Shared qualified genome resolution and exact indexed interval extraction."""


def resolve_genome(assembly, species=None):
    from .sequence_prediction import library
    lib = library()
    matches = [e for e in lib.entries() if e['kind'] == 'genome'
               and e['resource']['assembly'] == assembly]
    if len(matches) != 1:
        raise ValueError('An explicit assembly with a unique qualified matching genome is required')
    genome = lib.check(matches[0])
    identity = genome['species']
    names = {str(v) for v in identity.values()}
    names.update({9606: ('human',), 10090: ('mouse',), 7955: ('zebrafish',)}.get(identity['taxonomy_id'], ()))
    if species is not None and species not in names:
        raise ValueError('Declared species differs from the qualified reference genome')
    return genome, matches[0]


def extract(fasta, chrom, start, end, strand='.'):
    if (strand not in ('+', '-', '.') or chrom not in fasta.references
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= fasta.get_reference_length(chrom)):
        raise ValueError('Invalid contig, strand or interval bounds; no clipping, padding or liftOver')
    sequence = fasta.fetch(chrom, start, end).upper()
    if len(sequence) != end - start:
        raise ValueError('Indexed reference returned an incomplete interval')
    if strand == '-':
        from Bio.Seq import Seq
        return str(Seq(sequence).reverse_complement())
    return sequence
