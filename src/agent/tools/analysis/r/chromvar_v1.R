# Thin native invocation: scientific algorithms remain chromVAR and motifmatchr.
suppressPackageStartupMessages({library(chromVAR);library(motifmatchr);library(TFBSTools)
 library(SummarizedExperiment);library(Rsamtools);library(Matrix);library(BiocParallel)})
stopifnot(as.character(packageVersion('chromVAR'))=='1.32.0', as.character(packageVersion('motifmatchr'))=='1.32.0')
register(SerialParam())
req <- jsonlite::fromJSON(commandArgs(TRUE)[1]); setwd(req$stage)
peaks <- read.delim('peaks.tsv',check.names=FALSE,quote='',stringsAsFactors=FALSE)
cells <- read.delim('cells.tsv',check.names=FALSE,quote='',stringsAsFactors=FALSE)
x <- as(readMM('counts.mtx'),'CsparseMatrix')
dimnames(x) <- list(peaks$feature_id,cells$cell_id)
gr <- GRanges(peaks$chrom,IRanges(peaks$start+1,peaks$end)); names(gr)<-peaks$feature_id
se <- SummarizedExperiment(list(counts=x),rowRanges=gr)
keep <- filterPeaks(se,non_overlapping=FALSE,ix_return=TRUE)
if (!length(keep)) stop('No positive-total peaks remain for native backgrounds')
se <- se[keep,]
fa <- FaFile(req$fasta)
se <- addGCBias(se,genome=fa)
if (any(!is.finite(rowData(se)$bias))) stop('Reference sequence has undefined GC for a retained peak; no base substitution is performed')
pfms <- readJASPARMatrix(req$motifs,matrixClass='PFM')
mm <- matchMotifs(pfms,se,genome=fa,out='matches',p.cutoff=req$p_cutoff,bg='subject')
set.seed(req$seed)
bg <- getBackgroundPeaks(se,niterations=req$background_iterations)
dev <- computeDeviations(se,mm,background_peaks=bg)
# Column-major R arrays are read with order='F'; no decimal precision loss.
writeBin(as.double(deviations(dev)),'deviations.bin',size=8,endian='little')
writeBin(as.double(deviationScores(dev)),'z.bin',size=8,endian='little')
writeBin(as.integer(bg),'background.bin',size=4,endian='little')
writeMM(motifMatches(mm),'matches.mtx')
write.table(data.frame(source_index=keep-1,gc=rowData(se)$bias), 'retained.tsv',sep='\t',quote=FALSE,row.names=FALSE)
write.table(data.frame(motif_id=ID(pfms),name=name(pfms),fraction_matches=rowData(dev)$fractionMatches,
 fraction_background_overlap=rowData(dev)$fractionBackgroundOverlap),'motifs.tsv',sep='\t',quote=FALSE,row.names=FALSE)
seqs <- getSeq(fa,rowRanges(se))
# Sequence identity is evidence; do not publish reference sequence as original counts.
write.table(data.frame(source_index=keep-1,sequence_sha256=vapply(as.character(seqs),function(s)
 digest::digest(s,algo='sha256',serialize=FALSE),'')), 'sequence-identities.tsv',sep='\t',quote=FALSE,row.names=FALSE)
versions<-vapply(c('chromVAR','motifmatchr','TFBSTools','Biostrings','Matrix','SummarizedExperiment','Rsamtools'),function(p)as.character(packageVersion(p)),'')
jsonlite::write_json(list(completion='succeeded',versions=as.list(versions),R=R.version.string,
 input_dim=dim(x),retained_dim=dim(se),motifs=length(pfms),seed=req$seed,
 background_iterations=req$background_iterations,p_cutoff=req$p_cutoff,motif_background='subject',
 filtering=list(min_fragments_per_peak=1,non_overlapping=FALSE,sample_filter='none'),
 undefined_deviations=sum(!is.finite(deviations(dev))),undefined_z=sum(!is.finite(deviationScores(dev))),
 session=capture.output(sessionInfo())), 'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
