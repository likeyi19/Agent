# Fixed package dispatch. Inputs are data; no evaluated user expressions.
suppressPackageStartupMessages({library(monaLisa); library(Biostrings); library(TFBSTools); library(SummarizedExperiment)})
stopifnot(as.character(packageVersion('monaLisa')) == '1.16.0')
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path, simplifyVector=FALSE)
setwd(req$stage)
stopifnot(identical(req$function_name, 'calcBinnedMotifEnrR'))
seqs <- readDNAStringSet('sequences.fa')
b <- read.delim('bins.tsv', check.names=FALSE, quote='', colClasses='character')
stopifnot(identical(names(seqs), b$feature_id))
bins <- factor(b$bin, levels=unlist(req$levels))
pfms <- readJASPARMatrix(req$motifs, matrixClass='PFM')
# monaLisa keys hit matrices by accessions; TF display names may be duplicated.
names(pfms) <- ID(pfms)
pwms <- toPWM(pfms)
kw <- req$kwargs
if (is.list(kw$GCbreaks)) kw$GCbreaks <- unlist(kw$GCbreaks)
result <- tryCatch(do.call(monaLisa::calcBinnedMotifEnrR, c(list(seqs=seqs, bins=bins, pwmL=pwms), kw)),
 error=function(e) {jsonlite::write_json(list(message=conditionMessage(e)), 'native-error.json', auto_unbox=TRUE); stop(e)})
stopifnot(inherits(result, 'SummarizedExperiment'), nrow(result) == length(pwms))
saveRDS(result, 'native-result.rds')
tab <- expand.grid(motif_index=seq_len(nrow(result)), bin_index=seq_len(ncol(result)))
tab$motif_id <- rownames(result)[tab$motif_index]
tab$bin <- colnames(result)[tab$bin_index]
for (a in assayNames(result)) tab[[a]] <- as.vector(assay(result, a))
write.table(tab, 'result.tsv', sep='\t', row.names=FALSE, quote=FALSE)
write.table(data.frame(motif_id=ID(pwms), name=name(pwms)), 'motifs.tsv', sep='\t', row.names=FALSE, quote=FALSE)
versions <- vapply(c('monaLisa','Biostrings','TFBSTools','SummarizedExperiment'), function(p) as.character(packageVersion(p)), '')
jsonlite::write_json(list(completion='succeeded', function_name=req$function_name,
 request_sha256=digest::digest(file=request_path, algo='sha256'), versions=as.list(versions),
 n_sequences=length(seqs), n_motifs=nrow(result), n_bins=ncol(result), assays=assayNames(result),
 result_class=class(result), session=capture.output(sessionInfo())), 'native-receipt.json', auto_unbox=TRUE, pretty=TRUE, digits=NA)
