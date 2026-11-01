# Fixed native dispatch. User arguments are JSON data, never parsed R code.
suppressPackageStartupMessages({library(motifmatchr);library(TFBSTools)
 library(GenomicRanges);library(Rsamtools);library(Matrix);library(SummarizedExperiment)})
stopifnot(as.character(packageVersion('motifmatchr'))=='1.32.0')
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path,simplifyVector=FALSE)
setwd(req$stage)
if (!identical(req$function_name,'matchMotifs')) stop('Unsupported native function')
p <- read.delim('peaks.tsv',check.names=FALSE,quote='',stringsAsFactors=FALSE,
 colClasses=c('integer','character','character','numeric','numeric'))
gr <- GRanges(p$chrom,IRanges(p$start+1,p$end)); names(gr)<-p$feature_id
pfms <- readJASPARMatrix(req$motifs,matrixClass='PFM')
kwargs <- req$kwargs
# JSON arrays/named objects represent native numeric background vectors.
if (is.list(kwargs$bg)) kwargs$bg <- unlist(kwargs$bg,use.names=TRUE)
result <- tryCatch(
 do.call(motifmatchr::matchMotifs,c(list(pwms=pfms,subject=gr,genome=FaFile(req$fasta)),kwargs)),
 error=function(e) {
   jsonlite::write_json(list(message=conditionMessage(e)),'native-error.json',auto_unbox=TRUE)
   stop(e)
 })
matches <- motifMatches(result)
stopifnot(inherits(matches,'lgCMatrix'),identical(rownames(result),p$feature_id),
 ncol(result)==length(pfms),identical(assayNames(result),'motifMatches'))
writeMM(matches,'matches.mtx')
write.table(data.frame(native_name=colnames(result),motif_id=ID(pfms),name=name(pfms)),
 'motifs.tsv',sep='\t',quote=FALSE,row.names=FALSE)
versions <- vapply(c('motifmatchr','TFBSTools','GenomicRanges','Biostrings','Matrix','SummarizedExperiment','Rsamtools'),
 function(p)as.character(packageVersion(p)),'')
jsonlite::write_json(list(completion='succeeded',function_name=req$function_name,
 request_sha256=digest::digest(file=request_path,algo='sha256'),versions=as.list(versions),R=R.version.string,
 result_class=class(result),matrix_class=class(matches),dim=dim(result),nnz=length(matches@x),
 rows_preserved=TRUE,assays=assayNames(result),chromVAR_loaded='chromVAR' %in% loadedNamespaces(),
 session=capture.output(sessionInfo())), 'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
