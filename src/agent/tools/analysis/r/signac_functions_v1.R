# Reviewed public matrix methods. JSON supplies data, never executable R.
suppressPackageStartupMessages({library(Signac); library(Matrix)})
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path, simplifyVector=FALSE)
setwd(req$stage)
tryCatch({
  stopifnot(as.character(packageVersion('Signac')) == req$version)
  cells <- read.delim('cells.tsv', check.names=FALSE, quote='', colClasses='character', na.strings=NULL)
  features <- read.delim('features.tsv', check.names=FALSE, quote='', colClasses='character', na.strings=NULL)
  object <- as(readMM('input.mtx'), 'CsparseMatrix')
  dimnames(object) <- list(features$feature_id, cells$cell_id)
  function_ <- switch(req$function_name,
    RunTFIDF=Signac::RunTFIDF, BinarizeCounts=Signac::BinarizeCounts,
    FindTopFeatures=Signac::FindTopFeatures, stop('Unsupported native function'))
  result <- do.call(function_, c(list(object=object), req$kwargs))
  if (identical(req$function_name, 'FindTopFeatures')) {
    table <- cbind(data.frame(feature_id=rownames(result)), result)
    for (key in names(table)) if (is.numeric(table[[key]])) table[[key]] <- sprintf('%.17g', table[[key]])
    write.table(table, 'feature-statistics.tsv', sep='\t', quote=FALSE, row.names=FALSE, na='NA')
    slots <- c('var.Signac_count', 'var.Signac_percentile', 'var.Signac_native_rank')
    measurement <- NULL
  } else {
    stopifnot(identical(dimnames(result), dimnames(object)))
    writeMM(result, 'output.mtx')
    slots <- c('X')
    measurement <- if (identical(req$function_name, 'BinarizeCounts')) 'binary_accessibility' else 'Signac_TFIDF'
  }
  versions <- vapply(c('Signac','Matrix','SeuratObject'),function(p)as.character(packageVersion(p)),'')
  method <- getS3method(req$function_name, 'default')
  jsonlite::write_json(list(completion='succeeded', package='Signac', function_name=req$function_name,
    request_sha256=digest::digest(file=request_path,algo='sha256'), versions=as.list(versions), R=R.version.string,
    native_signature=paste(capture.output(args(method)),collapse=' '),
    function_source_sha256=digest::digest(paste(deparse(method),collapse='\n'),algo='sha256',serialize=FALSE),
    native_output_type=paste(class(result),collapse=','), result_slots=as.list(slots), measurement=measurement,
    session=capture.output(sessionInfo())), 'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
}, error=function(e){
  jsonlite::write_json(list(message=conditionMessage(e)), 'native-error.json',auto_unbox=TRUE)
  stop(e)
})
