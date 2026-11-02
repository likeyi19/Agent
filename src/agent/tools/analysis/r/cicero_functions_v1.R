# Ordinary data frames and native public functions; never deserialize user R objects.
suppressPackageStartupMessages(library(cicero))
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path, simplifyVector=FALSE)
setwd(req$stage)
tryCatch({
  stopifnot(as.character(packageVersion('cicero')) == req$version)
  if (!is.null(req$seed)) set.seed(req$seed)
  input <- read.delim('input.tsv', check.names=FALSE, quote='', colClasses='character', na.strings=NULL)
  if (identical(req$function_name, 'generate_ccans')) {
    if ('coaccess' %in% names(input)) {
      numeric_scores <- suppressWarnings(as.numeric(input$coaccess))
      if (any(is.na(numeric_scores) & !input$coaccess %in% c('NA','NaN','nan'))) stop('Cannot faithfully convert a supplied coaccess value to numeric')
      input$coaccess <- numeric_scores
    }
    f <- cicero::generate_ccans
  } else if (identical(req$function_name, 'find_overlapping_ccans')) {
    f <- cicero::find_overlapping_ccans
  } else stop('Unsupported native function')
  result <- do.call(f, c(list(input), req$kwargs))
  stopifnot(is.data.frame(result))
  if (identical(req$function_name, 'generate_ccans')) {
    stopifnot(all(result$Peak %in% c(input$Peak1, input$Peak2)), !anyDuplicated(result$Peak))
  } else stopifnot(all(result$CCAN1 %in% input[[2]]), all(result$CCAN2 %in% input[[2]]))
  write.table(result, 'result.tsv', sep='\t', quote=FALSE, row.names=FALSE)
  versions <- vapply(c('cicero','monocle','igraph','GenomicRanges'),function(p)as.character(packageVersion(p)),'')
  jsonlite::write_json(list(completion='succeeded',package='cicero',function_name=req$function_name,
    request_sha256=digest::digest(file=request_path,algo='sha256'),versions=as.list(versions),
    native_signature=paste(capture.output(args(f)),collapse=' '),
    function_source_sha256=digest::digest(paste(deparse(f),collapse='\n'),algo='sha256',serialize=FALSE),
    native_output_type=class(result),R=R.version.string,session=capture.output(sessionInfo())),
    'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
},error=function(e){
  jsonlite::write_json(list(message=conditionMessage(e)),'native-error.json',auto_unbox=TRUE)
  stop(e)
})
