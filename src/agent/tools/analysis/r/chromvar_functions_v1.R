# Closed dispatch on repository-reviewed functions; JSON is data, never R source.
suppressPackageStartupMessages({library(chromVAR);library(Matrix);library(SummarizedExperiment);library(BiocParallel)})
stopifnot(as.character(packageVersion('chromVAR'))=='1.32.0')
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path,simplifyVector=FALSE)
setwd(req$stage)
tryCatch({
 cells <- read.delim('cells.tsv',check.names=FALSE,quote='',colClasses='character',na.strings=NULL)
 features <- read.delim('features.tsv',check.names=FALSE,quote='',colClasses='character',na.strings=NULL)
 n <- nrow(cells); m <- nrow(features)
 kwargs <- req$kwargs
 if (!is.null(req$seed)) set.seed(req$seed)
 register(SerialParam(RNGseed=req$seed))
 if (req$function_name %in% c('computeVariability','getSampleCorrelation','getSampleDistance','differentialDeviations','differentialVariability')) {
   dev <- matrix(readBin('deviations.bin','double',n=m*n,size=8,endian='little'),nrow=m,ncol=n)
   z <- matrix(readBin('z.bin','double',n=m*n,size=8,endian='little'),nrow=m,ncol=n)
   dimnames(dev) <- dimnames(z) <- list(features$feature_id,cells$cell_id)
   rd <- S4Vectors::DataFrame(row.names=features$feature_id)
   if ('name' %in% names(features)) rd$name <- features$name
   object <- new('chromVARDeviations',SummarizedExperiment(list(deviations=dev,z=z),rowData=rd))
   if (is.list(kwargs$bootstrap_quantiles)) kwargs$bootstrap_quantiles <- unlist(kwargs$bootstrap_quantiles,use.names=FALSE)
   if (!is.null(req$group_key)) kwargs$groups <- cells$group
   fn <- switch(req$function_name, computeVariability=chromVAR::computeVariability,
     getSampleCorrelation=chromVAR::getSampleCorrelation, getSampleDistance=chromVAR::getSampleDistance,
     differentialDeviations=chromVAR::differentialDeviations, differentialVariability=chromVAR::differentialVariability)
   result <- do.call(fn,c(list(object=object),kwargs))
 } else {
   object <- as(readMM('counts.mtx'),'CsparseMatrix')
   dimnames(object) <- list(features$feature_id,cells$cell_id)
   if (!is.null(req$group_key)) kwargs$group <- cells$group
   if (req$function_name == 'getBackgroundPeaks') kwargs$bias <- as.numeric(features$bias)
   result <- switch(req$function_name,
     getBackgroundPeaks=do.call(chromVAR::getBackgroundPeaks,c(list(object=object),kwargs)),
     getTotalFragments=do.call(chromVAR::getTotalFragments,list(object=object)),
     computeExpectations=do.call(chromVAR::computeExpectations,c(list(object=object),kwargs)),
     getFragmentsPerPeak=do.call(chromVAR::getFragmentsPerPeak,c(list(object=object),kwargs)),
     getFragmentsPerSample=do.call(chromVAR::getFragmentsPerSample,c(list(object=object),kwargs)),
     stop('Unsupported native function'))
 }
 axis <- if (identical(req$function_name,'getFragmentsPerSample')) cells$cell_id else features$feature_id
 native_type <- paste(class(result),collapse=',')
 result_axis <- if (req$function_name == 'getTotalFragments') 'scalar' else if (req$function_name %in% c('getSampleCorrelation','getSampleDistance')) 'cells_by_cells' else if (req$function_name == 'getFragmentsPerSample') 'cells' else 'features'
 if (result_axis == 'cells_by_cells') {
   mat <- as.matrix(result)
   stopifnot(identical(dim(mat),c(n,n)), identical(rownames(mat),cells$cell_id), identical(colnames(mat),cells$cell_id))
   table <- as.data.frame(mat); names(table) <- paste0('cell_',seq_len(n)-1)
   axis <- cells$cell_id
 } else if (result_axis == 'scalar') {
   stopifnot(length(result)==1); table <- data.frame(value=as.numeric(result))
 } else if (req$function_name == 'getBackgroundPeaks') {
   stopifnot(is.matrix(result), nrow(result)==m)
   table <- as.data.frame(result); names(table) <- paste0('iteration_', seq_len(ncol(result)))
 } else if (is.data.frame(result)) {
   stopifnot(identical(rownames(result),axis))
   table <- result
 } else {
   stopifnot(length(result)==length(axis))
   table <- data.frame(value=as.numeric(result))
 }
 numeric_columns <- vapply(table,is.numeric,TRUE)
 undefined <- sum(vapply(table[numeric_columns],function(x)sum(!is.finite(x)),0L))
 if (result_axis != 'scalar') table <- cbind(data.frame(source_index=seq_along(axis)-1,source_id=axis),table)
 # 17 significant digits retain native doubles, including very small p-values.
 for (key in names(table)) if (is.numeric(table[[key]])) table[[key]] <- sprintf('%.17g',table[[key]])
 write.table(table,'result.tsv',sep='\t',quote=FALSE,row.names=FALSE,na='NA')
 versions <- vapply(c('chromVAR','Matrix','SummarizedExperiment','BiocParallel'),function(p)as.character(packageVersion(p)),'')
 jsonlite::write_json(list(completion='succeeded',function_name=req$function_name,
   request_sha256=digest::digest(file=request_path,algo='sha256'),versions=as.list(versions),R=R.version.string,
   input_dim=c(m,n),rows=nrow(table),columns=as.list(names(table)),native_output_type=native_type,
   result_axis=result_axis,undefined_values=undefined,seed=req$seed,RNGkind=RNGkind(),parallel_backend='SerialParam',
   session=capture.output(sessionInfo())), 'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
},error=function(e){
 jsonlite::write_json(list(message=conditionMessage(e)),'native-error.json',auto_unbox=TRUE)
 stop(e)
})
