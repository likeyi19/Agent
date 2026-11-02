# Closed native CDS adaptation; JSON and MatrixMarket are data, never R code/RDS.
suppressPackageStartupMessages({library(cicero); library(Matrix)})
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path, simplifyVector=FALSE)
setwd(req$stage)
tryCatch({
  stopifnot(as.character(packageVersion('cicero')) == '1.28.3')
  if (!is.null(req$seed)) set.seed(req$seed)
  to_frame <- function(obj) {
    ids <- unlist(obj$ids, use.names=FALSE)
    stopifnot(is.character(ids), !anyDuplicated(ids), !anyNA(ids))
    out <- data.frame(row.names=ids)
    for (key in names(obj$columns)) {
      col <- obj$columns[[key]]
      v <- vapply(col$values, function(x) if (is.null(x)) NA_character_ else as.character(x), '')
      stopifnot(length(v) == length(ids))
      out[[key]] <- switch(col$type, numeric=as.numeric(v), integer=as.integer(v),
        character=v, logical=as.logical(v), factor=factor(v, levels=unlist(col$levels), ordered=isTRUE(col$ordered)),
        stop('Unsupported safe CDS column representation'))
    }
    out
  }
  from_frame <- function(df) {
    columns <- lapply(df, function(v) {
      if (is.factor(v)) return(list(type='factor',values=as.list(as.character(v)),levels=as.list(levels(v)),ordered=is.ordered(v)))
      if (is.numeric(v)) return(list(type=if(is.integer(v)) 'integer' else 'numeric',values=as.list(sprintf('%.17g',v))))
      if (is.logical(v)) return(list(type='logical',values=as.list(as.character(v))))
      if (is.character(v)) return(list(type='character',values=as.list(v)))
      stop('Native CDS returned an unrepresented column type')
    })
    list(ids=as.list(rownames(df)),columns=columns)
  }
  obj <- jsonlite::fromJSON('object.json', simplifyVector=FALSE)
  stopifnot(identical(obj$format,'cicero-cds-exchange.v1'))
  pd <- to_frame(obj$observations); fd <- to_frame(obj$regions)
  x <- as(readMM('exprs.mtx'), 'CsparseMatrix')
  stopifnot(identical(dim(x), c(nrow(fd),nrow(pd))))
  if (identical(obj$storage, 'dense')) x <- as.matrix(x)
  dimnames(x) <- list(rownames(fd),rownames(pd))
  cds <- monocle::newCellDataSet(x, phenoData=new('AnnotatedDataFrame',data=pd),
    featureData=new('AnnotatedDataFrame',data=fd), expressionFamily=VGAM::negbinomial.size(),lowerDetectionLimit=0)
  # The constructor initializes size factors; supplied actual factors must survive.
  if ('Size_Factor' %in% names(pd)) Biobase::pData(cds)$Size_Factor <- pd$Size_Factor
  kw <- req$kwargs
  if (is.list(kw$summary_stats)) kw$summary_stats <- unlist(kw$summary_stats,use.names=FALSE)
  writeLines('cell\tagg_cell','aggregation.tsv')
  writeLines('Peak1\tPeak2\tcoaccess','result.tsv')
  genome_evidence <- NULL
  if (identical(req$function_name,'make_cicero_cds')) {
    reduced <- as.matrix(read.table('coordinates.tsv',header=FALSE))
    rownames(reduced) <- colnames(cds)
    f <- cicero::make_cicero_cds
    result <- do.call(f,c(list(cds=cds,reduced_coordinates=reduced),kw))
    native_type <- class(result)
    if (is.list(result)) {
      write.table(result[[2]],'aggregation.tsv',sep='\t',quote=FALSE,row.names=FALSE)
      result <- result[[1]]
    }
    stopifnot(is(result,'CellDataSet'),identical(rownames(result),rownames(cds)))
    cds <- result
  } else {
    f <- switch(req$function_name,run_cicero=cicero::run_cicero,
      estimate_distance_parameter=cicero::estimate_distance_parameter,stop('Unsupported native CDS function'))
    inputs <- list(cds=cds)
    if (file.exists('genome.tsv')) {
      inputs$genomic_coords <- read.table('genome.tsv',header=FALSE,stringsAsFactors=FALSE,quote='',comment.char='')
      genome_evidence <- list(source='supplied_file')
    } else if (identical(req$function_name,'estimate_distance_parameter')) {
      # Omit the argument: the installed function owns its bundled default.
      genome_evidence <- list(source='native_default',object='cicero::human.hg19.genome',
        assembly='hg19',species='human',input_reference_qualified=FALSE,
        chromosomes=cicero::human.hg19.genome,
        sha256=digest::digest(jsonlite::toJSON(cicero::human.hg19.genome,digits=NA),algo='sha256',serialize=FALSE))
    }
    result <- do.call(f,c(inputs,kw))
    native_type <- class(result)
    if (identical(req$function_name,'estimate_distance_parameter')) result <- data.frame(distance_parameter=unlist(result))
    for (key in names(result)) if (is.numeric(result[[key]])) result[[key]] <- sprintf('%.17g',result[[key]])
    write.table(result,'result.tsv',sep='\t',quote=FALSE,row.names=FALSE,na='NA')
  }
  writeMM(as(exprs(cds),'CsparseMatrix'),'out-exprs.mtx')
  jsonlite::write_json(list(format='cicero-cds-exchange.v1',storage=if(is.matrix(exprs(cds))) 'dense' else 'sparse',observations=from_frame(pData(cds)),regions=from_frame(fData(cds))),
    'out-object.json',auto_unbox=TRUE,digits=NA,na='null')
  versions <- vapply(c('cicero','monocle','Matrix','FNN','VGAM','Biobase'),function(p)as.character(packageVersion(p)),'')
  jsonlite::write_json(list(completion='succeeded',function_name=req$function_name,
    request_sha256=digest::digest(file=request_path,algo='sha256'),versions=as.list(versions),
    genomic_coords=genome_evidence,
    native_signature=paste(capture.output(args(f)),collapse=' '),native_output_type=paste(native_type,collapse=','),
    n_observations=ncol(cds),n_regions=nrow(cds),R=R.version.string,session=capture.output(sessionInfo())),
    'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
},error=function(e){
  jsonlite::write_json(list(message=conditionMessage(e)),'native-error.json',auto_unbox=TRUE)
  stop(e)
})
