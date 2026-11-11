# Data-only public R exports. No parse/eval, user code, or arbitrary namespace.
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
metadata <- function(package, name) {
  fn <- getExportedValue(package, name)
  effective <- fn
  method <- tryCatch(getS3method(name, 'default', optional=TRUE, envir=asNamespace(package)), error=function(e) NULL)
  method_label <- 'default method'
  if(is.null(method) && methods::isGeneric(name,where=asNamespace(package))) {
    method <- tryCatch(methods::selectMethod(fn,signature='Matrix',optional=TRUE),error=function(e) NULL)
    method_label <- 'Matrix method'
  }
  if (!is.null(method)) effective <- method
  formal <- formals(effective)
  # Installed S4 methods often expose (object, ...) and delegate to a literal
  # .local function. Inspect its formal pairlist without evaluating its body or
  # expression defaults. The selected export still performs native dispatch.
  expressions <- as.list(body(effective))
  for(expression in expressions) {
    if(is.call(expression) && identical(expression[[1]],as.name('<-')) &&
       identical(expression[[2]],as.name('.local'))) {
      if(is.function(expression[[3]])) formal <- formals(expression[[3]])
      else if(is.call(expression[[3]]) && identical(expression[[3]][[1]],as.name('function'))) formal <- expression[[3]][[2]]
    }
  }
  parameters <- lapply(names(formal), function(key) {
    missing <- identical(formal[key],setNames(alist(a=),key))
    if(missing) return(list(name=key,kind='value',annotation='',required=key!='...',
      parameter_kind=if(key=='...') 'VAR_KEYWORD' else 'POSITIONAL_OR_KEYWORD',
      default_known=FALSE,default=NULL,default_label='Native default'))
    value <- formal[[key]]
    known <- !missing && (is.null(value) || is.atomic(value))
    kind <- if(missing) 'value' else if (is.logical(value)) 'boolean' else if (is.integer(value)) 'integer' else if (is.double(value)) 'number' else if (is.character(value)) 'string' else 'value'
    list(name=key, kind=kind, annotation='', required=missing && key != '...',
      parameter_kind=if (key == '...') 'VAR_KEYWORD' else 'POSITIONAL_OR_KEYWORD',
      default_known=known, default=if(known) value else NULL,
      default_label=if(known) paste(deparse(value),collapse=' ') else 'Native default')
  })
  list(name=name, signature=paste(deparse(args(effective)),collapse=' '), parameters=parameters,
    description=if(is.null(method)) '' else paste('Installed generic; defaults shown for its',method_label), limitation=NULL)
}
if (args[[1]] == '--discover') {
  package <- args[[2]]
  exports <- getNamespaceExports(package)
  names <- sort(exports[vapply(exports, function(n) is.function(getExportedValue(package,n)), logical(1))])
  selected <- if(length(args) > 2 && nzchar(args[[3]])) args[[3]] else NULL
  values <- if(is.null(selected)) lapply(names,function(n)list(name=n)) else lapply(intersect(names,selected),function(n)metadata(package,n))
  encoded <- toJSON(list(package=package,version=as.character(packageVersion(package)),functions=values),auto_unbox=TRUE,null='null')
  if(length(args)>3) writeLines(encoded,args[[4]]) else cat(encoded)
  quit(status=0)
}
request <- fromJSON(args[[1]], simplifyVector=FALSE)
root <- request$stage
write_json <- function(value, name) jsonlite::write_json(value,file.path(root,name),auto_unbox=TRUE,null='null',digits=NA)
executed <- FALSE
tryCatch({
  package <- request$package; name <- request[['function']]
  fn <- getExportedValue(package,name)
  inputs <- lapply(request$inputs,function(binding) {
    if(binding$adapter == 'r_matrix') {
      value <- readMM(binding$path)
      if(!is.null(binding$row_names)) rownames(value) <- unlist(binding$row_names)
      if(!is.null(binding$column_names)) colnames(value) <- unlist(binding$column_names)
      return(value)
    }
    if(binding$adapter == 'r_dataframe') return(read.table(binding$path,sep=binding$separator,header=TRUE,check.names=FALSE,stringsAsFactors=FALSE))
    if(binding$adapter == 'r_accepted') return(readRDS(binding$path))
    if(binding$adapter == 'file') return(binding$path)
    if(binding$adapter == 'value') return(if(is.list(binding$value)) unlist(binding$value,recursive=FALSE) else binding$value)
    stop(paste('Unsupported R input adapter:',binding$adapter))
  })
  parameters <- lapply(request$parameters,function(value) if(is.list(value) && is.null(names(value))) unlist(value,recursive=FALSE) else value)
  for(key in names(request$outputs)) parameters[key] <- list(file.path(root,'work',request$outputs[[key]]))
  setwd(file.path(root,'work'))
  before <- tools::md5sum(list.files('.',recursive=TRUE,full.names=TRUE,all.files=TRUE,no..=TRUE))
  value <- do.call(fn,c(inputs,parameters))
  executed <- TRUE
  receipt <- list(native_execution='succeeded',package=package,version=as.character(packageVersion(package)),function_name=name,
    supplied_parameters=request$parameters,native_output_type=paste(class(value),collapse='/'))
  write_json(receipt,'execution.json')
  retain <- function(value, name) {
    supported <- is.null(value) || is.atomic(value) || is.data.frame(value) || inherits(value,'sparseMatrix') || (is.list(value) && all(vapply(value,function(x) is.null(x)||is.atomic(x)||is.data.frame(x)||inherits(x,'sparseMatrix'),logical(1))))
    if(!supported) stop(paste('Native execution succeeded; faithful output adapter unavailable for',paste(class(value),collapse='/')))
    if(is.null(value)) return(list(type='R.NULL',encoding='json',value=NULL))
    file <- paste0(name,'.rds');saveRDS(value,file.path(root,'objects',file),version=3)
    dimensions <- dim(value)
    list(type=paste0('R.',paste(class(value),collapse='/')),encoding='r',file=file,
      shape=if(is.null(dimensions)) list(length(value)) else as.list(dimensions))
  }
  representation_error <- NULL
  returned <- tryCatch(retain(value,'return'), error=function(e) {
    representation_error <<- conditionMessage(e); NULL
  })
  produced <- list()
  after <- tools::md5sum(list.files('.',recursive=TRUE,full.names=TRUE,all.files=TRUE,no..=TRUE))
  changed <- names(after)[!names(after) %in% names(before) | is.na(before[names(after)]) | after != before[names(after)]]
  for(path in changed) {
    if(nzchar(Sys.readlink(path))) stop('Native-produced symbolic links are not portable outputs')
    file <- paste0('file-',length(produced),'.',tools::file_ext(path))
    file.copy(path,file.path(root,'objects',file)); produced[[length(produced)+1]] <- list(name=path,file=file)
  }
  result <- list(return_value=returned,modified_inputs=structure(list(),names=character()),produced_files=produced)
  if(!is.null(representation_error)) {
    write_json(result,'unpublished.json'); stop(representation_error)
  }
  write_json(result,'result.json')
  if(is.atomic(value) && is.null(dim(value))) returned$preview <- as.list(head(value,12))
  if(is.data.frame(value)) returned$preview <- value[seq_len(min(nrow(value),8)),seq_len(min(ncol(value),8)),drop=FALSE]
  write_json(list(return_value=returned,modified_inputs=structure(list(),names=character()),produced_files=produced),'preview.json')
  receipt$result_representation <- 'succeeded';write_json(receipt,'execution.json')
}, error=function(e) {
  write_json(list(message=conditionMessage(e),type=class(e)[[1]],native_execution=if(executed)'succeeded' else 'failed',result_representation=if(executed)'failed' else 'not_started'),'error.json')
  quit(status=1)
})
