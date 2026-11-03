# Native functions only. Trusted snapshots are admitted by Python before loading.
suppressPackageStartupMessages({library(ArchR); library(SummarizedExperiment); library(Matrix)})
stopifnot(as.character(packageVersion('ArchR')) == '1.0.3')
request_path <- commandArgs(TRUE)[1]
req <- jsonlite::fromJSON(request_path, simplifyVector=FALSE)
setwd(req$stage)
kw <- req$kwargs
for (key in c('promoterRegion','excludeChr','useSeqnames')) if (is.list(kw[[key]])) kw[[key]] <- unlist(kw[[key]])
tryCatch({
 if (getArchRThreads() > 16 && is.null(kw$threads)) stop('Native default threads exceeds execution bound 16; explicitly select threads')
 if (req$function_name == 'createArrowFiles') {
   addArchRGenome('hg38', install=FALSE)
   paths <- paste0('input-', seq_len(req$n_inputs)-1L, '.tsv.gz')
   a <- do.call(ArchR::createArrowFiles, c(list(inputFiles=paths, sampleNames=unlist(req$sample_names), outputNames=paste0('sample-',seq_along(paths))), kw))
   # ArchR may catch an internal error and return incomplete/empty paths.
   stopifnot(length(a)==length(paths), all(file.exists(a)), all(nzchar(a)))
   cells <- unlist(lapply(a, function(f) paste0(rhdf5::h5read(f,'Metadata/Sample'), '#',
     rhdf5::h5read(f,'Metadata/CellNames'))), use.names=FALSE)
   # getAvailableMatrices accepts a Project only. Arrow creation can produce
   # exactly these two native matrix groups; inspect their actual HDF5 presence.
   matrices <- Reduce(intersect, lapply(a, function(f) {
     h <- rhdf5::h5ls(f, recursive=FALSE)
     intersect(c('TileMatrix','GeneScoreMatrix'), h$name[h$otype=='H5I_GROUP'])
   }))
   dir.create('published'); stopifnot(all(file.copy(a, 'published')))
   object_type <- 'ArrowFiles'
 } else if (req$function_name == 'ArchRProject') {
   addArchRGenome('hg38', install=FALSE)
   a <- sort(list.files('snapshot', pattern='\\.arrow$', full.names=TRUE))
   stopifnot(length(a)==length(req$sample_names))
   p <- do.call(ArchR::ArchRProject, c(list(ArrowFiles=a, outputDirectory='project', copyArrows=TRUE), kw))
   saveArchRProject(p, outputDirectory='published', load=FALSE, threads=1)
   cells <- getCellNames(p); matrices <- getAvailableMatrices(p); object_type <- 'ArchRProject'
 } else {
   p <- loadArchRProject('snapshot', force=FALSE, showLogo=FALSE)
   cells <- getCellNames(p); matrices <- getAvailableMatrices(p); object_type <- 'ArchRProject'
   if (req$function_name == 'getMatrixFromProject') {
     m <- do.call(ArchR::getMatrixFromProject, c(list(ArchRProj=p), kw))
     stopifnot(inherits(m,'SummarizedExperiment'), identical(colnames(m), cells))
     writeMM(assay(m), 'matrix.mtx'); saveRDS(m, 'native-result.rds')
     features <- as.data.frame(rowData(m)); features <- cbind(native_row_index=seq_len(nrow(m)), features)
     write.table(features, 'features.tsv', sep='\t', quote=FALSE, row.names=FALSE)
   } else if (req$function_name != 'getAvailableMatrices') stop('Unsupported native ArchR function')
 }
 tab <- if (req$function_name == 'getAvailableMatrices') data.frame(matrix=matrices) else data.frame(cell_id=cells)
 write.table(tab, 'result.tsv', sep='\t', quote=FALSE, row.names=FALSE)
 versions <- vapply(c('ArchR','rhdf5','SummarizedExperiment','Matrix'), function(p)as.character(packageVersion(p)), '')
 jsonlite::write_json(list(completion='succeeded', function_name=req$function_name,
   request_sha256=digest::digest(file=request_path,algo='sha256'), versions=as.list(versions),
   object_type=object_type,n_cells=length(cells),available_matrices=I(matrices),
   matrix_dimensions=if (exists('m')) dim(m) else NULL, session=capture.output(sessionInfo())),
   'native-receipt.json',auto_unbox=TRUE,pretty=TRUE,digits=NA)
}, error=function(e) {jsonlite::write_json(list(message=conditionMessage(e)), 'native-error.json', auto_unbox=TRUE); stop(e)})
