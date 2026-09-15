# Repository-controlled sparse IO adapter; scientific functions remain external and hash-pinned.
suppressPackageStartupMessages({library(Matrix);library(presto);library(dplyr)})
a <- commandArgs(trailingOnly=TRUE); stopifnot(length(a)==2); root <- a[[1]]; upstream <- a[[2]]
stopifnot(as.character(getRversion())=='4.6.1')
stopifnot(as.character(packageVersion('presto'))=='1.1.0', as.character(packageVersion('Matrix'))=='1.7.6', as.character(packageVersion('dplyr'))=='1.2.1')
p <- function(x) file.path(root,x)
# One tiny test checks Presto direction, native bounds and tied zero signal.
tiny <- Matrix(rbind(c(0,0,0,1,2,3),c(0,0,0,0,0,0)),sparse=TRUE)
rownames(tiny)<-c('up','zero'); z<-presto::wilcoxauc(tiny,c('a','a','a','b','b','b'))
stopifnot(z$auc[z$feature=='up' & z$group=='b']==1,all(z$pval>=0 & z$pval<=1),all(z$auc[z$feature=='zero']==.5))
X <- as(readMM(p('rp-genes-by-cells.mtx')),'CsparseMatrix')
rownames(X)<-readLines(p('genes.tsv')); colnames(X)<-readLines(p('cells.tsv'))
groups<-read.delim(p('groups.tsv'),colClasses='character',check.names=FALSE,na.strings=NULL,quote='',comment.char='')
stopifnot(identical(colnames(X),groups$cell_id),!anyNA(groups$group),all(nzchar(groups$group)),length(unique(groups$group))>1)
y<-groups$group
stopifnot(all(is.finite(X@x)),all(X@x>=0),all(colSums(X)>0))
# Sparse algebra implementing the adopted Seurat LogNormalize definition; no scaling/imputation.
X <- X %*% Diagonal(x=10000/colSums(X)); X@x<-log1p(X@x)
stopifnot(all(is.finite(X@x)),max(abs(colSums(expm1(X))-10000))<1e-7)
e<-new.env(parent=globalenv());sys.source(file.path(upstream,'R/FindAllMarkersMAESTRO.R'),envir=e)
# Only Seurat data access is adapted. Differential/statistical function bodies unchanged.
e$GetAssayData<-function(object,slot) {stopifnot(slot=='data');object}
e$Idents<-function(object)y
e$wilcoxauc<-function(X,y){v<-presto::wilcoxauc(X,y,nthreads=1);e$native<-v;v}
f<-e$FindAllMarkersPresto(X,min.pct=.1,logfc.threshold=.25,only.pos=FALSE,return.thresh=.01,slot='data')
v<-e$native
stopifnot(nrow(v)==nrow(X)*length(unique(y)),!anyDuplicated(v[,c('feature','group')]), setequal(v$feature,rownames(X)), setequal(v$group,y))
for(k in c('auc','pval','padj'))stopifnot(all(is.finite(v[[k]])),all(v[[k]]>=0 & v[[k]]<=1))
stopifnot(all(is.finite(v$logFC)),all(v$pct_in>=0 & v$pct_in<=100),all(v$pct_out>=0 & v$pct_out<=100))
write.table(v,gzfile(p('markers-native.tsv.gz')),sep='\t',row.names=FALSE,quote=FALSE)
write.table(f,p('markers-maestro-filtered.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
f<-f[f$p_val_adj<1e-5,];write.table(f,p('markers-signature-input.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(data.frame(group=names(table(y)),n_cells=as.integer(table(y))),p('group-sizes.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
writeLines(c(capture.output(sessionInfo()),paste('native_rows',nrow(v)),paste('signature_rows',nrow(f)),'tiny_Presto_check=passed','exact_cell_order=passed','native_numeric_checks=passed'),p('marker-runtime.txt'))
cat('Native rows:',nrow(v),'Signature input rows:',nrow(f),'\n');print(table(f$cluster))
