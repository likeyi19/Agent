#!/usr/bin/env Rscript

# Independent test-only oracle for the frozen M8.2-B edgeR calculation.

arguments <- commandArgs(trailingOnly = TRUE)
stopifnot(length(arguments) == 1L)
directory <- normalizePath(arguments[[1L]], mustWork = TRUE)
counts <- as.matrix(
    read.table(
        file.path(directory, "counts.tsv"),
        header = FALSE,
        sep = "\t",
        check.names = FALSE
    )
)
design <- as.matrix(
    read.table(
        file.path(directory, "design.tsv"),
        header = FALSE,
        sep = "\t",
        check.names = FALSE
    )
)
condition <- scan(
    file.path(directory, "condition.tsv"), what = integer(), quiet = TRUE
)
contrast <- scan(
    file.path(directory, "contrast.tsv"), what = double(), quiet = TRUE
)
suppressPackageStartupMessages(library(edgeR))
options(digits = 17)
group <- factor(condition, levels = c(0L, 1L))
y <- DGEList(counts = counts, group = group)
keep <- filterByExpr(
    y,
    group = group,
    min.count = 10,
    min.total.count = 15,
    large.n = 10,
    min.prop = 0.7
)
y <- y[keep, , keep.lib.sizes = FALSE]
y <- normLibSizes(
    y,
    method = "TMM",
    refColumn = NULL,
    logratioTrim = 0.30,
    sumTrim = 0.05,
    doWeighting = TRUE,
    Acutoff = -1e10
)
fit <- glmQLFit(
    y,
    design = design,
    dispersion = NULL,
    abundance.trend = TRUE,
    robust = TRUE,
    winsor.tail.p = c(0.05, 0.10),
    legacy = FALSE,
    top.proportion = NULL,
    keep.unit.mat = FALSE,
    prior.count = 0.125
)
test <- glmQLFTest(fit, contrast = contrast, poisson.bound = TRUE)
statistics <- cbind(
    test$table[, c("logFC", "logCPM", "F", "PValue")],
    FDR = p.adjust(test$table[, "PValue"], method = "BH")
)
write.table(
    data.frame(
        post_filter_library_size = y$samples$lib.size,
        normalization_factor = y$samples$norm.factors,
        effective_library_size = y$samples$lib.size * y$samples$norm.factors
    ),
    file.path(directory, "sample_results.tsv"),
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
    sep = "\t"
)
write.table(
    cbind(feature_index = which(keep) - 1L, statistics),
    file.path(directory, "feature_results.tsv"),
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
    sep = "\t"
)
write.table(
    as.integer(keep),
    file.path(directory, "filter_mask.tsv"),
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
)
