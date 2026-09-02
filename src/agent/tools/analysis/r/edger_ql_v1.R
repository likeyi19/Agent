#!/usr/bin/env Rscript

# Fixed production backend for Agent Milestone 8.2-B.
# Inputs are machine-generated binary files and a strict two-column manifest.

PROTOCOL_VERSION <- "1"
EXPECTED_VERSIONS <- c(
    R = "4.6.1",
    Bioconductor = "3.23",
    BiocManager = "1.30.27",
    edgeR = "4.10.4",
    limma = "3.68.5",
    locfit = "1.5.9.12",
    statmod = "1.5.2",
    lattice = "0.23.1"
)

write_status <- function(directory, fields) {
    path <- file.path(directory, "backend_status.tsv")
    lines <- vapply(
        names(fields),
        function(name) paste(name, fields[[name]], sep = "\t"),
        character(1)
    )
    writeLines(lines, path, useBytes = TRUE)
}

fail_backend <- function(directory, code, status = 70L) {
    write_status(
        directory,
        c(
            protocol_version = PROTOCOL_VERSION,
            status = "error",
            error_code = code
        )
    )
    quit(save = "no", status = status, runLast = FALSE)
}

package_versions <- function(directory) {
    required <- c(
        "BiocManager", "edgeR", "limma", "locfit", "statmod", "lattice"
    )
    available <- vapply(required, requireNamespace, logical(1), quietly = TRUE)
    if (!available[["edgeR"]]) {
        fail_backend(directory, "EDGER_PACKAGE_UNAVAILABLE", 41L)
    }
    if (!all(available)) {
        fail_backend(directory, "R_PACKAGE_VERSION_INCOMPATIBLE", 42L)
    }
    versions <- c(
        R = paste(R.version$major, R.version$minor, sep = "."),
        Bioconductor = as.character(BiocManager::version()),
        BiocManager = as.character(packageVersion("BiocManager")),
        edgeR = as.character(packageVersion("edgeR")),
        limma = as.character(packageVersion("limma")),
        locfit = as.character(packageVersion("locfit")),
        statmod = as.character(packageVersion("statmod")),
        lattice = as.character(packageVersion("lattice"))
    )
    if (versions[["edgeR"]] != EXPECTED_VERSIONS[["edgeR"]]) {
        fail_backend(directory, "EDGER_VERSION_UNSUPPORTED", 43L)
    }
    if (!identical(unname(versions), unname(EXPECTED_VERSIONS))) {
        fail_backend(directory, "R_PACKAGE_VERSION_INCOMPATIBLE", 44L)
    }
    versions
}

strict_manifest <- function(directory) {
    path <- file.path(directory, "input_manifest.tsv")
    lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
    pieces <- strsplit(lines, "\t", fixed = TRUE)
    if (length(lines) != 8L || any(lengths(pieces) != 2L)) {
        stop("invalid manifest")
    }
    keys <- vapply(pieces, `[[`, character(1), 1L)
    values <- vapply(pieces, `[[`, character(1), 2L)
    expected <- c(
        "protocol_version",
        "preparation_sha256",
        "n_features",
        "n_samples",
        "n_design_columns",
        "design_sha256",
        "contrast_sha256",
        "input_matrix_sha256"
    )
    if (!identical(keys, expected) || anyDuplicated(keys)) {
        stop("invalid manifest")
    }
    result <- as.list(values)
    names(result) <- keys
    if (result$protocol_version != PROTOCOL_VERSION) {
        stop("invalid protocol")
    }
    for (key in c(
        "preparation_sha256", "design_sha256", "contrast_sha256",
        "input_matrix_sha256"
    )) {
        if (!grepl("^[0-9a-f]{64}$", result[[key]])) {
            stop("invalid digest")
        }
    }
    for (key in c("n_features", "n_samples", "n_design_columns")) {
        parsed <- suppressWarnings(as.integer(result[[key]]))
        if (is.na(parsed) || parsed <= 0L || as.character(parsed) != result[[key]]) {
            stop("invalid dimension")
        }
        result[[key]] <- parsed
    }
    result
}

read_doubles <- function(path, count) {
    expected_size <- 8 * count
    observed_size <- file.info(path)$size
    if (is.na(observed_size) || observed_size != expected_size) {
        stop("invalid binary size")
    }
    connection <- file(path, open = "rb")
    on.exit(close(connection))
    values <- readBin(
        connection, what = double(), n = count, size = 8L, endian = "little"
    )
    if (length(values) != count || any(!is.finite(values))) {
        stop("invalid binary values")
    }
    values
}

read_integers <- function(path, count) {
    expected_size <- 4 * count
    observed_size <- file.info(path)$size
    if (is.na(observed_size) || observed_size != expected_size) {
        stop("invalid binary size")
    }
    connection <- file(path, open = "rb")
    on.exit(close(connection))
    values <- readBin(
        connection, what = integer(), n = count, size = 4L, endian = "little"
    )
    if (length(values) != count || any(is.na(values))) {
        stop("invalid binary values")
    }
    values
}

write_doubles <- function(path, values) {
    connection <- file(path, open = "wb")
    on.exit(close(connection))
    writeBin(as.double(values), connection, size = 8L, endian = "little")
}

write_integers <- function(path, values) {
    connection <- file(path, open = "wb")
    on.exit(close(connection))
    writeBin(as.integer(values), connection, size = 4L, endian = "little")
}

probe <- function(directory) {
    versions <- package_versions(directory)
    write_status(
        directory,
        c(
            protocol_version = PROTOCOL_VERSION,
            status = "success",
            mode = "probe",
            r_version = versions[["R"]],
            bioconductor_version = versions[["Bioconductor"]],
            biocmanager_version = versions[["BiocManager"]],
            edger_version = versions[["edgeR"]],
            limma_version = versions[["limma"]],
            locfit_version = versions[["locfit"]],
            statmod_version = versions[["statmod"]],
            lattice_version = versions[["lattice"]]
        )
    )
}

run_analysis <- function(directory) {
    versions <- package_versions(directory)
    manifest <- strict_manifest(directory)
    n_features <- manifest$n_features
    n_samples <- manifest$n_samples
    n_columns <- manifest$n_design_columns
    counts_vector <- read_doubles(
        file.path(directory, "counts.bin"), n_features * n_samples
    )
    if (any(counts_vector < 0) || any(counts_vector != floor(counts_vector))) {
        stop("invalid counts")
    }
    counts <- matrix(counts_vector, nrow = n_features, ncol = n_samples)
    design <- matrix(
        read_doubles(
            file.path(directory, "design.bin"), n_samples * n_columns
        ),
        nrow = n_samples,
        ncol = n_columns,
        byrow = TRUE
    )
    contrast <- read_doubles(file.path(directory, "contrast.bin"), n_columns)
    condition <- read_integers(
        file.path(directory, "condition.bin"), n_samples
    )
    if (!all(condition %in% c(0L, 1L)) || length(unique(condition)) != 2L) {
        stop("invalid condition")
    }
    condition_group <- factor(condition, levels = c(0L, 1L))

    suppressPackageStartupMessages(library(edgeR))
    result <- withCallingHandlers(
        {
            y <- DGEList(counts = counts, group = condition_group)
            keep <- filterByExpr(
                y,
                group = condition_group,
                min.count = 10,
                min.total.count = 15,
                large.n = 10,
                min.prop = 0.7
            )
            if (!any(keep)) {
                fail_backend(directory, "DA_NO_FEATURES_AFTER_FILTER", 51L)
            }
            y <- y[keep, , keep.lib.sizes = FALSE]
            post_filter_library_sizes <- y$samples$lib.size
            if (any(!is.finite(post_filter_library_sizes)) ||
                any(post_filter_library_sizes <= 0)) {
                fail_backend(directory, "DA_FILTERED_LIBRARY_ZERO", 52L)
            }
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
            test <- glmQLFTest(
                fit,
                contrast = contrast,
                poisson.bound = TRUE
            )
            table <- test$table[, c("logFC", "logCPM", "F", "PValue")]
            statistics <- as.matrix(
                cbind(
                    table,
                    FDR = p.adjust(table[, "PValue"], method = "BH")
                )
            )
            list(
                keep = keep,
                post_filter_library_sizes = post_filter_library_sizes,
                normalization_factors = y$samples$norm.factors,
                effective_library_sizes = (
                    y$samples$lib.size * y$samples$norm.factors
                ),
                statistics = statistics
            )
        },
        warning = function(warning) stop("backend warning")
    )
    if (
        any(!is.finite(result$normalization_factors)) ||
        any(result$normalization_factors <= 0) ||
        any(!is.finite(result$effective_library_sizes)) ||
        any(result$effective_library_sizes <= 0) ||
        any(!is.finite(result$statistics)) ||
        any(result$statistics[, "F"] < 0) ||
        any(result$statistics[, "PValue"] < 0) ||
        any(result$statistics[, "PValue"] > 1) ||
        any(result$statistics[, "FDR"] < 0) ||
        any(result$statistics[, "FDR"] > 1)
    ) {
        fail_backend(directory, "DA_NUMERICAL_RESULT_INVALID", 53L)
    }
    write_integers(file.path(directory, "filter_mask.bin"), result$keep)
    write_integers(
        file.path(directory, "tested_indices.bin"), which(result$keep) - 1L
    )
    write_doubles(
        file.path(directory, "post_filter_library_sizes.bin"),
        result$post_filter_library_sizes
    )
    write_doubles(
        file.path(directory, "normalization_factors.bin"),
        result$normalization_factors
    )
    write_doubles(
        file.path(directory, "effective_library_sizes.bin"),
        result$effective_library_sizes
    )
    write_doubles(
        file.path(directory, "statistics.bin"),
        as.double(t(result$statistics))
    )
    write_status(
        directory,
        c(
            protocol_version = PROTOCOL_VERSION,
            status = "success",
            mode = "run",
            n_features = n_features,
            n_samples = n_samples,
            n_tested = sum(result$keep),
            n_filtered = n_features - sum(result$keep),
            r_version = versions[["R"]],
            bioconductor_version = versions[["Bioconductor"]],
            biocmanager_version = versions[["BiocManager"]],
            edger_version = versions[["edgeR"]],
            limma_version = versions[["limma"]],
            locfit_version = versions[["locfit"]],
            statmod_version = versions[["statmod"]],
            lattice_version = versions[["lattice"]]
        )
    )
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 2L || !(arguments[[1L]] %in% c("probe", "run"))) {
    quit(save = "no", status = 64L, runLast = FALSE)
}
directory <- normalizePath(arguments[[2L]], mustWork = TRUE)
tryCatch(
    {
        if (arguments[[1L]] == "probe") {
            probe(directory)
        } else {
            run_analysis(directory)
        }
    },
    error = function(error) {
        fail_backend(directory, "R_BACKEND_EXECUTION_FAILED", 70L)
    }
)
