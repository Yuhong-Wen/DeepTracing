library(Seurat)
library(SeuratObject)
library(Matrix)

base_dir <- "./data/Tumor_Immune_Evasion_eTRACER"

rds_file <- file.path(
    base_dir,
    "scRNA/sc_UMI_count.rds"
)

cat("Reading Seurat object...\n")

obj <- readRDS(rds_file)

cat("Object class:\n")
print(class(obj))

cat("Available assays:\n")
print(Assays(obj))

cat("Default assay:\n")
print(DefaultAssay(obj))


DefaultAssay(obj) <- "RNA"

cat("Extracting RNA counts...\n")

counts <- tryCatch(
    {
        GetAssayData(
            obj,
            assay = "RNA",
            layer = "counts"
        )
    },
    error = function(e) {
        GetAssayData(
            obj,
            assay = "RNA",
            slot = "counts"
        )
    }
)

cat("Counts class:\n")
print(class(counts))

cat("Counts dimensions:\n")
print(dim(counts))

cat("First genes:\n")
print(head(rownames(counts)))

cat("First cells:\n")
print(head(colnames(counts)))


if (!inherits(counts, "dgCMatrix")) {
    counts <- as(counts, "dgCMatrix")
}

cat("Writing matrix...\n")

writeMM(
    counts,
    file.path(
        base_dir,
        "scRNA/sc_UMI_count.mtx"
    )
)

write.table(
    rownames(counts),
    file = file.path(
        base_dir,
        "scRNA/genes.tsv"
    ),
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
)


write.table(
    colnames(counts),
    file = file.path(
        base_dir,
        "scRNA/cells.tsv"
    ),
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
)


cat("\nExport finished.\n")

cat(
    "Matrix:",
    nrow(counts),
    "genes x",
    ncol(counts),
    "cells\n"
)