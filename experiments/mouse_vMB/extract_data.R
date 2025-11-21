file_path <- ".../GSE210139_single_cell_CREST_barcode_qc_updated20230422.rds.gz"
con <- gzcon(file(file_path, "rb"))
data <- readRDS(con)
str(data)
names(data)
sub_data <- data$E11_rep1
names(sub_data)

patterns <- sub_data$pattern.v2
pattern_split <- strsplit(patterns, "_")

max_len <- max(sapply(pattern_split, length))
pattern_mat <- t(sapply(pattern_split, function(x) {
  length(x) <- max_len
  return(x)
}))

unique_mutations <- unique(as.vector(pattern_mat))
unique_mutations <- setdiff(unique_mutations, c("NONE", NA, "NA", ""))
mutation_map <- setNames(seq_along(unique_mutations), unique_mutations)
mutation_map[["NONE"]] <- 0
mutation_map[["NA"]] <- 0
mutation_map[[""]] <- 0

encoded_mat <- apply(pattern_mat, c(1,2), function(x) {
  if (is.na(x) || x == "NA" || x == "") {
    return(0)
  } else if (x %in% names(mutation_map)) {
    return(mutation_map[[x]])
  } else {
    return(0)
  }
})

write.csv(encoded_mat, file = "/data/DeepTracing/experiments/mouse_vMB/data/E11_rep1_seq_data_mt.csv", row.names = FALSE)

