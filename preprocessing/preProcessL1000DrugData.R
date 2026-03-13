library(tidyverse)
library(caret)
library(jsonlite)

## Load and prepare data---------------------------------------------------------
## Load sigInfo (metadata) to:
## A) FOR EVERY CELL LINE PAIR: 
## split into:
## unpaired conditions 
## paired conditions
## B) FOR EVERY CELL LINE TRIPLET: 
## split into:
## unpaired conditions 
## paired conditions
## C) FOR EVERY CELL LINE QUADRUPLET: 
## split into:
## unpaired conditions 
## paired conditions
### MAKE THESE 3 LISTS OF DATAFRAMES
# Function to split data into paired and unpaired conditions

# Function to split data into paired and unpaired conditions
split_paired_unpaired <- function(data, group_size) {
  # Get unique cell lines
  cell_lines <- unique(data$cell_iname)
  
  # Initialize lists to store results
  paired_list <- list()
  unpaired_list <- list()
  
  # Iterate over each combination of cell lines
  combinations <- combn(cell_lines, group_size, simplify = FALSE)
  
  for (comb in combinations) {
    # Subset data for the current combination of cell lines
    subset_data <- data %>% filter(cell_iname %in% comb)
    
    # Find paired conditions (same conditionID across all cell lines, including duplicates)
    paired_conditions <- subset_data %>%
      group_by(conditionID) %>%
      filter(n_distinct(cell_iname) == group_size) %>%  # Ensure conditionID appears in all cell lines
      ungroup()
    
    # Store paired conditions only if there are observations
    if (nrow(paired_conditions) > 0) {
      paired_list[[paste(comb, collapse = "_")]] <- paired_conditions
    }
    
    # Unpaired conditions are those that don't meet the pairing criteria
    unpaired_conditions <- subset_data %>%
      anti_join(paired_conditions, by = c("cell_iname", "conditionID"))
    
    # Store unpaired conditions only if there are observations
    if (nrow(unpaired_conditions) > 0) {
      unpaired_list[[paste(comb, collapse = "_")]] <- unpaired_conditions
    }
  }
  
  return(list(paired = paired_list, unpaired = unpaired_list))
}
# load
X <- readRDS('../../BechmarkDatasetTranslation/preprocessing/preprocessed_data/cmap_drugs_filtered.rds')
geneInfo <- read.delim('../../BechmarkDatasetTranslation/data/geneinfo_beta.txt') %>% filter(feature_space=='landmark')
sigInfo <- read.delim('../../BechmarkDatasetTranslation/preprocessing/preprocessed_data/metadata.txt')
## keep cell lines with at least 100 samples each
sigInfo <- sigInfo %>% filter(pert_type=="trt_cp") %>%
  group_by(cell_iname) %>% filter(n_distinct(sig_id)>=100) %>% ungroup()
sigInfo <- sigInfo %>% select(sig_id,conditionID,cmap_name,cell_iname,duplIdentifier,pert_idose,pert_itime) %>% 
  unique()
X <- X[geneInfo$gene_symbol,]
X <- X[,sigInfo$sig_id]
X <- t(X)
gc()

# make list of pairs
pairs_results <- split_paired_unpaired(sigInfo, group_size = 2)
paired_pairs <- pairs_results$paired
unpaired_pairs <- pairs_results$unpaired
saveRDS(pairs_results,'preprocessed_data/drug_pairs_dataset.rds')

# make a list of triplets
triplets_results <- split_paired_unpaired(sigInfo, group_size = 3)
paired_triplets <- triplets_results$paired
unpaired_triplets <- triplets_results$unpaired
saveRDS(triplets_results,'preprocessed_data/drug_triplets_dataset.rds')

# make a list of quadruplets
quadruplets_results <- split_paired_unpaired(sigInfo, group_size = 4)
paired_quadruplets <- quadruplets_results$paired
unpaired_quadruplets <- quadruplets_results$unpaired
saveRDS(quadruplets_results,'preprocessed_data/drug_quadruplets_dataset.rds')

## Keep the most difficult (based on cosine similarity) possible test set with a 20% / 80% split----------------
## Make sure that the 80% / 20% is performed per modality level

# Function to find the 20% most dissimilar samples
find_most_dissimilar_samples <- function(sample_ids, X, test_size = 0.2) {
  # Subset the gene expression matrix for the given samples
  X_subset <- X[sample_ids, ]
  
  # Calculate the correlation matrix
  cor_matrix <- cor(t(X_subset))  # Correlation between samples
  # Ignore the diagonal (set to NA)
  diag(cor_matrix) <- NA
  
  # Calculate the maximum correlation for each sample (ignoring the diagonal)
  max_correlation <- apply(abs(cor_matrix), 1, max, na.rm = TRUE)
  
  # Identify the 20% most dissimilar samples (lowest average correlation)
  n_test <- ceiling(test_size * length(sample_ids))
  test_samples <- names(sort(max_correlation)[1:n_test])
  
  return(test_samples)
}

# Function to create test sets with separate selection for paired and unpaired data
create_test_sets <- function(paired_list, unpaired_list, X) {
  # Use lapply to iterate over combinations
  test_sets <- lapply(names(unpaired_list), function(comb) {
    # Get paired and unpaired samples for the current combination
    paired_samples <- if (comb %in% names(paired_list)) paired_list[[comb]] else data.frame()
    unpaired_samples <- unpaired_list[[comb]]
    
    # Initialize list to store paired and unpaired test samples
    test_samples <- list(paired = c(), unpaired = c())
    
    # Select test samples for paired data (if available)
    if (nrow(paired_samples) >= 10) {
      paired_sample_ids <- paired_samples$sig_id
      test_samples$paired <- find_most_dissimilar_samples(paired_sample_ids, X)
    }
    
    # Select test samples for unpaired data (if available)
    if (nrow(unpaired_samples) > 0) {
      unpaired_sample_ids <- unpaired_samples$sig_id
      test_samples$unpaired <- find_most_dissimilar_samples(unpaired_sample_ids, X)
    }
    
    # Return the test samples for the current combination
    return(test_samples)
  })
  
  # Name the list elements with the combinations
  names(test_sets) <- names(unpaired_list)
  
  return(test_sets)
}

# Create test sets for pairs
test_sets_pairs <- create_test_sets(paired_pairs, unpaired_pairs, X)
saveRDS(test_sets_pairs,'preprocessed_data/drug_test_sets_pairs.rds')
gc()

# repeat for triplets
test_sets_triplets <- create_test_sets(paired_triplets, unpaired_triplets, X)
saveRDS(test_sets_triplets,'preprocessed_data/drug_test_sets_triplets.rds')
gc()

# repeat for quadruplets
test_sets_quadruplets <- create_test_sets(paired_quadruplets, unpaired_quadruplets, X)
saveRDS(test_sets_quadruplets,'preprocessed_data/drug_test_sets_quadruplets.rds')
gc()

## Random 10-fold cross-validation------------------------------------

# Function to create tuning sets and perform 10-fold cross-validation
create_tuning_and_cv_splits <- function(paired_list, unpaired_list, test_sets) {
  # Initialize list to store tuning sets and CV splits
  tuning_and_cv_splits <- list()
  
  # Iterate over each combination
  for (comb in names(test_sets)) {
    # Get paired and unpaired samples for the current combination
    paired_samples <- if (comb %in% names(paired_list)) paired_list[[comb]] else data.frame()
    unpaired_samples <- unpaired_list[[comb]]
    
    # Get test samples for the current combination
    test_samples <- test_sets[[comb]]
    
    # Combine paired and unpaired samples
    all_samples <- bind_rows(paired_samples, unpaired_samples)
    
    # Remove test samples to create the tuning set
    tuning_samples <- all_samples %>% filter(!sig_id %in% c(test_samples$paired, test_samples$unpaired))
    
    # Perform 10-fold cross-validation on the tuning set
    set.seed(123)  # For reproducibility
    folds <- createFolds(tuning_samples$pert_type, k = 10)  # Stratified by pert_type
    
    # Store only sig_ids of training and validation sets
    cv_folds <- lapply(folds, function(val_idx) {
      list(
        validation = tuning_samples$sig_id[val_idx],
        training = tuning_samples$sig_id[-val_idx]
      )
    })
    
    # Store the CV splits
    tuning_and_cv_splits[[comb]] <- cv_folds
  }
  
  return(tuning_and_cv_splits)
}

# Create tuning sets and CV splits for pairs
tuning_and_cv_pairs <- create_tuning_and_cv_splits(paired_pairs, unpaired_pairs, test_sets_pairs)
write_json(tuning_and_cv_pairs, 'preprocessed_data/drug_tuning_and_cv_pairs.json', auto_unbox = TRUE, pretty = TRUE)
gc()

# Create tuning sets and CV splits for triplets
tuning_and_cv_triplets <- create_tuning_and_cv_splits(paired_triplets, unpaired_triplets, test_sets_triplets)
write_json(tuning_and_cv_triplets, 'preprocessed_data/drug_tuning_and_cv_triplets.json', auto_unbox = TRUE, pretty = TRUE)
gc()

# Create tuning sets and CV splits for quadruplets
tuning_and_cv_quadruplets <- create_tuning_and_cv_splits(paired_quadruplets, unpaired_quadruplets, test_sets_quadruplets)
write_json(tuning_and_cv_quadruplets[1:500], "preprocessed_data/drug_tuning_and_cv_quadruplets_part1.json", auto_unbox = TRUE, pretty = TRUE)
write_json(tuning_and_cv_quadruplets[501:1000], "preprocessed_data/drug_tuning_and_cv_quadruplets_part2.json", auto_unbox = TRUE, pretty = TRUE)
write_json(tuning_and_cv_quadruplets[1001:1500], "preprocessed_data/drug_tuning_and_cv_quadruplets_part3.json", auto_unbox = TRUE, pretty = TRUE)
write_json(tuning_and_cv_quadruplets[1501:2000], "preprocessed_data/drug_tuning_and_cv_quadruplets_part4.json", auto_unbox = TRUE, pretty = TRUE)

write_json(tuning_and_cv_quadruplets[2001:2380], "preprocessed_data/drug_tuning_and_cv_quadruplets_part5.json", auto_unbox = TRUE, pretty = TRUE)
gc()
