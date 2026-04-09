library(tidyverse)
library(dplyr)
library(purrr)
library(tidyr)
library(stringr)
library(readr)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(patchwork)
#### Paired ratio results ------------
folder_autotransop <- "pairedPercs"
root_dir <- file.path(
  "../../TranslationalModels/OmicTranslationBenchmark/preprocessing/preprocessed_data/sampledDatasetes",
  folder_autotransop
)
# List files in the folder
files_autotransop <- list.files(root_dir, recursive = FALSE, full.names = TRUE)
files_autotransop <- files_autotransop[
  str_detect(files_autotransop, "landmarks") | str_detect(files_autotransop, "\\.rds$")
]
files_autotransop <- tibble(files = files_autotransop)
# Separate RDS and CSV lists (if you still need CSV elsewhere)
filesRDS_autotransop <- files_autotransop %>%
  filter(str_detect(files, "\\.rds$")) %>%
  distinct()
filesCSV_autotransop <- files_autotransop %>%
  filter(str_detect(files, "\\.csv$")) %>%
  distinct()
paired_ratios_cv_folder <- function(cv_dir, folds = 1:5) {
  
  # small helper to safely count rows of a CSV
  n_csv <- function(path) {
    if (!file.exists(path)) return(NA_integer_)
    # read just enough to count rows (simple + reliable)
    nrow(read_csv(path, show_col_types = FALSE, progress = FALSE))
  }
  
  # fallback: rebuild pairs by joining A & H on conditionId + cmap_name
  # (only used if paired file is missing)
  n_pairs_from_join <- function(path_A, path_H) {
    if (!file.exists(path_A) || !file.exists(path_H)) return(NA_integer_)
    
    a <- read_csv(path_A, show_col_types = FALSE, progress = FALSE) %>%
      select(sig_id, conditionId, cmap_name) %>%
      distinct() %>%
      rename(sig_id.x = sig_id)
    
    h <- read_csv(path_H, show_col_types = FALSE, progress = FALSE) %>%
      select(sig_id, conditionId, cmap_name) %>%
      distinct() %>%
      rename(sig_id.y = sig_id)
    
    nrow(inner_join(a, h, by = c("conditionId", "cmap_name")))
  }
  
  map_dfr(folds, function(k) {
    
    # expected filenames
    train_A <- file.path(cv_dir, sprintf("train_A375_%d.csv", k))
    train_H <- file.path(cv_dir, sprintf("train_HT29_%d.csv", k))
    train_P <- file.path(cv_dir, sprintf("train_paired_%d.csv", k))
    
    val_A   <- file.path(cv_dir, sprintf("val_A375_%d.csv", k))
    val_H   <- file.path(cv_dir, sprintf("val_HT29_%d.csv", k))
    val_P   <- file.path(cv_dir, sprintf("val_paired_%d.csv", k))
    
    # counts
    n_train_A <- n_csv(train_A)
    n_train_H <- n_csv(train_H)
    n_train_P <- n_csv(train_P)
    if (is.na(n_train_P)) {
      n_train_P <- n_pairs_from_join(train_A, train_H)
    }
    
    n_val_A <- n_csv(val_A)
    n_val_H <- n_csv(val_H)
    n_val_P <- n_csv(val_P)
    if (is.na(n_val_P)) {
      n_val_P <- n_pairs_from_join(val_A, val_H)
    }
    
    tot_train <- if (all(!is.na(c(n_train_A, n_train_H)))) n_train_A + n_train_H else NA_integer_
    tot_val   <- if (all(!is.na(c(n_val_A, n_val_H)))) n_val_A + n_val_H else NA_integer_
    
    ratio_train <- if (!is.na(n_train_P) && !is.na(tot_train) && tot_train > 0) {
      2 * n_train_P / tot_train
    } else NA_real_
    
    ratio_val <- if (!is.na(n_val_P) && !is.na(tot_val) && tot_val > 0) {
      2 * n_val_P / tot_val
    } else NA_real_
    
    tibble(
      fold = k,
      split = c("train", "val"),
      n_A375 = c(n_train_A, n_val_A),
      n_HT29 = c(n_train_H, n_val_H),
      n_pairs = c(n_train_P, n_val_P),
      total_samples = c(tot_train, tot_val),
      paired_ratio = c(ratio_train, ratio_val),
      paired_percent = 100 * c(ratio_train, ratio_val)
    )
  })
}
# Link each RDS to its CV folder (same basename without extension)
cv_results <- filesRDS_autotransop %>%
  mutate(
    rds_basename = tools::file_path_sans_ext(basename(files)),
    cv_folder = file.path(dirname(files), rds_basename)
  ) %>%
  filter(dir.exists(cv_folder)) %>%
  mutate(cv_summary = map(cv_folder, paired_ratios_cv_folder)) %>%
  select(rds_file = files, rds_basename, cv_folder, cv_summary) %>%
  unnest(cv_summary)
cv_results <- cv_results %>%
  select(rds_basename, fold, split, total_samples, n_pairs, paired_percent)

cv_results <- cv_results %>%
  group_by(rds_basename, split) %>%
  summarise(
    paired_percent_mu = mean(paired_percent, na.rm = TRUE),
    paired_percent_se = sd(paired_percent, na.rm = TRUE) / sqrt(n()),
    number_samples = mean(total_samples, na.rm = TRUE),
    number_paired_samples = round(mean(n_pairs, na.rm = TRUE)),  # <-- keep paired count
    .groups = "drop"
  )

## Load AutoTransOP performance----------
files_autotransop <- files_autotransop %>% 
  mutate(csv=grepl('.csv',files)) %>% 
  mutate(rds=grepl('.rds',files)) %>% 
  filter(csv==T) %>% select(-rds,-csv) %>% unique()
results_autotransop <- data.frame()
for (j in 1:nrow(files_autotransop)){
  data <- data.table::fread(files_autotransop$files[j])
  rownames(data) <- NULL
  colnames(data)[1] <- 'fold'
  data <- data %>% gather('metric','value',-fold,-Direct_pearson)
  data <- data %>% mutate(metric=ifelse(metric=='model_pearsonHT29' | metric=='model_pearsonA375','pearson translation',
                                        ifelse(metric=='model_spearHT29' | metric=='model_spearA375','spearman translation',
                                               ifelse(metric=='model_accHT29' | metric=='model_accA375','sign accuracy translation',
                                                      ifelse(metric=='recon_pear_ht29' | metric=='recon_pear_a375','pearson reconstruction',
                                                             ifelse(metric=='recon_spear_ht29' | metric=='recon_spear_a375','spearman reconstruction',
                                                                    ifelse(metric=='F1_score','F1 score',
                                                                           ifelse(metric=='ClassAccuracy','Accuracy','other'))))))))
  data <- data %>% filter(metric == 'pearson translation')
  data <- data %>% group_by(fold,metric) %>% mutate(value=mean(value)) %>% ungroup() %>% unique()
  rds_basename <- sub("_landmarks\\.csv$","",basename(files_autotransop$files[j]))
  data <- data %>% mutate(rds_basename=rds_basename)
  results_autotransop <- rbind(results_autotransop,data)
}
results_autotransop <- left_join(results_autotransop,
                                 cv_results %>% filter(split=='train') %>% select(-split))
colnames(results_autotransop)[2] <- 'direct_translation'
results_autotransop <- results_autotransop %>% group_by(metric,rds_basename) %>%
  mutate(mean_value=mean(value)) %>% 
  mutate(std_value=sd(value)) %>% 
  # mutate(direct_translation=mean(direct_translation)) %>%
  ungroup()

## sanity check
dt_mean <- mean(results_autotransop$direct_translation, na.rm = TRUE)
dt_ci <- 1.96 * sd(results_autotransop$direct_translation, na.rm = TRUE) / sqrt(30)
p <- ggplot(results_autotransop %>% filter(metric=='pearson translation'),
            aes(paired_percent_mu,mean_value)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5)),
              linetype=0,alpha=0.2,fill = "grey70") +
  geom_line(size=0.8,alpha=0.8) + geom_point(color='black',size=2) +
  xlab('Percentage of paired conditions (%)') + ylab('Average pearson`s r for translation')+
  ylim(c(0,1))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        legend.text = element_text("Arial",size = 20),
        legend.title = element_text("Arial",size = 20),
        panel.grid.major = element_line(linewidth=1.5))+
  geom_hline(yintercept= mean(results_autotransop$direct_translation),linetype=2,color='red')+
  annotate("rect",
           xmin = -Inf, xmax = Inf,
           ymin = dt_mean - dt_ci,
           ymax = dt_mean + dt_ci,
           alpha = 0.1, fill = "red")
print(p)

## Load FlowTransOP performance (FlowMatch) ----------
get_rds_basename <- function(f) {
  bn <- basename(f)
  out <- str_extract(bn, "^sample_ratio_\\d+")
  if (is.na(out)) out <- sub("_flow.*$", "", bn)
  out
}

load_generalised_test_only <- function(flow_dir, method_label) {
  
  files <- list.files(flow_dir, full.names = TRUE, recursive = FALSE) %>%
    .[str_detect(basename(.), "GeneralizedTransOP")] %>%
    .[str_detect(basename(.), "translation")] %>%
    .[str_detect(basename(.), "\\.csv$")]
  
  if (length(files) == 0) {
    warning(sprintf("No FlowTransOP translation CSVs found in: %s", flow_dir))
    return(tibble())
  }
  
  out <- map_dfr(files, function(f) {
    df <- read_csv(f, show_col_types = FALSE)
    
    # Expected columns: train, test, fold, folder
    # If fold not explicitly named, assume first col is fold
    if (!"fold" %in% colnames(df)) colnames(df)[1] <- "fold"
    
    rds_basename <- get_rds_basename(f)
    
    df %>%
      mutate(
        rds_basename = rds_basename,
        metric = "pearson translation",
        source_file = basename(f)
      )
  }) %>%
    # join paired % (use TRAIN split pairing summary, as in AutoTransOP)
    left_join(
      cv_results %>% filter(split == "train") %>% select(-split),
      by = "rds_basename"
    ) %>%
    # compute per-dataset mean/sd for TEST
    group_by(metric, rds_basename) %>%
    mutate(
      mean_value = mean(test, na.rm = TRUE),
      std_value  = sd(test, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(
      value = test,
      direct_translation = NA_real_,
      method = method_label
    ) %>%
    # keep test-only columns
    select(
      method,
      fold,
      direct_translation,
      metric,
      value,
      rds_basename,
      paired_percent_mu,
      paired_percent_se,
      number_samples,
      number_paired_samples,
      folder,
      mean_value,
      std_value,
      source_file
    )
  
  out
}

## Load both FlowTransOP variants

results_generalised_unpaired <- load_generalised_test_only(
  flow_dir = "../results/FlowMatch_lowPairsPercentage",
  method_label = "FlowTransOP (unpaired)"
)

results_generalised_paired <- load_generalised_test_only(
  flow_dir = "../results/FlowMatch_lowPairsPercentage_withPairs",
  method_label = "FlowTransOP (paired)"
)

## Prepare AutoTransOP for merge
common_cols <- c(
  "method",
  "fold",
  "direct_translation",
  "metric",
  "value",
  "rds_basename",
  "paired_percent_mu",
  "paired_percent_se",
  "number_samples",
  "number_paired_samples",
  "mean_value",
  "std_value"
)

results_autotransop_merged <- results_autotransop %>%
  mutate(method = "AutoTransOP") %>%
  select(any_of(common_cols))

results_generalised_unpaired_merged <- results_generalised_unpaired %>%
  select(any_of(common_cols))

results_generalised_paired_merged <- results_generalised_paired %>%
  select(any_of(common_cols))

### Load generalised transop extremely low pairs case
n_avail <- 965.8333 # on average this is the traning data size here
load_extremely_fewpairs_translation <- function(
    file_path,
    total_samples_full,
    method_label = "GeneralizedTransOP (paired)"
) {
  # file has: train, test, fold, n_pairs, repeat
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # 1) summarize repeats -> one number per n_pairs x fold
  df_s <- df %>%
    group_by(n_pairs, fold) %>%
    summarise(
      value = mean(test, na.rm = TRUE),
      .groups = "drop"
    )
  
  # 2) compute paired % using your rule
  #    This matches your earlier definition:
  #    paired_ratio = 2*n_pairs / total_samples
  df_s <- df_s %>%
    mutate(
      paired_percent_mu = 100 * (n_pairs / total_samples_full),
      paired_percent_se = 0,
      number_samples = total_samples_full,
      number_paired_samples = n_pairs,
      metric = "pearson translation",
      direct_translation = NA_real_,
      method = method_label,
      rds_basename = paste0("A375_HT29_extreme_nPairs_", n_pairs)
    )
  
  # 3) compute mean/sd across folds for each n_pairs
  df_s <- df_s %>%
    group_by(metric, rds_basename) %>%
    mutate(
      mean_value = mean(value, na.rm = TRUE),
      std_value  = sd(value, na.rm = TRUE)
    ) %>%
    ungroup()
  
  # 4) return only the columns that match your merge schema
  df_s %>%
    select(
      method,
      fold,
      direct_translation,
      metric,
      value,
      rds_basename,
      paired_percent_mu,
      paired_percent_se,
      number_samples,
      number_paired_samples,
      mean_value,
      std_value
    )
}

extreme_translation_file_withpairs <- "../results/FlowMatch_extremely_fewPairs_A375_HT29_withPairs/A375_HT29_ExtremelyfewPairs_translation.csv"
results_generalised_paired_extreme <- load_extremely_fewpairs_translation(
  file_path = extreme_translation_file_withpairs,
  total_samples_full = n_avail,
  method_label = "FlowTransOP (paired)"
)
# Append to your existing paired FlowTransOP curve
results_generalised_paired_merged_extended <- bind_rows(
  results_generalised_paired_merged,
  results_generalised_paired_extreme
)
# # repeat for unpaired
extreme_translation_file_unpaired <- "../results/FlowMatch_extremely_fewPairs_A375_HT29/A375_HT29_ExtremelyfewPairs_translation.csv"
results_generalised_unpaired_extreme <- load_extremely_fewpairs_translation(
  file_path = extreme_translation_file_unpaired,
  total_samples_full = n_avail,
  method_label = "FlowTransOP (unpaired)"
)
# Append to your existing paired FlowTransOP curve
results_generalised_unpaired_merged_extended <- bind_rows(
  results_generalised_unpaired_merged,
  results_generalised_unpaired_extreme
)

## Final merged dataframe
results_both <- bind_rows(
  results_autotransop_merged,
  results_generalised_unpaired_merged_extended,
  results_generalised_paired_merged_extended
)

## Load FlowTransOP performance with both pairs and similarity ----------
SimPairMax_results_generalised <- load_generalised_test_only(
  flow_dir = "../results/FlowMatch_fewPairs_A375_HT29_PairAndSimilarity",
  method_label = "FlowTransOP (pairs+sim max agg.)"
)
SimPairMean_results_generalised <- load_generalised_test_only(
  flow_dir = "../results/FlowMatch_fewPairs_A375_HT29_PairAndSimilarity_meanAgg",
  method_label = "FlowTransOP (pairs+sim mean agg.)"
)
SimPairSum_results_generalised <- load_generalised_test_only(
  flow_dir = "../results/FlowMatch_fewPairs_A375_HT29_PairAndSimilarity_sumAgg",
  method_label = "FlowTransOP (pairs+sim sum agg.)"
)

## extremely low pairs---
SimPairMax_results_extreme <- load_extremely_fewpairs_translation(
  file_path = "../results/FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity/A375_HT29_ExtremelyfewPairs_translation.csv",
  total_samples_full = n_avail,
  method_label = "FlowTransOP (pairs+sim max agg.)"
)
SimPairSum_results_extreme <- load_extremely_fewpairs_translation(
  file_path = "../results/FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity_sumAgg/A375_HT29_ExtremelyfewPairs_translation.csv",
  total_samples_full = n_avail,
  method_label = "FlowTransOP (pairs+sim sum agg.)"
)

## Merge and visualize-------------
results_all <- bind_rows(results_both,
                         SimPairMax_results_generalised,
                         SimPairMax_results_extreme,
                         SimPairMean_results_generalised,
                         SimPairSum_results_generalised,
                         SimPairSum_results_extreme)

## First compare aggregation methods
pagg <- ggplot(results_all  %>% filter(metric=='pearson translation') %>%
                 filter(method %in% c("FlowTransOP (pairs+sim max agg.)",
                                      "FlowTransOP (pairs+sim mean agg.)",
                                      "FlowTransOP (pairs+sim sum agg.)")) %>%
                 filter(number_paired_samples<50),
               aes(number_paired_samples,mean_value,
                   color = method, group = method)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5),
                  fill = method),
              linetype=0,alpha=0.2) +
  geom_line(size=0.8,alpha=0.8) + geom_point(size=2) +
  xlab('number of paired samples') + ylab('Average pearson`s r for translation')+
  ylim(c(0,1))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        legend.text = element_text("Arial",size = 14),
        legend.title = element_blank(),
        legend.position = 'top',
        panel.grid.major = element_line(linewidth=1.5))
print(pagg)
tmp <- results_all  %>% filter(metric=='pearson translation') %>%
  filter(method %in% c("FlowTransOP (pairs+sim max agg.)",
                       "FlowTransOP (pairs+sim mean agg.)",
                       "FlowTransOP (pairs+sim sum agg.)")) %>%
  filter(grepl('sample_ratio',rds_basename)) %>%
  ungroup()
ggpaired(tmp,
       x= 'method',y='mean_value',id = 'number_paired_samples',color = 'method') +
  ylab('Average pearson`s r for translation')+
  ylim(c(0,1))+
  stat_compare_means(ref.group = "FlowTransOP (pairs+sim max agg.)",
                     label = 'p.signif',
                     method='wilcox.test',
                     paired = TRUE)+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        legend.text = element_text("Arial",size = 14),
        legend.title = element_blank(),
        legend.position = 'top',
        panel.grid.major = element_line(linewidth=1.5))


final_vis_res <- results_all  %>% filter(metric=='pearson translation') %>%
  # filter(number_paired_samples<50) %>%
  filter(!(method %in% c("FlowTransOP (pairs+sim mean agg.)",
                         "FlowTransOP (pairs+sim max agg.)")))

## Check statistical significance at x = 32
vals_auto <- atanh(final_vis_res %>%
  filter(number_paired_samples == 32, method == "AutoTransOP") %>%
  pull(value))
vals_flow <- atanh(final_vis_res %>%
  filter(number_paired_samples == 32, method == "FlowTransOP (unpaired)") %>%
  pull(value))
p_val    <- t.test(vals_auto, vals_flow, paired = TRUE, var.equal = FALSE)$p.value
sig_label <- ifelse(p_val < 0.001, "***",
                    ifelse(p_val < 0.01,  "**",
                           ifelse(p_val < 0.05,  "*", "ns")))
## Check also at 7 pairs
vals_auto2 <- atanh(final_vis_res %>%
                     filter(number_paired_samples == 7, method == "AutoTransOP") %>%
                     pull(value))
vals_flow2 <- atanh(final_vis_res %>%
                     filter(number_paired_samples == 7, method == "FlowTransOP (unpaired)") %>%
                     pull(value))
p_val2    <- t.test(vals_auto2, vals_flow2, paired = TRUE, var.equal = FALSE)$p.value
sig_label2 <- ifelse(p_val2 < 0.001, "***",
                    ifelse(p_val2 < 0.01,  "**",
                           ifelse(p_val2 < 0.05,  "*", "ns")))
y_top     <- final_vis_res %>%
  filter(number_paired_samples == 32,
         method %in% c("AutoTransOP", "FlowTransOP (unpaired)")) %>%
  summarise(y = max(mean_value + std_value / sqrt(5))) %>%   # top of ribbon
  pull(y)
y_bracket <- y_top + 0.04   # base of bracket
y_text    <- y_bracket + 0.03
sig_layers <- list(
  annotate("text",
           x = 32, y = y_text,
           label = sig_label,
           size  = 7, fontface = "bold", color = "black"),
  annotate("text",
           x = 7, y = y_text-0.02,
           label = sig_label2,
           size  = 7, fontface = "bold", color = "black")
)

p1 <- ggplot(final_vis_res ,
            aes(number_paired_samples,mean_value,
                color = method, group = method)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5),
                  fill = method),
              linetype=0,alpha=0.2) +
  geom_line(size=0.8,alpha=0.8) + geom_point(size=2) +
  xlab('number of paired samples') + ylab('Average pearson`s r for translation')+
  ylim(c(0,1))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        legend.text = element_text("Arial",size = 14),
        legend.title = element_blank(),
        legend.position = 'top',
        panel.grid.major = element_line(linewidth=1.5))
p2 <-  ggplot(final_vis_res %>% filter(number_paired_samples<50),
              aes(number_paired_samples,mean_value,
                  color = method, group = method)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5),
                  fill = method),
              linetype=0,alpha=0.2) +
  geom_line(size=0.8,alpha=0.8) + geom_point(size=2) +
  xlab('number of paired samples') + ylab('Average pearson`s r for translation')+
  ylim(c(0,1))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.title.y =  element_blank(),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        legend.text = element_text("Arial",size = 14),
        legend.title = element_blank(),
        legend.position = 'top',
        panel.grid.major = element_line(linewidth=1.5))
p1 <- p1 + sig_layers
p2 <- p2 + sig_layers
p1c <- p1 + theme(legend.position = "top")
p2c <- p2 + theme(legend.position = "top")
p <- p1 + p2 +
  plot_layout(guides = "collect") +
  plot_annotation(theme = theme(legend.position = "top",
                                legend.justification = "left"))
print(p)
ggsave("../figures/low_pairs_comparison.png", p, 
       width = 32, height = 18,units = 'cm', dpi = 600, bg = "white")

ggsave("../figures/low_pairs_comparison.eps", p, 
       device = cairo_ps,
       width = 32, height = 18,units = 'cm', dpi = 600, bg = "white")
