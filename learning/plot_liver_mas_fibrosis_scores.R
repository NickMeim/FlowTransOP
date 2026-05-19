#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(scales)
})

get_arg <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) return(default)
  args[[hit + 1]]
}

score_dir <- get_arg("--score_dir", file.path("..", "archs4", "evaluation", "liver_mas_fibrosis"))
score_dir <- normalizePath(score_dir, mustWork = TRUE)
out_dir <- get_arg("--out_dir", file.path(score_dir, "figures"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
analysis_mode <- get_arg(
  "--analysis_mode",
  ifelse(grepl("full_ensemble|ensemble", basename(score_dir), ignore.case = TRUE), "ensemble", "fold")
)
replicate_singular <- ifelse(analysis_mode == "ensemble", "ensemble member", "fold")
replicate_plural <- ifelse(analysis_mode == "ensemble", "ensemble members", "available folds")
replicate_mean_label <- ifelse(analysis_mode == "ensemble", "Ensemble-mean", "Fold-mean")
replicate_file_token <- ifelse(analysis_mode == "ensemble", "ensemble_mean", "fold_mean")
plot_file_token <- ifelse(analysis_mode == "ensemble", "ensemble_members", "fold_mean")

method_cols <- c(
  PLSR = "#2C7FB8",
  `SVM RBF` = "#1B9E77",
  `Rank/aREA` = "#D95F02"
)

treatment_cols <- c(
  `Control chow+placebo` = "#4C78A8",
  `GS-444217` = "#E15759",
  `DIO-NASH Vehicle` = "#4C78A8",
  Lanifibranor = "#E15759",
  `CDAA-HFD vehicle` = "#4C78A8",
  Chow = "#59A14F"
)

treatment_labels <- c(
  `Control chow+placebo` = "Control\nchow + placebo",
  `GS-444217` = "GS-444217",
  `DIO-NASH Vehicle` = "DIO-NASH\nvehicle",
  Lanifibranor = "Lanifibranor",
  `CDAA-HFD vehicle` = "CDAA-HFD\nvehicle",
  Chow = "Chow"
)

endpoint_labels <- c(
  nas_score = "NAS/MAS score",
  fibrosis_stage = "Fibrosis stage"
)

gene_space_labels <- c(
  all_genes = "All genes",
  all_human_genes = "All genes",
  orthologues = "Orthologues",
  latent = "Latent"
)

scoring_space_labels <- c(
  translated_human = "Translated all human genes",
  translated_orthologues = "Translated orthologues",
  raw_mouse_orthologues = "Raw mouse orthologues",
  translated_human_latent = "Translated latent",
  decoder_sampled_translated_human = "Sampled translated all genes",
  decoder_sampled_translated_orthologues = "Sampled translated orthologues"
)

human_ml_space_labels <- c(
  human_expression = "Human all genes",
  human_orthologue_expression = "Human orthologues",
  human_latent = "Human latent"
)

ml_method_labels <- c(
  plsr = "PLSR",
  svm_rbf = "SVM RBF",
  rank_area = "Rank/aREA"
)

dataset_specs <- list(
  mouse_nlrp3 = list(
    label = "GSE196908 Nlrp3A350V",
    file_slug = "gse196908_nlrp3a350v",
    treatment_levels = c("Control chow+placebo", "GS-444217"),
    control = "Control chow+placebo",
    treatment = "GS-444217",
    delta_baseline = "Control chow+placebo"
  ),
  mouse_gan_dio_lanifibranor = list(
    label = "GSE196908 GAN DIO-NASH",
    file_slug = "gse196908_gan_dio_nash",
    treatment_levels = c("DIO-NASH Vehicle", "Lanifibranor"),
    control = "DIO-NASH Vehicle",
    treatment = "Lanifibranor",
    delta_baseline = "DIO-NASH Vehicle"
  ),
  mouse_cdaa_lanifibranor = list(
    label = "GSE269493 CDAA-HFD",
    file_slug = "gse269493_cdaa_hfd",
    treatment_levels = c("Chow", "CDAA-HFD vehicle", "Lanifibranor"),
    control = "CDAA-HFD vehicle",
    treatment = "Lanifibranor",
    delta_baseline = "Chow"
  )
)

read_fold_files <- function(pattern) {
  files <- list.files(score_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) {
    stop("No files matched pattern: ", pattern, " in ", score_dir)
  }
  bind_rows(lapply(files, function(path) {
    fold <- str_match(basename(path), "(?:fold|ensemble)([0-9]+)\\.csv$")[, 2]
    read_csv(path, show_col_types = FALSE) %>%
      mutate(
        fold = as.integer(fold),
        source_file = basename(path)
      )
  }))
}

read_optional_fold_files <- function(pattern) {
  files <- list.files(score_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(tibble())
  bind_rows(lapply(files, function(path) {
    fold <- str_match(basename(path), "(?:fold|ensemble)([0-9]+)\\.csv$")[, 2]
    read_csv(path, show_col_types = FALSE) %>%
      mutate(
        fold = as.integer(fold),
        source_file = basename(path)
      )
  }))
}

read_optional_csv <- function(path) {
  if (!file.exists(path)) return(tibble())
  read_csv(path, show_col_types = FALSE)
}

pretty_p <- function(p) {
  case_when(
    is.na(p) ~ "p = NA",
    p < 1e-4 ~ "p < 1e-4",
    p < 1e-3 ~ paste0("p = ", formatC(p, format = "e", digits = 1)),
    TRUE ~ paste0("p = ", signif(p, 2))
  )
}

pretty_p_less <- function(p) {
  str_replace(pretty_p(p), "^p", "p_less")
}

spearman_stats <- function(data, group_cols) {
  data %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      n_samples = n(),
      rho = if (n() >= 3 && n_distinct(observed) > 1 && n_distinct(predicted) > 1) {
        suppressWarnings(cor(observed, predicted, method = "spearman", use = "complete.obs"))
      } else {
        NA_real_
      },
      p_value = if (n() >= 3 && n_distinct(observed) > 1 && n_distinct(predicted) > 1) {
        suppressWarnings(cor.test(observed, predicted, method = "spearman", exact = FALSE)$p.value)
      } else {
        NA_real_
      },
      .groups = "drop"
    ) %>%
    mutate(
      label = paste0(
        "Spearman rho = ", sprintf("%.2f", rho),
        "\n", pretty_p(p_value)
      )
    )
}

human_long_from_wide <- function(data, method_name, gene_space_name, nas_col, fib_col) {
  data %>%
    transmute(
      sample_id,
      fold,
      method = method_name,
      gene_space = gene_space_name,
      observed_nas_score,
      observed_fibrosis_stage,
      predicted_nas_score = .data[[nas_col]],
      predicted_fibrosis_stage = .data[[fib_col]]
    ) %>%
    pivot_longer(
      cols = c(observed_nas_score, observed_fibrosis_stage,
               predicted_nas_score, predicted_fibrosis_stage),
      names_to = c(".value", "endpoint"),
      names_pattern = "(observed|predicted)_(nas_score|fibrosis_stage)"
    )
}

human_ml_long_from_csv <- function(data) {
  data %>%
    filter(model_type != "neural_net") %>%
    transmute(
      sample_id,
      fold,
      method = recode(model_type, !!!ml_method_labels),
      feature_space = input_space,
      gene_space,
      observed_nas_score,
      observed_fibrosis_stage,
      predicted_nas_score,
      predicted_fibrosis_stage
    ) %>%
    pivot_longer(
      cols = c(observed_nas_score, observed_fibrosis_stage,
               predicted_nas_score, predicted_fibrosis_stage),
      names_to = c(".value", "endpoint"),
      names_pattern = "(observed|predicted)_(nas_score|fibrosis_stage)"
    )
}

human_ml_loocv_long_from_csv <- function(data) {
  data %>%
    filter(model_type != "neural_net") %>%
    transmute(
      sample_id,
      method = recode(model_type, !!!ml_method_labels),
      feature_space = input_space,
      gene_space,
      observed_nas_score,
      observed_fibrosis_stage,
      predicted_loocv_nas_score,
      predicted_loocv_fibrosis_stage,
      n_train_samples,
      n_features
    ) %>%
    pivot_longer(
      cols = c(observed_nas_score, observed_fibrosis_stage,
               predicted_loocv_nas_score, predicted_loocv_fibrosis_stage),
      names_to = c(".value", "endpoint"),
      names_pattern = "(observed|predicted_loocv)_(nas_score|fibrosis_stage)"
    ) %>%
    rename(predicted = predicted_loocv)
}

mouse_long_from_wide <- function(data, method_name, cohort_name, nas_col, fib_col) {
  data %>%
    transmute(
      sample_id,
      fold,
      cohort = cohort_name,
      dataset,
      mouse_model,
      mouse_treatment,
      time_point,
      input_space,
      gene_space,
      method = method_name,
      nas_score = .data[[nas_col]],
      fibrosis_stage = .data[[fib_col]]
    ) %>%
    pivot_longer(
      cols = c(nas_score, fibrosis_stage),
      names_to = "endpoint",
      values_to = "score"
    )
}

mouse_ml_long_from_csv <- function(data, cohort_name) {
  data %>%
    filter(model_type != "neural_net") %>%
    filter(model_type != "plsr" | input_space == "translated_human_latent") %>%
    transmute(
      sample_id,
      fold,
      cohort = cohort_name,
      dataset,
      mouse_model,
      mouse_treatment,
      time_point,
      input_space,
      gene_space,
      method = recode(model_type, !!!ml_method_labels),
      nas_score = predicted_nas_score,
      fibrosis_stage = predicted_fibrosis_stage
    ) %>%
    pivot_longer(
      cols = c(nas_score, fibrosis_stage),
      names_to = "endpoint",
      values_to = "score"
    )
}

mouse_decoder_draws_long_from_csv <- function(data, cohort_name) {
  if (nrow(data) == 0) return(tibble())
  data %>%
    filter(model_type != "neural_net") %>%
    transmute(
      sample_id,
      fold,
      cohort = cohort_name,
      dataset,
      mouse_model,
      mouse_treatment,
      time_point,
      input_space,
      gene_space,
      decoder_sample,
      method = recode(model_type, !!!ml_method_labels),
      nas_score = predicted_nas_score,
      fibrosis_stage = predicted_fibrosis_stage
    ) %>%
    pivot_longer(
      cols = c(nas_score, fibrosis_stage),
      names_to = "endpoint",
      values_to = "score"
    )
}

human_fold_long <- bind_rows(
  human_long_from_wide(
    read_fold_files("^human_govaere_plsr_scores_(?:fold|ensemble)[0-9]+\\.csv$"),
    "PLSR", "all_human_genes",
    "predicted_plsr_nas_score", "predicted_plsr_fibrosis_stage"
  ),
  human_long_from_wide(
    read_fold_files("^human_govaere_orthologue_plsr_scores_(?:fold|ensemble)[0-9]+\\.csv$"),
    "PLSR", "orthologues",
    "predicted_orthologue_plsr_nas_score", "predicted_orthologue_plsr_fibrosis_stage"
  ),
  human_long_from_wide(
    read_fold_files("^human_govaere_signature_scores_(?:fold|ensemble)[0-9]+\\.csv$"),
    "Rank/aREA", "all_human_genes",
    "predicted_signature_nas_score", "predicted_signature_fibrosis_stage"
  ),
  human_long_from_wide(
    read_fold_files("^human_govaere_orthologue_signature_scores_(?:fold|ensemble)[0-9]+\\.csv$"),
    "Rank/aREA", "orthologues",
    "predicted_signature_nas_score", "predicted_signature_fibrosis_stage"
  )
) %>%
  mutate(
    method = factor(method, levels = names(method_cols)),
    gene_space = factor(gene_space, levels = names(gene_space_labels), labels = gene_space_labels),
    endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels)
  )

human_agg <- human_fold_long %>%
  group_by(sample_id, method, gene_space, endpoint) %>%
  summarise(
    observed = mean(observed, na.rm = TRUE),
    predicted = mean(predicted, na.rm = TRUE),
    n_folds = n_distinct(fold),
    .groups = "drop"
  )

human_cor <- spearman_stats(human_agg, c("method", "gene_space", "endpoint"))
human_plot_data <- if (analysis_mode == "ensemble") human_fold_long else human_agg

human_ml_fold_long <- human_ml_long_from_csv(
  read_fold_files("^human_govaere_ml_model_scores_(?:fold|ensemble)[0-9]+\\.csv$")
) %>%
  mutate(
    method = factor(method, levels = names(method_cols)),
    feature_space = factor(
      feature_space,
      levels = names(human_ml_space_labels),
      labels = human_ml_space_labels
    ),
    gene_space = factor(gene_space, levels = names(gene_space_labels), labels = gene_space_labels),
    endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels)
  )

human_ml_agg <- human_ml_fold_long %>%
  group_by(sample_id, method, feature_space, gene_space, endpoint) %>%
  summarise(
    observed = mean(observed, na.rm = TRUE),
    predicted = mean(predicted, na.rm = TRUE),
    n_folds = n_distinct(fold),
    .groups = "drop"
  )

human_ml_cor <- spearman_stats(human_ml_agg, c("method", "feature_space", "endpoint"))
human_ml_plot_data <- if (analysis_mode == "ensemble") human_ml_fold_long else human_ml_agg

loocv_all <- read_optional_csv(file.path(score_dir, "human_govaere_plsr_loocv_all_genes.csv"))
loocv_orth <- read_optional_csv(file.path(score_dir, "human_govaere_plsr_loocv_orthologues.csv"))
human_loocv <- bind_rows(loocv_all, loocv_orth) %>%
  pivot_longer(
    cols = c(observed_nas_score, observed_fibrosis_stage,
             predicted_loocv_nas_score, predicted_loocv_fibrosis_stage),
    names_to = c(".value", "endpoint"),
    names_pattern = "(observed|predicted_loocv)_(nas_score|fibrosis_stage)"
  ) %>%
  rename(predicted = predicted_loocv) %>%
  mutate(
    gene_space = factor(gene_space, levels = names(gene_space_labels), labels = gene_space_labels),
    endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels)
  )

loocv_cor <- spearman_stats(human_loocv, c("gene_space", "endpoint"))

ml_loocv_raw <- read_optional_csv(file.path(score_dir, "human_govaere_ml_loocv_predictions.csv"))
if (nrow(ml_loocv_raw) > 0) {
  human_ml_loocv <- human_ml_loocv_long_from_csv(ml_loocv_raw) %>%
    mutate(
      method = factor(method, levels = names(method_cols)),
      feature_space = factor(
        feature_space,
        levels = names(human_ml_space_labels),
        labels = human_ml_space_labels
      ),
      gene_space = factor(gene_space, levels = names(gene_space_labels), labels = gene_space_labels),
      endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels)
    )
  human_ml_loocv_cor <- spearman_stats(human_ml_loocv, c("method", "feature_space", "endpoint"))
} else {
  human_ml_loocv <- tibble()
  human_ml_loocv_cor <- tibble()
}

read_mouse_cohort <- function(cohort_name) {
  translated_orth_plsr <- read_optional_fold_files(
    paste0("^", cohort_name, "_translated_orthologue_plsr_scores_(?:fold|ensemble)[0-9]+\\.csv$")
  )
  translated_orth_sig <- read_optional_fold_files(
    paste0("^", cohort_name, "_translated_orthologue_signature_scores_(?:fold|ensemble)[0-9]+\\.csv$")
  )
  decoder_draws <- read_optional_fold_files(
    paste0("^", cohort_name, "_decoder_sampled_score_draws_(?:fold|ensemble)[0-9]+\\.csv$")
  )

  pieces <- list(
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_translated_plsr_scores_(?:fold|ensemble)[0-9]+\\.csv$")),
      "PLSR", cohort_name,
      "predicted_translated_human_plsr_nas_score",
      "predicted_translated_human_plsr_fibrosis_stage"
    ),
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_raw_orthologue_plsr_scores_(?:fold|ensemble)[0-9]+\\.csv$")),
      "PLSR", cohort_name,
      "predicted_raw_mouse_orthologue_plsr_nas_score",
      "predicted_raw_mouse_orthologue_plsr_fibrosis_stage"
    ),
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_translated_signature_scores_(?:fold|ensemble)[0-9]+\\.csv$")),
      "Rank/aREA", cohort_name,
      "predicted_signature_nas_score",
      "predicted_signature_fibrosis_stage"
    ),
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_raw_orthologue_signature_scores_(?:fold|ensemble)[0-9]+\\.csv$")),
      "Rank/aREA", cohort_name,
      "predicted_signature_nas_score",
      "predicted_signature_fibrosis_stage"
    ),
    mouse_ml_long_from_csv(
      read_fold_files(paste0("^", cohort_name, "_ml_model_scores_(?:fold|ensemble)[0-9]+\\.csv$")),
      cohort_name
    )
  )

  if (nrow(translated_orth_plsr) > 0) {
    pieces <- c(
      pieces,
      list(mouse_long_from_wide(
        translated_orth_plsr,
        "PLSR", cohort_name,
        "predicted_translated_orthologue_plsr_nas_score",
        "predicted_translated_orthologue_plsr_fibrosis_stage"
      ))
    )
  }

  if (nrow(translated_orth_sig) > 0) {
    pieces <- c(
      pieces,
      list(mouse_long_from_wide(
        translated_orth_sig,
        "Rank/aREA", cohort_name,
        "predicted_signature_nas_score",
        "predicted_signature_fibrosis_stage"
      ))
    )
  }

  if (nrow(decoder_draws) > 0) {
    pieces <- c(
      pieces,
      list(mouse_decoder_draws_long_from_csv(decoder_draws, cohort_name))
    )
  }

  bind_rows(pieces)
}

mouse_fold_long <- bind_rows(lapply(names(dataset_specs), read_mouse_cohort)) %>%
  mutate(
    method = factor(method, levels = names(method_cols)),
    endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels),
    scoring_space = factor(
      input_space,
      levels = names(scoring_space_labels),
      labels = scoring_space_labels
    ),
    time_point = ifelse(is.na(time_point) | time_point == "", "not_specified", time_point),
    time_point = factor(time_point, levels = c("8w", "12w", "not_specified"))
  )

mouse_agg <- mouse_fold_long %>%
  group_by(sample_id, cohort, dataset, mouse_model, mouse_treatment,
           time_point, method, scoring_space, endpoint) %>%
  summarise(
    score = mean(score, na.rm = TRUE),
    n_folds = n_distinct(fold),
    n_decoder_samples = n_distinct(decoder_sample, na.rm = TRUE),
    n_score_rows = n(),
    .groups = "drop"
  )

average_sampled_scores_across_replicates <- function(data) {
  sampled <- data %>%
    filter(grepl("^decoder_sampled_", input_space)) %>%
    group_by(sample_id, cohort, dataset, mouse_model, mouse_treatment,
             time_point, input_space, gene_space, method, scoring_space,
             endpoint, decoder_sample) %>%
    summarise(
      score = mean(score, na.rm = TRUE),
      n_folds_averaged = n_distinct(fold),
      fold = NA_integer_,
      .groups = "drop"
    )

  deterministic <- data %>%
    filter(!grepl("^decoder_sampled_", input_space)) %>%
    mutate(n_folds_averaged = NA_integer_)

  bind_rows(deterministic, sampled)
}

average_scores_across_replicates <- function(data) {
  data %>%
    group_by(sample_id, cohort, dataset, mouse_model, mouse_treatment,
             time_point, input_space, gene_space, method, scoring_space,
             endpoint) %>%
    summarise(
      score = mean(score, na.rm = TRUE),
      n_folds = n_distinct(fold),
      n_decoder_samples = n_distinct(decoder_sample, na.rm = TRUE),
      n_score_rows = n(),
      .groups = "drop"
    )
}

mouse_plot_scores <- if (analysis_mode == "ensemble") {
  average_sampled_scores_across_replicates(mouse_fold_long)
} else {
  mouse_agg
}

mouse_mu_member_scores <- if (analysis_mode == "ensemble") {
  mouse_fold_long %>%
    filter(!grepl("^decoder_sampled_", input_space)) %>%
    mutate(n_folds_averaged = NA_integer_)
} else {
  mouse_agg %>% filter(!grepl("^Sampled translated", as.character(scoring_space)))
}
mouse_mu_agg <- mouse_agg %>% filter(!grepl("^Sampled translated", as.character(scoring_space)))
mouse_mu_mean_scores <- if (analysis_mode == "ensemble") {
  average_scores_across_replicates(mouse_mu_member_scores)
} else {
  mouse_mu_agg
}

mouse_stats_for <- function(data, spec, score_col = "score") {
  levels_x <- spec$treatment_levels
  data %>%
    mutate(mouse_treatment = factor(mouse_treatment, levels = levels_x)) %>%
    group_by(cohort, dataset, method, scoring_space, endpoint, time_point) %>%
    group_modify(~ {
      panel <- .x %>% filter(!is.na(.data[[score_col]]), !is.na(mouse_treatment))
      control <- panel %>% filter(mouse_treatment == spec$control)
      treatment <- panel %>% filter(mouse_treatment == spec$treatment)
      p_val <- if (nrow(control) > 0 && nrow(treatment) > 0) {
        tryCatch(
          wilcox.test(
            treatment[[score_col]],
            control[[score_col]],
            alternative = "less",
            exact = FALSE
          )$p.value,
          error = function(e) NA_real_
        )
      } else {
        NA_real_
      }
      y_min <- min(panel[[score_col]], na.rm = TRUE)
      y_max <- max(panel[[score_col]], na.rm = TRUE)
      span <- max(y_max - y_min, 0.25)
      tibble(
        comparison = paste(spec$control, "vs", spec$treatment),
        n_control = nrow(control),
        n_treated = nrow(treatment),
        mean_control = mean(control[[score_col]], na.rm = TRUE),
        mean_treated = mean(treatment[[score_col]], na.rm = TRUE),
        mean_diff_treated_minus_control = mean_treated - mean_control,
        alternative = paste(spec$treatment, "<", spec$control),
        p_value = p_val,
        x_start = match(spec$control, levels_x),
        x_end = match(spec$treatment, levels_x),
        x_mid = mean(c(x_start, x_end)),
        y_bracket = y_max + 0.16 * span,
        y_tip = y_max + 0.08 * span,
        y_label = y_max + 0.27 * span
      )
    }) %>%
    ungroup() %>%
    mutate(label = pretty_p_less(p_value))
}

mouse_stats <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_stats_for(mouse_agg %>% filter(cohort == cohort_name), dataset_specs[[cohort_name]])
}))

mouse_mu_stats <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_stats_for(mouse_mu_agg %>% filter(cohort == cohort_name), dataset_specs[[cohort_name]])
}))

mouse_deltas_for <- function(cohort_name, data = mouse_agg) {
  spec <- dataset_specs[[cohort_name]]
  data %>%
    filter(cohort == cohort_name) %>%
    group_by(cohort, dataset, method, scoring_space, endpoint, time_point) %>%
    group_modify(~ {
      baseline_scores <- .x$score[as.character(.x$mouse_treatment) == spec$delta_baseline]
      baseline_mean <- if (length(baseline_scores) > 0 && any(!is.na(baseline_scores))) {
        mean(baseline_scores, na.rm = TRUE)
      } else {
        NA_real_
      }
      .x %>%
        mutate(
          delta_baseline = spec$delta_baseline,
          delta_baseline_mean = baseline_mean,
          score_delta = score - baseline_mean
        )
    }) %>%
    ungroup()
}

mouse_deltas <- bind_rows(lapply(names(dataset_specs), mouse_deltas_for))
mouse_delta_plot_scores <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_deltas_for(cohort_name, mouse_plot_scores)
}))

mouse_delta_stats <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_stats_for(
    mouse_deltas %>% filter(cohort == cohort_name),
    dataset_specs[[cohort_name]],
    score_col = "score_delta"
  )
}))

mouse_mu_deltas <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_deltas_for(cohort_name, mouse_mu_agg)
}))
mouse_mu_member_delta_scores <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_deltas_for(cohort_name, mouse_mu_member_scores)
}))
mouse_mu_mean_delta_scores <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_deltas_for(cohort_name, mouse_mu_mean_scores)
}))
mouse_mu_delta_stats <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_stats_for(
    mouse_mu_deltas %>% filter(cohort == cohort_name),
    dataset_specs[[cohort_name]],
    score_col = "score_delta"
  )
}))

scatter_point_alpha <- ifelse(analysis_mode == "ensemble", 0.34, 0.82)
scatter_point_size <- ifelse(analysis_mode == "ensemble", 1.45, 1.9)
human_scatter_y <- ifelse(
  analysis_mode == "ensemble",
  "Predicted score from each ensemble member",
  paste0(replicate_mean_label, " predicted score")
)
performance_subtitle <- ifelse(
  analysis_mode == "ensemble",
  "Points show individual ensemble-member predictions; correlations use ensemble-mean predictions per sample",
  paste0("Predictions are averaged across ", replicate_plural, " before correlation testing")
)

p_human <- ggplot(human_plot_data, aes(x = observed, y = predicted, color = method)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
  geom_point(size = scatter_point_size, alpha = scatter_point_alpha) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, color = "grey20") +
  geom_text(
    data = human_cor,
    aes(x = -Inf, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.06,
    vjust = 1.12,
    size = 3.2,
    lineheight = 0.95,
    color = "grey15"
  ) +
  facet_grid(endpoint ~ method + gene_space, scales = "free") +
  scale_color_manual(values = method_cols, guide = "none") +
  labs(
    x = "Observed human score",
    y = human_scatter_y,
    title = "Human Govaere score inference",
    subtitle = performance_subtitle
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.95
  )

ggsave(file.path(out_dir, paste0("human_govaere_", plot_file_token, "_prediction_scatter.png")), p_human,
       width = 13.2, height = 7.2, dpi = 300)

p_human_ml <- ggplot(human_ml_plot_data, aes(x = observed, y = predicted, color = method)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
  geom_point(size = scatter_point_size, alpha = scatter_point_alpha) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.55, color = "grey20") +
  geom_text(
    data = human_ml_cor,
    aes(x = -Inf, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.06,
    vjust = 1.12,
    size = 2.8,
    lineheight = 0.95,
    color = "grey15"
  ) +
  facet_grid(endpoint ~ method + feature_space, scales = "free") +
  scale_color_manual(values = method_cols, guide = "none") +
  labs(
    x = "Observed human score",
    y = human_scatter_y,
    title = "Human Govaere ML score inference",
    subtitle = ifelse(
      analysis_mode == "ensemble",
      "Points show individual ensemble-member predictions; correlations use ensemble-mean predictions per sample",
      paste0("PLSR and RBF-SVM predictions are averaged across ", replicate_plural)
    )
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.95,
    strip.text.x = element_text(size = 8.5)
  )

ggsave(file.path(out_dir, paste0("human_govaere_ml_model_", plot_file_token, "_scatter.png")), p_human_ml,
       width = 17.6, height = 7.6, dpi = 300)

p_loocv <- ggplot(human_loocv, aes(x = observed, y = predicted)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
  geom_point(size = 1.95, alpha = 0.82, color = method_cols[["PLSR"]]) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, color = "grey20") +
  geom_text(
    data = loocv_cor,
    aes(x = -Inf, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.06,
    vjust = 1.12,
    size = 3.3,
    lineheight = 0.95,
    color = "grey15"
  ) +
  facet_grid(endpoint ~ gene_space, scales = "free") +
  labs(
    x = "Observed human score",
    y = "LOOCV predicted score",
    title = "Human Govaere PLSR leave-one-out performance",
    subtitle = "Each point is predicted by a PLSR model trained without that sample"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.95
  )

ggsave(file.path(out_dir, "human_govaere_plsr_loocv_scatter.png"), p_loocv,
       width = 8.4, height = 7.2, dpi = 300)

if (nrow(human_ml_loocv) > 0) {
  p_ml_loocv <- ggplot(human_ml_loocv, aes(x = observed, y = predicted, color = method)) +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
    geom_point(size = 1.75, alpha = 0.82) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.55, color = "grey20") +
    geom_text(
      data = human_ml_loocv_cor,
      aes(x = -Inf, y = Inf, label = label),
      inherit.aes = FALSE,
      hjust = -0.06,
      vjust = 1.12,
      size = 2.8,
      lineheight = 0.95,
      color = "grey15"
    ) +
    facet_grid(endpoint ~ method + feature_space, scales = "free") +
    scale_color_manual(values = method_cols, guide = "none") +
    labs(
      x = "Observed human score",
      y = "LOOCV predicted score",
      title = "Human Govaere ML leave-one-out performance",
      subtitle = "RBF-SVM models are trained without the held-out sample"
    ) +
    theme_bw(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      panel.grid.minor = element_blank(),
      aspect.ratio = 0.95,
      strip.text.x = element_text(size = 8.5)
    )

  ggsave(file.path(out_dir, "human_govaere_ml_loocv_scatter.png"), p_ml_loocv,
         width = 14.2, height = 7.6, dpi = 300)
}

plot_mouse_dataset_method <- function(cohort_name, method_name,
                                      plot_source = mouse_plot_scores,
                                      stat_source = mouse_stats,
                                      score_col = "score",
                                      y_label = ifelse(
                                        analysis_mode == "ensemble",
                                        "Inferred score from each ensemble member",
                                        paste0(replicate_mean_label, " inferred score")
                                      ),
                                      title_label = "score inference",
                                      file_suffix = "boxplots",
                                      subtitle_prefix = "One-sided Wilcoxon tests compare",
                                      ensemble_note = NULL,
                                      ensemble_delta_note = NULL) {
  spec <- dataset_specs[[cohort_name]]
  plot_data <- plot_source %>%
    filter(cohort == cohort_name, method == method_name) %>%
    mutate(mouse_treatment = factor(mouse_treatment, levels = spec$treatment_levels))
  stat_data <- stat_source %>%
    filter(cohort == cohort_name, method == method_name)

  if (nrow(plot_data) == 0) return(invisible(NULL))

  facet_formula <- if (n_distinct(plot_data$time_point) > 1) {
    endpoint ~ time_point + scoring_space
  } else {
    endpoint ~ scoring_space
  }
  subtitle_text <- paste0(subtitle_prefix, " ", spec$control, " vs ", spec$treatment,
                          "; scores are averaged across ", replicate_plural)
  if (analysis_mode == "ensemble") {
    if (is.null(ensemble_note)) {
      ensemble_note <- "Deterministic panels show ensemble members; sampled panels average predicted scores across ensemble members per decoder draw; statistics use per-sample mean scores"
    }
    subtitle_text <- paste0(
      ensemble_note, "; ",
      tolower(subtitle_prefix), " ", spec$control, " vs ", spec$treatment
    )
  }
  if (score_col == "score_delta") {
    subtitle_text <- paste0("Deltas are relative to ", spec$delta_baseline,
                            "; statistics compare ", spec$control, " vs ", spec$treatment,
                            "; scores are averaged across ", replicate_plural)
    if (analysis_mode == "ensemble") {
      if (is.null(ensemble_delta_note)) {
        ensemble_delta_note <- paste0(
          "Deterministic deltas show ensemble members; sampled deltas average predicted scores across ensemble members per decoder draw relative to ",
          spec$delta_baseline, "; statistics use per-sample mean scores"
        )
      }
      subtitle_text <- paste0(
        ensemble_delta_note, "; compare ", spec$control, " vs ", spec$treatment
      )
    }
  }

  mouse_point_alpha <- ifelse(analysis_mode == "ensemble", 0.24, 0.78)
  mouse_point_size <- ifelse(analysis_mode == "ensemble", 1.15, 1.95)

  p <- ggplot(plot_data, aes(x = mouse_treatment, y = .data[[score_col]], fill = mouse_treatment)) +
    {if (score_col == "score_delta") geom_hline(yintercept = 0, linewidth = 0.35,
                                                linetype = "dashed", color = "grey50")} +
    geom_boxplot(width = 0.58, outlier.shape = NA, alpha = 0.75, color = "grey25") +
    geom_jitter(width = 0.12, size = mouse_point_size, alpha = mouse_point_alpha, color = "grey15") +
    geom_segment(data = stat_data, aes(x = x_start, xend = x_end, y = y_bracket, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.35, color = "grey20") +
    geom_segment(data = stat_data, aes(x = x_start, xend = x_start, y = y_tip, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.35, color = "grey20") +
    geom_segment(data = stat_data, aes(x = x_end, xend = x_end, y = y_tip, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.35, color = "grey20") +
    geom_label(
      data = stat_data,
      aes(x = x_mid, y = y_label, label = label),
      inherit.aes = FALSE,
      size = 2.75,
      color = "grey15",
      fill = alpha("white", 0.88),
      linewidth = 0,
      label.padding = grid::unit(0.08, "lines")
    ) +
    facet_grid(facet_formula, scales = "free_y") +
    scale_y_continuous(expand = expansion(mult = c(0.06, 0.34))) +
    scale_fill_manual(values = treatment_cols, guide = "none", drop = FALSE) +
    scale_x_discrete(labels = treatment_labels, drop = FALSE) +
    coord_cartesian(clip = "off") +
    labs(
      x = NULL,
      y = y_label,
      title = paste0(spec$label, " ", title_label, ": ", method_name),
      subtitle = subtitle_text
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 8.5),
      plot.margin = margin(8, 12, 10, 8)
    )

  suffix <- paste0(spec$file_slug, "_", str_replace_all(tolower(method_name), "[^a-z0-9]+", "_"), "_", file_suffix)
  n_facet_cols <- n_distinct(plot_data$scoring_space) *
    ifelse(n_distinct(plot_data$time_point) > 1, n_distinct(plot_data$time_point), 1)
  width <- max(8.8, min(24, 2.8 * n_facet_cols + 3.2))
  ggsave(file.path(out_dir, paste0(suffix, ".png")), p, width = width, height = 7.2, dpi = 300)
}

for (cohort_name in names(dataset_specs)) {
  for (method_name in names(method_cols)) {
    plot_mouse_dataset_method(cohort_name, method_name)
    plot_mouse_dataset_method(
      cohort_name,
      method_name,
      plot_source = mouse_delta_plot_scores,
      stat_source = mouse_delta_stats,
      score_col = "score_delta",
      y_label = ifelse(
        analysis_mode == "ensemble",
        "Score delta from each ensemble member",
        paste0(replicate_mean_label, " score delta")
      ),
      title_label = "disease-score deltas",
      file_suffix = "delta_boxplots"
    )

    plot_mouse_dataset_method(
      cohort_name,
      method_name,
      plot_source = mouse_mu_member_scores,
      stat_source = mouse_mu_stats,
      y_label = ifelse(
        analysis_mode == "ensemble",
        "Deterministic score from each ensemble member",
        paste0(replicate_mean_label, " deterministic score")
      ),
      title_label = "deterministic-mu score inference",
      file_suffix = "mu_ensemble_member_boxplots",
      ensemble_note = "Panels show deterministic decoder-mean predictions from each ensemble member; statistics use ensemble-mean scores per biological sample"
    )
    plot_mouse_dataset_method(
      cohort_name,
      method_name,
      plot_source = mouse_mu_member_delta_scores,
      stat_source = mouse_mu_delta_stats,
      score_col = "score_delta",
      y_label = ifelse(
        analysis_mode == "ensemble",
        "Deterministic score delta from each ensemble member",
        paste0(replicate_mean_label, " deterministic score delta")
      ),
      title_label = "deterministic-mu disease-score deltas",
      file_suffix = "mu_ensemble_member_delta_boxplots",
      ensemble_delta_note = paste0(
        "Panels show deterministic decoder-mean deltas from each ensemble member relative to ",
        dataset_specs[[cohort_name]]$delta_baseline,
        "; statistics use ensemble-mean scores per biological sample"
      )
    )

    plot_mouse_dataset_method(
      cohort_name,
      method_name,
      plot_source = mouse_mu_mean_scores,
      stat_source = mouse_mu_stats,
      y_label = ifelse(
        analysis_mode == "ensemble",
        "Ensemble-mean deterministic score",
        paste0(replicate_mean_label, " deterministic score")
      ),
      title_label = "ensemble-mean deterministic-mu score inference",
      file_suffix = "mu_ensemble_mean_boxplots",
      ensemble_note = "Panels show deterministic decoder-mean predictions averaged across ensemble members per biological sample"
    )
    plot_mouse_dataset_method(
      cohort_name,
      method_name,
      plot_source = mouse_mu_mean_delta_scores,
      stat_source = mouse_mu_delta_stats,
      score_col = "score_delta",
      y_label = ifelse(
        analysis_mode == "ensemble",
        "Ensemble-mean deterministic score delta",
        paste0(replicate_mean_label, " deterministic score delta")
      ),
      title_label = "ensemble-mean deterministic-mu disease-score deltas",
      file_suffix = "mu_ensemble_mean_delta_boxplots",
      ensemble_delta_note = paste0(
        "Panels show deterministic decoder-mean predictions averaged across ensemble members before calculating deltas relative to ",
        dataset_specs[[cohort_name]]$delta_baseline
      )
    )
  }
}

message("Wrote PNG figures to: ", out_dir)
