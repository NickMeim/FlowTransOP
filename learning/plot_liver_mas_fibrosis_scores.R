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

method_cols <- c(
  PLSR = "#2C7FB8",
  `Neural net` = "#7B3294",
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
  raw_mouse_orthologues = "Raw mouse orthologues",
  translated_human_latent = "Translated latent"
)

human_ml_space_labels <- c(
  human_expression = "Human all genes",
  human_orthologue_expression = "Human orthologues",
  human_latent = "Human latent"
)

ml_method_labels <- c(
  plsr = "PLSR",
  neural_net = "Neural net",
  svm_rbf = "SVM RBF"
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
    fold <- str_match(basename(path), "fold([0-9]+)\\.csv$")[, 2]
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

human_fold_long <- bind_rows(
  human_long_from_wide(
    read_fold_files("^human_govaere_plsr_scores_fold[0-9]+\\.csv$"),
    "PLSR", "all_human_genes",
    "predicted_plsr_nas_score", "predicted_plsr_fibrosis_stage"
  ),
  human_long_from_wide(
    read_fold_files("^human_govaere_orthologue_plsr_scores_fold[0-9]+\\.csv$"),
    "PLSR", "orthologues",
    "predicted_orthologue_plsr_nas_score", "predicted_orthologue_plsr_fibrosis_stage"
  ),
  human_long_from_wide(
    read_fold_files("^human_govaere_signature_scores_fold[0-9]+\\.csv$"),
    "Rank/aREA", "all_human_genes",
    "predicted_signature_nas_score", "predicted_signature_fibrosis_stage"
  ),
  human_long_from_wide(
    read_fold_files("^human_govaere_orthologue_signature_scores_fold[0-9]+\\.csv$"),
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

human_ml_fold_long <- human_ml_long_from_csv(
  read_fold_files("^human_govaere_ml_model_scores_fold[0-9]+\\.csv$")
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
  bind_rows(
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_translated_plsr_scores_fold[0-9]+\\.csv$")),
      "PLSR", cohort_name,
      "predicted_translated_human_plsr_nas_score",
      "predicted_translated_human_plsr_fibrosis_stage"
    ),
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_raw_orthologue_plsr_scores_fold[0-9]+\\.csv$")),
      "PLSR", cohort_name,
      "predicted_raw_mouse_orthologue_plsr_nas_score",
      "predicted_raw_mouse_orthologue_plsr_fibrosis_stage"
    ),
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_translated_signature_scores_fold[0-9]+\\.csv$")),
      "Rank/aREA", cohort_name,
      "predicted_signature_nas_score",
      "predicted_signature_fibrosis_stage"
    ),
    mouse_long_from_wide(
      read_fold_files(paste0("^", cohort_name, "_raw_orthologue_signature_scores_fold[0-9]+\\.csv$")),
      "Rank/aREA", cohort_name,
      "predicted_signature_nas_score",
      "predicted_signature_fibrosis_stage"
    ),
    mouse_ml_long_from_csv(
      read_fold_files(paste0("^", cohort_name, "_ml_model_scores_fold[0-9]+\\.csv$")),
      cohort_name
    )
  )
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
    .groups = "drop"
  )

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
          wilcox.test(as.formula(paste(score_col, "~ mouse_treatment")),
                      data = panel %>% filter(mouse_treatment %in% c(spec$control, spec$treatment)),
                      exact = FALSE)$p.value,
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
        p_value = p_val,
        x_start = match(spec$control, levels_x),
        x_end = match(spec$treatment, levels_x),
        x_mid = mean(c(x_start, x_end)),
        y_bracket = y_max + 0.08 * span,
        y_tip = y_max + 0.03 * span,
        y_label = y_max + 0.14 * span
      )
    }) %>%
    ungroup() %>%
    mutate(label = pretty_p(p_value))
}

mouse_stats <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_stats_for(mouse_agg %>% filter(cohort == cohort_name), dataset_specs[[cohort_name]])
}))

mouse_deltas_for <- function(cohort_name) {
  spec <- dataset_specs[[cohort_name]]
  mouse_agg %>%
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

mouse_delta_stats <- bind_rows(lapply(names(dataset_specs), function(cohort_name) {
  mouse_stats_for(
    mouse_deltas %>% filter(cohort == cohort_name),
    dataset_specs[[cohort_name]],
    score_col = "score_delta"
  )
}))

write_csv(human_agg, file.path(out_dir, "human_fold_mean_predictions.csv"))
write_csv(human_cor, file.path(out_dir, "human_fold_mean_spearman_stats.csv"))
write_csv(human_ml_agg, file.path(out_dir, "human_ml_fold_mean_predictions.csv"))
write_csv(human_ml_cor, file.path(out_dir, "human_ml_fold_mean_spearman_stats.csv"))
write_csv(human_loocv, file.path(out_dir, "human_plsr_loocv_predictions.csv"))
write_csv(loocv_cor, file.path(out_dir, "human_plsr_loocv_spearman_stats.csv"))
if (nrow(human_ml_loocv) > 0) {
  write_csv(human_ml_loocv, file.path(out_dir, "human_ml_loocv_predictions.csv"))
  write_csv(human_ml_loocv_cor, file.path(out_dir, "human_ml_loocv_spearman_stats.csv"))
}
write_csv(mouse_agg, file.path(out_dir, "mouse_fold_mean_scores.csv"))
write_csv(mouse_stats, file.path(out_dir, "mouse_treatment_wilcox_stats.csv"))
write_csv(mouse_deltas, file.path(out_dir, "mouse_fold_mean_score_deltas.csv"))
write_csv(mouse_delta_stats, file.path(out_dir, "mouse_treatment_delta_wilcox_stats.csv"))

p_human <- ggplot(human_agg, aes(x = observed, y = predicted, color = method)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
  geom_point(size = 1.9, alpha = 0.82) +
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
    y = "Fold-mean predicted score",
    title = "Human Govaere score inference",
    subtitle = "Predictions are averaged across available folds before correlation testing"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.95
  )

ggsave(file.path(out_dir, "human_govaere_fold_mean_prediction_scatter.png"), p_human,
       width = 13.2, height = 7.2, dpi = 300)
ggsave(file.path(out_dir, "human_govaere_fold_mean_prediction_scatter.pdf"), p_human,
       width = 13.2, height = 7.2, device = cairo_pdf)

p_human_ml <- ggplot(human_ml_agg, aes(x = observed, y = predicted, color = method)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
  geom_point(size = 1.75, alpha = 0.82) +
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
    y = "Fold-mean predicted score",
    title = "Human Govaere ML score inference",
    subtitle = "PLSR, neural net, and RBF-SVM predictions are averaged across folds"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    panel.grid.minor = element_blank(),
    aspect.ratio = 0.95,
    strip.text.x = element_text(size = 8.5)
  )

ggsave(file.path(out_dir, "human_govaere_ml_model_fold_mean_scatter.png"), p_human_ml,
       width = 17.6, height = 7.6, dpi = 300)
ggsave(file.path(out_dir, "human_govaere_ml_model_fold_mean_scatter.pdf"), p_human_ml,
       width = 17.6, height = 7.6, device = cairo_pdf)

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
ggsave(file.path(out_dir, "human_govaere_plsr_loocv_scatter.pdf"), p_loocv,
       width = 8.4, height = 7.2, device = cairo_pdf)

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
      subtitle = "Neural net and RBF-SVM models are trained without the held-out sample"
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
  ggsave(file.path(out_dir, "human_govaere_ml_loocv_scatter.pdf"), p_ml_loocv,
         width = 14.2, height = 7.6, device = cairo_pdf)
}

plot_mouse_dataset_method <- function(cohort_name, method_name,
                                      plot_source = mouse_agg,
                                      stat_source = mouse_stats,
                                      score_col = "score",
                                      y_label = "Fold-mean inferred score",
                                      title_label = "score inference",
                                      file_suffix = "boxplots",
                                      subtitle_prefix = "Statistics compare") {
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
                          "; scores are averaged across available folds")
  if (score_col == "score_delta") {
    subtitle_text <- paste0("Deltas are relative to ", spec$delta_baseline,
                            "; statistics compare ", spec$control, " vs ", spec$treatment,
                            "; scores are averaged across available folds")
  }

  p <- ggplot(plot_data, aes(x = mouse_treatment, y = .data[[score_col]], fill = mouse_treatment)) +
    {if (score_col == "score_delta") geom_hline(yintercept = 0, linewidth = 0.35,
                                                linetype = "dashed", color = "grey50")} +
    geom_boxplot(width = 0.58, outlier.shape = NA, alpha = 0.75, color = "grey25") +
    geom_jitter(width = 0.12, size = 1.95, alpha = 0.86, color = "grey15") +
    geom_segment(data = stat_data, aes(x = x_start, xend = x_end, y = y_bracket, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.35, color = "grey20") +
    geom_segment(data = stat_data, aes(x = x_start, xend = x_start, y = y_tip, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.35, color = "grey20") +
    geom_segment(data = stat_data, aes(x = x_end, xend = x_end, y = y_tip, yend = y_bracket),
                 inherit.aes = FALSE, linewidth = 0.35, color = "grey20") +
    geom_text(data = stat_data, aes(x = x_mid, y = y_label, label = label),
              inherit.aes = FALSE, size = 3.1, color = "grey15") +
    facet_grid(facet_formula, scales = "free_y") +
    scale_fill_manual(values = treatment_cols, guide = "none", drop = FALSE) +
    scale_x_discrete(labels = treatment_labels, drop = FALSE) +
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
      axis.text.x = element_text(size = 8.5)
    )

  suffix <- paste0(spec$file_slug, "_", str_replace_all(tolower(method_name), "[^a-z0-9]+", "_"), "_", file_suffix)
  n_facet_cols <- n_distinct(plot_data$scoring_space) *
    ifelse(n_distinct(plot_data$time_point) > 1, n_distinct(plot_data$time_point), 1)
  width <- max(8.8, min(18, 2.8 * n_facet_cols + 3.2))
  ggsave(file.path(out_dir, paste0(suffix, ".png")), p, width = width, height = 7.2, dpi = 300)
  ggsave(file.path(out_dir, paste0(suffix, ".pdf")), p, width = width, height = 7.2, device = cairo_pdf)
}

for (cohort_name in names(dataset_specs)) {
  for (method_name in names(method_cols)) {
    plot_mouse_dataset_method(cohort_name, method_name)
    plot_mouse_dataset_method(
      cohort_name,
      method_name,
      plot_source = mouse_deltas,
      stat_source = mouse_delta_stats,
      score_col = "score_delta",
      y_label = "Fold-mean score delta",
      title_label = "disease-score deltas",
      file_suffix = "delta_boxplots"
    )
  }
}

message("Wrote figures and aggregated tables to: ", out_dir)
