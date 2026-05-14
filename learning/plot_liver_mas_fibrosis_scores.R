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
  orthologues = "Orthologues"
)

scoring_space_labels <- c(
  translated_human = "Translated all human genes",
  raw_mouse_orthologues = "Raw mouse orthologues"
)

dataset_specs <- list(
  mouse_nlrp3 = list(
    label = "GSE196908 Nlrp3A350V",
    file_slug = "gse196908_nlrp3a350v",
    treatment_levels = c("Control chow+placebo", "GS-444217"),
    control = "Control chow+placebo",
    treatment = "GS-444217"
  ),
  mouse_gan_dio_lanifibranor = list(
    label = "GSE196908 GAN DIO-NASH",
    file_slug = "gse196908_gan_dio_nash",
    treatment_levels = c("DIO-NASH Vehicle", "Lanifibranor"),
    control = "DIO-NASH Vehicle",
    treatment = "Lanifibranor"
  ),
  mouse_cdaa_lanifibranor = list(
    label = "GSE269493 CDAA-HFD",
    file_slug = "gse269493_cdaa_hfd",
    treatment_levels = c("Chow", "CDAA-HFD vehicle", "Lanifibranor"),
    control = "CDAA-HFD vehicle",
    treatment = "Lanifibranor"
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

mouse_stats_for <- function(data, spec) {
  levels_x <- spec$treatment_levels
  data %>%
    mutate(mouse_treatment = factor(mouse_treatment, levels = levels_x)) %>%
    group_by(cohort, dataset, method, scoring_space, endpoint, time_point) %>%
    group_modify(~ {
      panel <- .x %>% filter(!is.na(score), !is.na(mouse_treatment))
      control <- panel %>% filter(mouse_treatment == spec$control)
      treatment <- panel %>% filter(mouse_treatment == spec$treatment)
      p_val <- if (nrow(control) > 0 && nrow(treatment) > 0) {
        tryCatch(
          wilcox.test(score ~ mouse_treatment,
                      data = panel %>% filter(mouse_treatment %in% c(spec$control, spec$treatment)),
                      exact = FALSE)$p.value,
          error = function(e) NA_real_
        )
      } else {
        NA_real_
      }
      y_min <- min(panel$score, na.rm = TRUE)
      y_max <- max(panel$score, na.rm = TRUE)
      span <- max(y_max - y_min, 0.25)
      tibble(
        comparison = paste(spec$control, "vs", spec$treatment),
        n_control = nrow(control),
        n_treated = nrow(treatment),
        mean_control = mean(control$score, na.rm = TRUE),
        mean_treated = mean(treatment$score, na.rm = TRUE),
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

write_csv(human_agg, file.path(out_dir, "human_fold_mean_predictions.csv"))
write_csv(human_cor, file.path(out_dir, "human_fold_mean_spearman_stats.csv"))
write_csv(human_loocv, file.path(out_dir, "human_plsr_loocv_predictions.csv"))
write_csv(loocv_cor, file.path(out_dir, "human_plsr_loocv_spearman_stats.csv"))
write_csv(mouse_agg, file.path(out_dir, "mouse_fold_mean_scores.csv"))
write_csv(mouse_stats, file.path(out_dir, "mouse_treatment_wilcox_stats.csv"))

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

plot_mouse_dataset_method <- function(cohort_name, method_name) {
  spec <- dataset_specs[[cohort_name]]
  plot_data <- mouse_agg %>%
    filter(cohort == cohort_name, method == method_name) %>%
    mutate(mouse_treatment = factor(mouse_treatment, levels = spec$treatment_levels))
  stat_data <- mouse_stats %>%
    filter(cohort == cohort_name, method == method_name)

  if (nrow(plot_data) == 0) return(invisible(NULL))

  facet_formula <- if (n_distinct(plot_data$time_point) > 1) {
    endpoint ~ time_point + scoring_space
  } else {
    endpoint ~ scoring_space
  }

  p <- ggplot(plot_data, aes(x = mouse_treatment, y = score, fill = mouse_treatment)) +
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
      y = "Fold-mean inferred score",
      title = paste0(spec$label, " score inference: ", method_name),
      subtitle = paste0("Statistics compare ", spec$control, " vs ", spec$treatment,
                        "; scores are averaged across available folds")
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 8.5)
    )

  suffix <- paste0(spec$file_slug, "_", str_replace_all(tolower(method_name), "[^a-z0-9]+", "_"), "_boxplots")
  width <- if (n_distinct(plot_data$time_point) > 1) 13.2 else 8.8
  ggsave(file.path(out_dir, paste0(suffix, ".png")), p, width = width, height = 7.2, dpi = 300)
  ggsave(file.path(out_dir, paste0(suffix, ".pdf")), p, width = width, height = 7.2, device = cairo_pdf)
}

for (cohort_name in names(dataset_specs)) {
  for (method_name in names(method_cols)) {
    plot_mouse_dataset_method(cohort_name, method_name)
  }
}

message("Wrote figures and aggregated tables to: ", out_dir)
