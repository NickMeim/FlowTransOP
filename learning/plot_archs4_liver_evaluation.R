#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(scales)
})

eval_dir <- normalizePath(file.path("..", "archs4", "evaluation"), mustWork = TRUE)
out_dir <- file.path(eval_dir, "figures_liver")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
unlink(file.path(out_dir, c(
  "liver_expression_centroid_similarity.png",
  "liver_expression_centroid_similarity.pdf",
  "liver_expression_centroid_specificity.png",
  "liver_expression_centroid_specificity.pdf",
  "liver_expression_centroid_specificity_values.csv",
  "liver_expression_centroid_specificity_all_genes.png",
  "liver_expression_centroid_specificity_all_genes.pdf",
  "liver_expression_centroid_specificity_orthologues.png",
  "liver_expression_centroid_specificity_orthologues.pdf",
  "liver_expression_mmd_specificity.png",
  "liver_expression_mmd_specificity.pdf"
)))

model_levels <- c("FlowTransOP", "permuted_mouse", "permuted_human", "permuted_both")
plot_model_levels <- c("FlowTransOP", "permuted_both")
expr_levels <- c("liver_test_orthologues", plot_model_levels)
background_levels <- c("target_liver", plot_model_levels)
model_labels <- c(
  liver_test_orthologues = "Liver test orthologues",
  target_liver = "Real target liver",
  target_liver_permuted_space = "Real target liver\n(permuted space)",
  FlowTransOP = "FlowTransOP",
  permuted_mouse = "Permuted mouse",
  permuted_human = "Permuted human",
  permuted_both = "Permuted both"
)
model_cols <- c(
  liver_test_orthologues = "#4C78A8",
  target_liver = "#E15759",
  target_liver_permuted_space = "#A0A0A0",
  FlowTransOP = "#1B9E77",
  permuted_mouse = "#D95F02",
  permuted_human = "#7570B3",
  permuted_both = "#666666"
)

read_species_files <- function(pattern, kind) {
  files <- list.files(eval_dir, pattern, full.names = TRUE)
  if (length(files) == 0) return(tibble())
  bind_rows(lapply(files, function(path) {
    pieces <- str_match(basename(path), pattern)
    read_csv(path, show_col_types = FALSE) %>%
      mutate(
        species = str_to_title(pieces[, 2]),
        fold = coalesce(as.integer(.data$fold), as.integer(pieces[, 3])),
        result_kind = kind,
        source_file = basename(path)
      )
  }))
}

read_direction_files <- function(pattern, kind) {
  files <- list.files(eval_dir, pattern, full.names = TRUE)
  if (length(files) == 0) return(tibble())
  bind_rows(lapply(files, function(path) {
    pieces <- str_match(basename(path), pattern)
    read_csv(path, show_col_types = FALSE) %>%
      mutate(
        direction = coalesce(.data$direction, pieces[, 2]),
        fold = coalesce(as.integer(.data$fold), as.integer(pieces[, 3])),
        result_kind = kind,
        source_file = basename(path)
      )
  }))
}

read_folded <- function(pattern) {
  files <- list.files(eval_dir, pattern, full.names = TRUE)
  if (length(files) == 0) return(tibble())
  bind_rows(lapply(files, function(path) {
    fold <- as.integer(str_match(basename(path), "fold([0-9]+)\\.csv$")[, 2])
    read_csv(path, show_col_types = FALSE) %>%
      mutate(fold = coalesce(as.integer(.data$fold), fold),
             source_file = basename(path))
  }))
}

fmt_direction <- function(x) {
  if (!"direction" %in% names(x)) return(tibble())
  x %>%
    mutate(
      direction = recode(direction, h2m = "Human to mouse", m2h = "Mouse to human"),
      direction = factor(direction, levels = c("Mouse to human", "Human to mouse"))
    )
}

fmt_model <- function(x, levels = model_levels) {
  if (!"model_type" %in% names(x)) return(tibble())
  x %>%
    mutate(
      model_type = as.character(model_type),
      model_label = factor(model_labels[model_type], levels = model_labels[levels])
    ) %>%
    filter(model_type %in% levels, !is.na(model_label))
}

p_stars <- function(p) {
  case_when(
    is.na(p) ~ "ns",
    p <= 0.001 ~ "***",
    p <= 0.01 ~ "**",
    p <= 0.05 ~ "*",
    TRUE ~ "ns"
  )
}

paired_stats <- function(data, group_cols, baselines, lower_better = NULL) {
  if (nrow(data) == 0) return(tibble())
  data %>%
    group_by_at(group_cols) %>%
    group_modify(~ {
      flow <- .x %>% filter(model_type == "FlowTransOP") %>% select(fold, flow = value)
      panel_lower_better <- if (is.null(lower_better)) isTRUE(.x$lower_better[1]) else isTRUE(lower_better)
      alt <- if (panel_lower_better) "less" else "greater"
      bind_rows(lapply(baselines, function(base_name) {
        base <- .x %>% filter(model_type == base_name) %>% select(fold, baseline = value)
        paired <- inner_join(flow, base, by = "fold")
        p_val <- if (nrow(paired) >= 3) {
          tryCatch(
            wilcox.test(paired$flow, paired$baseline, paired = TRUE,
                        alternative = alt, exact = FALSE)$p.value,
            error = function(e) NA_real_
          )
        } else NA_real_
        tibble(
          baseline = base_name,
          n_folds = nrow(paired),
          mean_flow = mean(paired$flow, na.rm = TRUE),
          mean_baseline = mean(paired$baseline, na.rm = TRUE),
          mean_diff = mean(paired$flow - paired$baseline, na.rm = TRUE),
          lower_better = panel_lower_better,
          p_value = p_val
        )
      }))
    }) %>%
    ungroup() %>%
    group_by_at(group_cols) %>%
    mutate(p_adj_holm = p.adjust(p_value, method = "holm"),
           stars = p_stars(p_adj_holm)) %>%
    ungroup()
}

bracket_labels <- function(data, group_cols, baselines, lower_better = NULL) {
  stats <- paired_stats(data, group_cols, baselines, lower_better)
  stats <- stats %>% filter(n_folds > 0)
  if (nrow(stats) == 0) return(tibble())
  x_levels <- levels(data$model_label)
  y_pos <- data %>%
    group_by_at(group_cols) %>%
    summarise(
      y_min = min(value, na.rm = TRUE),
      y_max = max(value, na.rm = TRUE),
      .groups = "drop"
    )
  stats %>%
    left_join(y_pos, by = group_cols) %>%
    group_by_at(group_cols) %>%
    arrange(baseline, .by_group = TRUE) %>%
    mutate(
      n_refs = n(),
      xmin = match(model_labels["FlowTransOP"], x_levels),
      xmax = match(model_labels[baseline], x_levels),
      x = xmax,
      y = y_max + row_number() * 0.055 * pmax(y_max - y_min, 0.5),
      y_tip = y - 0.015 * pmax(y_max - y_min, 0.5),
      xmid = (xmin + xmax) / 2
    ) %>%
    ungroup()
}

single_ref_labels <- function(data, group_cols, baseline = "permuted_both", lower_better = NULL) {
  stats <- paired_stats(data, group_cols, baseline, lower_better)
  stats <- stats %>% filter(n_folds > 0)
  if (nrow(stats) == 0) return(tibble())
  y_pos <- data %>%
    group_by_at(group_cols) %>%
    summarise(
      y_min = min(value, na.rm = TRUE),
      y_max = max(value, na.rm = TRUE),
      .groups = "drop"
    )
  stats %>%
    left_join(y_pos, by = group_cols) %>%
    mutate(
      model_label = factor(model_labels[baseline], levels = levels(data$model_label)),
      y = y_max + 0.12 * pmax(y_max - y_min, 0.1)
    )
}

add_brackets <- function(labels) {
  if (nrow(labels) == 0 || !"n_refs" %in% names(labels)) return(list())
  list(
    geom_segment(data = labels %>% filter(n_refs > 1), aes(x = xmin, xend = xmax, y = y, yend = y),
                 inherit.aes = FALSE, linewidth = 0.3, color = "grey25"),
    geom_segment(data = labels %>% filter(n_refs > 1), aes(x = xmin, xend = xmin, y = y, yend = y_tip),
                 inherit.aes = FALSE, linewidth = 0.3, color = "grey25"),
    geom_segment(data = labels %>% filter(n_refs > 1), aes(x = xmax, xend = xmax, y = y, yend = y_tip),
                 inherit.aes = FALSE, linewidth = 0.3, color = "grey25"),
    geom_text(data = labels %>% filter(n_refs > 1), aes(x = xmid, y = y, label = stars),
              inherit.aes = FALSE, vjust = -0.16, size = 4.8, color = "grey20"),
    geom_text(data = labels %>% filter(n_refs == 1), aes(x = x, y = y, label = stars),
              inherit.aes = FALSE, vjust = -0.16, size = 5.2, color = "grey20")
  )
}

save_both <- function(plot, stem, width, height) {
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height, dpi = 300)
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height)
}

theme_archs4 <- function(base_size = 12) {
  theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 4),
      plot.subtitle = element_text(size = base_size, color = "grey30"),
      plot.caption = element_text(hjust = 0, size = base_size - 3, color = "grey35"),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      strip.text = element_text(face = "bold"),
      axis.title = element_text(size = base_size * 1.3),
      axis.text = element_text(size = base_size * 1.05),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "bottom",
      plot.margin = margin(8, 24, 8, 8)
    )
}

cycle_ps <- read_species_files("^liver_cycle_(human|mouse)_persample_fold([0-9]+)\\.csv$", "cycle_ps") %>%
  fmt_model(model_levels) %>%
  mutate(species = factor(species, levels = c("Human", "Mouse")))
orth_ps <- read_direction_files("^liver_orthologue_(h2m|m2h)_fold([0-9]+)\\.csv$", "orth_ps") %>%
  fmt_direction() %>%
  fmt_model(model_levels)
orth_summary_raw <- read_direction_files("^liver_orthologue_(h2m|m2h)_summary_fold([0-9]+)\\.csv$", "orth_summary")
orth_summary_available <- nrow(orth_summary_raw) > 0 &&
  all(c("gene_marginal_r_mean", "gene_marginal_r_var") %in% names(orth_summary_raw))
orth_summary <- if (orth_summary_available) {
  orth_summary_raw %>%
    fmt_direction() %>%
    fmt_model(model_levels)
} else {
  tibble()
}
latent_mmd <- read_folded("^liver_latent_mmd_fold[0-9]+\\.csv$") %>%
  fmt_direction() %>%
  fmt_model(c("FlowTransOP", "permuted_both"))
expr_mmd <- read_folded("^liver_expression_mmd_fold[0-9]+\\.csv$") %>%
  fmt_direction() %>%
  mutate(
    feature_set = recode(feature_set, all_target_genes = "All target genes", orthologues = "Orthologues"),
    feature_set = factor(feature_set, levels = c("Orthologues", "All target genes"))
  )
latent_centroid <- read_folded("^liver_centroid_latent_fold[0-9]+\\.csv$") %>%
  fmt_direction()
expr_centroid <- read_folded("^liver_centroid_expression_fold[0-9]+\\.csv$") %>%
  fmt_direction() %>%
  mutate(
    feature_set = recode(feature_set, all_target_genes = "All target genes", orthologues = "Orthologues", latent = "Latent"),
    feature_set = factor(feature_set, levels = c("Orthologues", "All target genes", "Latent"))
  )
reconstruction <- read_folded("^liver_reconstruction_fold[0-9]+\\.csv$") %>%
  fmt_model(c("FlowTransOP", "permuted_both"))
if (nrow(reconstruction) > 0) {
  reconstruction <- reconstruction %>%
    mutate(
      species = str_to_title(species),
      species = factor(species, levels = c("Human", "Mouse"))
    )
}

if (nrow(cycle_ps) == 0) {
  stop("No liver cycle CSVs found in ", eval_dir)
}

cycle_perf <- bind_rows(
  cycle_ps %>%
    transmute(fold, species, model_type, model_label, metric = "Per-sample", value = per_sample_mean),
  cycle_ps %>%
    select(fold, species, model_type, model_label, gene_marginal_r_mean, gene_marginal_r_var) %>%
    pivot_longer(c(gene_marginal_r_mean, gene_marginal_r_var), names_to = "metric_raw", values_to = "value") %>%
    mutate(metric = recode(metric_raw,
      gene_marginal_r_mean = "Gene-marginal mean",
      gene_marginal_r_var = "Gene-marginal variance"
    )) %>%
    select(-metric_raw)
) %>%
  filter(model_type %in% plot_model_levels) %>%
  mutate(
    model_type = factor(model_type, levels = plot_model_levels),
    model_label = factor(model_labels[as.character(model_type)], levels = model_labels[plot_model_levels]),
    metric = factor(metric, levels = c("Per-sample", "Gene-marginal mean", "Gene-marginal variance")),
    lower_better = FALSE
  )
cycle_stats <- paired_stats(cycle_perf, c("species", "metric"), "permuted_both")
cycle_labels <- single_ref_labels(cycle_perf, c("species", "metric"), "permuted_both")
write_csv(cycle_perf, file.path(out_dir, "liver_cycle_plot_values.csv"))
write_csv(cycle_stats, file.path(out_dir, "liver_cycle_paired_wilcoxon_stats.csv"))

p_cycle <- ggplot(cycle_perf, aes(model_label, value, fill = model_type)) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
  geom_point(position = position_jitter(width = 0.08, height = 0, seed = 61),
             size = 1.7, alpha = 0.78, color = "grey15") +
  geom_text(data = cycle_labels, aes(model_label, y, label = stars),
            inherit.aes = FALSE, vjust = -0.15, size = 5.8, color = "grey20") +
  facet_grid(metric ~ species, scales = "free_y") +
  scale_fill_manual(values = model_cols, breaks = plot_model_levels, labels = model_labels[plot_model_levels]) +
  scale_y_continuous(limits = c(NA, 1.2), n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
  labs(
    title = "External Liver Test: Cycle Consistency",
    subtitle = "Per-sample and gene-marginal distribution agreement on held-out liver samples",
    x = NULL,
    y = "Pearson correlation",
    fill = NULL,
    caption = "Statistics compare FlowTransOP to permuted both using paired one-sided Wilcoxon tests across folds."
  ) +
  coord_cartesian(clip = "off") +
  theme_archs4(15)
save_both(p_cycle, "liver_cycle_consistency_boxplots", 14, 10.5)

if (nrow(reconstruction) > 0) {
  reconstruction_perf <- reconstruction %>%
    filter(model_type %in% plot_model_levels) %>%
    select(fold, species, model_type, model_label, pearson_mu, pearson_var) %>%
    pivot_longer(c(pearson_mu, pearson_var),
                 names_to = "metric_raw", values_to = "value") %>%
    mutate(
      metric = recode(metric_raw,
        pearson_mu = "Mean reconstruction Pearson",
        pearson_var = "Variance reconstruction Pearson"
      ),
      metric = factor(metric, levels = c(
        "Mean reconstruction Pearson", "Variance reconstruction Pearson"
      )),
      model_type = factor(model_type, levels = plot_model_levels),
      model_label = factor(model_labels[as.character(model_type)], levels = model_labels[plot_model_levels]),
      lower_better = FALSE
    ) %>%
    select(-metric_raw)
  reconstruction_stats <- paired_stats(reconstruction_perf, c("species", "metric"), "permuted_both")
  reconstruction_labels <- single_ref_labels(reconstruction_perf, c("species", "metric"), "permuted_both")
  write_csv(reconstruction_perf, file.path(out_dir, "liver_reconstruction_plot_values.csv"))
  write_csv(reconstruction_stats, file.path(out_dir, "liver_reconstruction_paired_wilcoxon_stats.csv"))

  p_reconstruction <- ggplot(reconstruction_perf, aes(model_label, value, fill = model_type)) +
    geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 60),
               size = 1.7, alpha = 0.78, color = "grey15") +
    geom_text(data = reconstruction_labels, aes(model_label, y, label = stars),
              inherit.aes = FALSE, vjust = -0.15, size = 5.2, color = "grey20") +
    facet_grid(metric ~ species, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = plot_model_levels, labels = model_labels[plot_model_levels]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.24))) +
    labs(
      title = "External Liver Test: Reconstruction",
      subtitle = "Direct Gaussian reconstruction metrics on held-out liver samples",
      x = NULL,
      y = "Metric value",
      fill = NULL,
      caption = "Statistics compare FlowTransOP to permuted both using paired one-sided Wilcoxon tests across folds."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(12)
  save_both(p_reconstruction, "liver_reconstruction_evaluation_boxplots", 13, 10.5)
}

if (nrow(orth_ps) > 0) {
  orth_sample_perf <- orth_ps %>%
    group_by(fold, direction, model_type, model_label) %>%
    summarise(value = mean(pearson, na.rm = TRUE), .groups = "drop") %>%
    filter(model_type %in% plot_model_levels) %>%
    mutate(
      model_type = factor(model_type, levels = plot_model_levels),
      model_label = factor(model_labels[as.character(model_type)], levels = model_labels[plot_model_levels])
    ) %>%
    mutate(metric = "Per-sample orthologues", lower_better = FALSE)
  orth_gm_perf <- if (nrow(orth_summary) > 0) {
    orth_summary %>%
      select(fold, direction, model_type, model_label,
             gene_marginal_r_mean, gene_marginal_r_var) %>%
      pivot_longer(c(gene_marginal_r_mean, gene_marginal_r_var),
                   names_to = "metric_raw", values_to = "value") %>%
      mutate(
        metric = recode(metric_raw,
          gene_marginal_r_mean = "Gene-marginal mean",
          gene_marginal_r_var = "Gene-marginal variance"
        ),
        lower_better = FALSE
      ) %>%
      select(-metric_raw)
  } else {
    tibble()
  }
  orth_perf <- orth_sample_perf %>%
    mutate(metric = factor(metric, levels = "Per-sample orthologues"))
  orth_stats <- paired_stats(orth_perf, c("direction", "metric"), "permuted_both")
  orth_labels <- single_ref_labels(orth_perf, c("direction", "metric"), "permuted_both")
  write_csv(orth_perf, file.path(out_dir, "liver_orthologue_plot_values.csv"))
  write_csv(orth_stats, file.path(out_dir, "liver_orthologue_paired_wilcoxon_stats.csv"))

  p_orth <- ggplot(orth_perf, aes(model_label, value, fill = model_type)) +
    geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 62),
               size = 1.8, alpha = 0.78, color = "grey15") +
    geom_text(data = orth_labels, aes(model_label, y, label = stars),
              inherit.aes = FALSE, vjust = -0.15, size = 5.5, color = "grey20") +
    facet_wrap(~ direction, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = plot_model_levels, labels = model_labels[plot_model_levels]) +
    scale_y_continuous(limits = c(NA, 1.2), n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "External Liver Test: Orthologue-Mediated Translation",
      subtitle = "Per-sample orthologue correlation by translation direction",
      x = NULL,
      y = "Pearson correlation",
      fill = NULL,
      caption = "Statistics compare FlowTransOP to permuted both using paired one-sided Wilcoxon tests across folds."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(14)
  save_both(p_orth, "liver_orthologue_evaluation_boxplots", 10.5, 5.4)
}

if (nrow(latent_mmd) > 0) {
  latent_mmd_main <- latent_mmd %>%
    filter(comparison == "translated_source_vs_real_target") %>%
    transmute(fold, direction, model_type, model_label, value = mmd, lower_better = TRUE)
  latent_mmd_stats <- paired_stats(latent_mmd_main, c("direction"), "permuted_both")
  latent_mmd_labels <- single_ref_labels(latent_mmd_main, c("direction"), "permuted_both")
  write_csv(latent_mmd_stats, file.path(out_dir, "liver_latent_mmd_paired_wilcoxon_stats.csv"))

  p_latent_mmd <- ggplot(latent_mmd_main, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 63),
               size = 1.8, alpha = 0.8, color = "grey15") +
    geom_text(data = latent_mmd_labels, aes(model_label, y, label = stars),
              inherit.aes = FALSE, vjust = -0.15, size = 5.2, color = "grey20") +
    facet_wrap(~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = c("FlowTransOP", "permuted_both"),
                      labels = model_labels[c("FlowTransOP", "permuted_both")]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.28))) +
    labs(
      title = "External Liver Test: Latent-Space MMD",
      subtitle = "Lower is better: translated liver source latents should match real target liver latents",
      x = NULL,
      y = expression(MMD^2),
      fill = NULL
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(12)
  save_both(p_latent_mmd, "liver_latent_mmd_lower_is_better", 9.5, 5.8)
}

if (nrow(expr_mmd) > 0) {
  expr_mmd_main <- expr_mmd %>%
    filter(comparison %in% c(
      "translated_liver_source_vs_liver_target",
      "source_liver_test_vs_target_liver_test",
      "source_liver_test_vs_target_liver_test_orthologue_reference"
    )) %>%
    fmt_model(expr_levels) %>%
    transmute(fold, direction, feature_set, model_type, model_label, value = mmd, lower_better = TRUE)
  expr_mmd_stats <- paired_stats(expr_mmd_main, c("direction", "feature_set"), "permuted_both")
  expr_mmd_labels <- single_ref_labels(expr_mmd_main, c("direction", "feature_set"), "permuted_both")
  write_csv(expr_mmd_main, file.path(out_dir, "liver_expression_mmd_plot_values.csv"))
  write_csv(expr_mmd_stats, file.path(out_dir, "liver_expression_mmd_paired_wilcoxon_stats.csv"))

  p_expr_mmd <- ggplot(expr_mmd_main, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 64),
               size = 1.6, alpha = 0.78, color = "grey15") +
    geom_text(data = expr_mmd_labels, aes(model_label, y, label = stars),
              inherit.aes = FALSE, vjust = -0.15, size = 5, color = "grey20") +
    facet_grid(feature_set ~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = expr_levels, labels = model_labels[expr_levels]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.24))) +
    labs(
      title = "External Liver Test: Expression-Space MMD",
      subtitle = "Lower is better; the liver-test orthologue baseline is shown as the uncorrected source-vs-target reference",
      x = NULL,
      y = expression(MMD^2),
      fill = NULL,
      caption = "Liver test orthologues are shown as a visual reference only. Statistics compare FlowTransOP to permuted both using paired one-sided Wilcoxon tests across folds."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(10.8)
  save_both(p_expr_mmd, "liver_expression_mmd_lower_is_better", 15, 8.2)

  expr_mmd_specificity <- expr_mmd %>%
    filter(comparison %in% c(
      "real_target_liver_vs_random_non_liver_target",
      "translated_liver_source_vs_random_non_liver_target"
    )) %>%
    fmt_model(background_levels) %>%
    mutate(
      model_label = factor(model_labels[model_type], levels = model_labels[background_levels])
    )
  if (nrow(expr_mmd_specificity) > 0) {
    write_csv(expr_mmd_specificity, file.path(out_dir, "liver_expression_mmd_specificity_values.csv"))
    for (fs in levels(expr_mmd_specificity$feature_set)) {
      fs_data <- expr_mmd_specificity %>% filter(feature_set == fs)
      if (nrow(fs_data) == 0) next
      fs_stats_data <- fs_data %>%
        filter(model_type %in% model_levels) %>%
        group_by(fold, direction, model_type, model_label) %>%
        summarise(value = mean(mmd, na.rm = TRUE), .groups = "drop") %>%
        mutate(lower_better = FALSE)
      fs_labels <- single_ref_labels(
        fs_stats_data,
        c("direction"),
        "permuted_both"
      )
      p_expr_mmd_specificity <- ggplot(fs_data, aes(model_label, mmd, fill = model_type)) +
        geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
        geom_point(position = position_jitter(width = 0.08, height = 0, seed = 164),
                   size = 1.25, alpha = 0.58, color = "grey15") +
        geom_text(data = fs_labels, aes(model_label, y, label = stars),
                  inherit.aes = FALSE, vjust = -0.15, size = 4.8, color = "grey20") +
        facet_wrap(~ direction, scales = "free_y") +
        scale_fill_manual(values = model_cols, breaks = background_levels,
                          labels = model_labels[background_levels]) +
        scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.24))) +
        labs(
          title = paste0("External Liver Test: Expression MMD Specificity (", fs, ")"),
          subtitle = "Each box is MMD to random non-liver target samples. Real target liver is a visual baseline; larger MMD means stronger separation from non-liver.",
          x = NULL,
          y = expression(MMD^2),
          fill = NULL
        ) +
        coord_cartesian(clip = "off") +
        theme_archs4(10.5)
      stem <- if (fs == "All target genes") "liver_expression_mmd_specificity_all_genes" else "liver_expression_mmd_specificity_orthologues"
      save_both(p_expr_mmd_specificity, stem, 12.5, 6.5)
    }
  }
}

centroid_long <- function(df, levels, comparisons = NULL) {
  if (nrow(df) == 0) return(tibble())
  out <- df
  if (!is.null(comparisons)) out <- out %>% filter(comparison %in% comparisons)
  out %>%
    fmt_model(levels) %>%
    select(fold, direction, feature_set, comparison, model_type, model_label,
           centroid_euclidean, centroid_cosine, centroid_pearson, centroid_spearman,
           any_of(c("mean_top1_cosine", "mean_topk_cosine", "random_repeat", "background_n"))) %>%
    pivot_longer(c(centroid_euclidean, centroid_cosine, centroid_pearson, centroid_spearman,
                   any_of(c("mean_top1_cosine", "mean_topk_cosine"))),
                 names_to = "metric", values_to = "value") %>%
    mutate(
      metric = recode(metric,
        centroid_euclidean = "Centroid distance",
        centroid_cosine = "Centroid cosine",
        centroid_pearson = "Centroid Pearson",
        centroid_spearman = "Centroid Spearman",
        mean_top1_cosine = "Mean top-1 target cosine",
        mean_topk_cosine = "Mean top-k target cosine"
      ),
      metric = factor(metric, levels = c(
        "Centroid distance", "Centroid cosine", "Centroid Pearson", "Centroid Spearman",
        "Mean top-1 target cosine", "Mean top-k target cosine"
      )),
      lower_better = metric == "Centroid distance"
    )
}

latent_centroid_main <- centroid_long(
  latent_centroid,
  c("FlowTransOP", "permuted_both"),
  "translated_source_vs_real_target"
) %>%
  filter(metric == "Centroid distance")
if (nrow(latent_centroid_main) > 0) {
  latent_centroid_stats <- paired_stats(latent_centroid_main, c("direction", "metric"), "permuted_both")
  latent_centroid_labels <- single_ref_labels(latent_centroid_main, c("direction", "metric"), "permuted_both")
  write_csv(latent_centroid_main, file.path(out_dir, "liver_latent_centroid_plot_values.csv"))
  write_csv(latent_centroid_stats, file.path(out_dir, "liver_latent_centroid_paired_wilcoxon_stats.csv"))

  p_latent_centroid <- ggplot(latent_centroid_main, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 65),
               size = 1.7, alpha = 0.78, color = "grey15") +
    geom_text(data = latent_centroid_labels, aes(model_label, y, label = stars),
              inherit.aes = FALSE, vjust = -0.15, size = 4.8, color = "grey20") +
    facet_wrap(~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = c("FlowTransOP", "permuted_both"),
                      labels = model_labels[c("FlowTransOP", "permuted_both")]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.28))) +
    labs(
      title = "External Liver Test: Latent Centroid Similarity",
      subtitle = "Centroid distance between translated liver source latents and real target liver latents; lower is better",
      x = NULL,
      y = "Metric value",
      fill = NULL
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(11)
  save_both(p_latent_centroid, "liver_latent_centroid_similarity", 9.5, 5.8)
}

latent_centroid_specificity <- centroid_long(
  latent_centroid,
  c("target_liver", "FlowTransOP"),
  c("real_target_liver_vs_random_non_liver_target", "translated_liver_source_vs_random_non_liver_target")
) %>%
  filter(metric == "Centroid distance") %>%
  mutate(
    model_label = factor(model_labels[model_type], levels = model_labels[c("target_liver", "FlowTransOP")])
  )
if (nrow(latent_centroid_specificity) > 0) {
  write_csv(latent_centroid_specificity, file.path(out_dir, "liver_latent_centroid_specificity_values.csv"))
  p_latent_specificity <- ggplot(latent_centroid_specificity, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 165),
               size = 1.2, alpha = 0.55, color = "grey15") +
    facet_wrap(~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = c("target_liver", "FlowTransOP"),
                      labels = model_labels[c("target_liver", "FlowTransOP")]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "External Liver Test: Latent Centroid Specificity Against Random Non-Liver Tissue",
      subtitle = "Centroid distance to random non-liver target latents. Real target liver is the expected liver-vs-non-liver baseline; larger distance means stronger separation.",
      x = NULL,
      y = "Metric value",
      fill = NULL
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(8.8)
  save_both(p_latent_specificity, "liver_latent_centroid_specificity", 9.5, 5.8)
}

expr_centroid_main <- centroid_long(
  expr_centroid,
  expr_levels,
  c("translated_liver_source_vs_liver_target", "source_liver_test_vs_target_liver_test")
)
if (nrow(expr_centroid_main) > 0) {
  expr_centroid_extra_baseline <- expr_centroid_main %>%
    filter(model_type == "liver_test_orthologues", feature_set == "Orthologues") %>%
    mutate(feature_set = factor("All target genes", levels = levels(expr_centroid_main$feature_set)))
  expr_centroid_main <- bind_rows(expr_centroid_main, expr_centroid_extra_baseline) %>%
    mutate(feature_set = factor(feature_set, levels = c("Orthologues", "All target genes")))
  expr_centroid_stats <- paired_stats(expr_centroid_main, c("direction", "feature_set", "metric"), "permuted_both")
  write_csv(expr_centroid_main, file.path(out_dir, "liver_expression_centroid_plot_values.csv"))
  write_csv(expr_centroid_stats, file.path(out_dir, "liver_expression_centroid_paired_wilcoxon_stats.csv"))

  for (fs in levels(expr_centroid_main$feature_set)) {
    fs_data <- expr_centroid_main %>% filter(feature_set == fs)
    if (nrow(fs_data) == 0) next
    fs_labels <- single_ref_labels(
      fs_data %>% filter(model_type %in% model_levels),
      c("direction", "metric"),
      "permuted_both"
    )
    p_expr_centroid <- ggplot(fs_data, aes(model_label, value, fill = model_type)) +
      geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
      geom_point(position = position_jitter(width = 0.08, height = 0, seed = 66),
                 size = 1.35, alpha = 0.74, color = "grey15") +
      geom_text(data = fs_labels, aes(model_label, y, label = stars),
                inherit.aes = FALSE, vjust = -0.15, size = 4.4, color = "grey20") +
      facet_grid(metric ~ direction, scales = "free_y") +
      scale_fill_manual(values = model_cols, breaks = expr_levels, labels = model_labels[expr_levels]) +
      scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.24))) +
      labs(
        title = paste0("External Liver Test: Expression Centroid Similarity (", fs, ")"),
        subtitle = "Similarity is translated liver source versus real target liver. The orthologue liver-test baseline is repeated in all-target-gene panels as a visual reference.",
        x = NULL,
        y = "Metric value",
        fill = NULL,
        caption = "Liver test orthologues are shown as a visual reference only. Statistics compare FlowTransOP to permuted both using paired one-sided Wilcoxon tests across folds."
      ) +
      coord_cartesian(clip = "off") +
      theme_archs4(9.5)
    stem <- if (fs == "All target genes") "liver_expression_centroid_similarity_all_genes" else "liver_expression_centroid_similarity_orthologues"
    save_both(p_expr_centroid, stem, 13.5, 11)
  }
}

message("Wrote liver figures and summaries to: ", out_dir)
