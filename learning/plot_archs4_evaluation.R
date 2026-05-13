#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(patchwork)
  library(scales)
})

eval_dir <- normalizePath(file.path("..", "archs4", "evaluation"), mustWork = TRUE)
out_dir <- file.path(eval_dir, "figures_flowtransop")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
unlink(file.path(out_dir, "cycle_pergene_fold_summary.csv"))

model_levels <- c("FlowTransOP", "permuted_mouse", "permuted_human", "permuted_both")
baseline_levels <- setdiff(model_levels, "FlowTransOP")
model_labels <- c(
  FlowTransOP = "FlowTransOP",
  permuted_mouse = "Permuted mouse",
  permuted_human = "Permuted human",
  permuted_both = "Permuted both",
  no_translation = "Untranslated source",
  validation_orthologues = "Validation orthologues"
)
model_cols <- c(
  FlowTransOP = "#1B9E77",
  permuted_mouse = "#D95F02",
  permuted_human = "#7570B3",
  permuted_both = "#666666",
  no_translation = "#A6761D",
  validation_orthologues = "#4C78A8"
)

read_many <- function(files, pattern, kind) {
  if (length(files) == 0) {
    return(tibble(
      source_file = character(),
      species_or_direction = character(),
      fold_from_file = integer(),
      result_kind = character()
    ))
  }
  bind_rows(lapply(files, function(path) {
    pieces <- str_match(basename(path), pattern)
    read_csv(path, show_col_types = FALSE) %>%
      mutate(
        source_file = basename(path),
        species_or_direction = pieces[, 2],
        fold_from_file = as.integer(pieces[, 3]),
        result_kind = kind
      )
  }))
}

pretty_p <- function(p) {
  case_when(
    is.na(p) ~ "p = NA",
    p < 1e-4 ~ "p < 1e-4",
    p < 1e-3 ~ paste0("p = ", formatC(p, format = "e", digits = 1)),
    TRUE ~ paste0("p = ", signif(p, 2))
  )
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

correlation_breaks <- function(x) {
  rng <- range(x, na.rm = TRUE)
  lo <- floor(rng[1] * 10) / 10
  hi <- min(1, ceiling(rng[2] * 10) / 10)
  seq(lo, hi, by = 0.2)
}

mmd_breaks <- function(x) {
  pretty(x, n = 7)
}

paired_wilcox <- function(data, value_col = "score", lower_better = FALSE) {
  value_sym <- rlang::sym(value_col)
  alternative <- if (lower_better) "less" else "greater"
  flow <- data %>%
    filter(model_type == "FlowTransOP") %>%
    select(fold, flow = !!value_sym)

  bind_rows(lapply(baseline_levels, function(baseline) {
    base <- data %>%
      filter(model_type == baseline) %>%
      select(fold, baseline_score = !!value_sym)
    paired <- inner_join(flow, base, by = "fold")
    if (nrow(paired) < 3) {
      return(tibble(
        comparison = paste0("FlowTransOP vs ", baseline),
        baseline = baseline,
        n_folds = nrow(paired),
        mean_flow = mean(paired$flow, na.rm = TRUE),
        mean_baseline = mean(paired$baseline_score, na.rm = TRUE),
        mean_diff = mean(paired$flow - paired$baseline_score, na.rm = TRUE),
        median_diff = median(paired$flow - paired$baseline_score, na.rm = TRUE),
        p_value = NA_real_
      ))
    }
    p_val <- tryCatch(
      wilcox.test(paired$flow, paired$baseline_score, paired = TRUE,
                  alternative = alternative, exact = FALSE)$p.value,
      error = function(e) NA_real_
    )
    diff <- paired$flow - paired$baseline_score
    tibble(
      comparison = paste0("FlowTransOP vs ", baseline),
      baseline = baseline,
      n_folds = nrow(paired),
      mean_flow = mean(paired$flow, na.rm = TRUE),
      mean_baseline = mean(paired$baseline_score, na.rm = TRUE),
      mean_diff = mean(diff, na.rm = TRUE),
      median_diff = median(diff, na.rm = TRUE),
      p_value = p_val
    )
  })) %>%
    mutate(
      p_adj_holm = p.adjust(p_value, method = "holm"),
      baseline_label = factor(model_labels[baseline], levels = model_labels[model_levels]),
      directionality = ifelse(lower_better, "lower is better", "higher is better")
    )
}

paired_wilcox_against_flow <- function(data, baselines, value_col = "score", lower_better = FALSE) {
  value_sym <- rlang::sym(value_col)
  alternative <- if (lower_better) "less" else "greater"
  flow <- data %>%
    filter(model_type == "FlowTransOP") %>%
    select(fold, flow = !!value_sym)

  bind_rows(lapply(baselines, function(baseline) {
    base <- data %>%
      filter(model_type == baseline) %>%
      select(fold, baseline_score = !!value_sym)
    paired <- inner_join(flow, base, by = "fold")
    p_val <- if (nrow(paired) >= 3) {
      tryCatch(wilcox.test(paired$flow, paired$baseline_score, paired = TRUE,
                           alternative = alternative, exact = FALSE)$p.value,
               error = function(e) NA_real_)
    } else NA_real_
    diff <- paired$flow - paired$baseline_score
    tibble(
      baseline = baseline,
      n_folds = nrow(paired),
      mean_flow = mean(paired$flow, na.rm = TRUE),
      mean_baseline = mean(paired$baseline_score, na.rm = TRUE),
      mean_diff = mean(diff, na.rm = TRUE),
      median_diff = median(diff, na.rm = TRUE),
      p_value = p_val
    )
  })) %>%
    mutate(
      p_adj_holm = p.adjust(p_value, method = "holm"),
      directionality = ifelse(lower_better, "lower is better", "higher is better")
    )
}

cycle_ps_files <- list.files(eval_dir, "^cycle_(human|mouse)_persample_fold([0-9]+)\\.csv$", full.names = TRUE)
orth_files <- list.files(eval_dir, "^orthologue_(h2m|m2h)_fold([0-9]+)\\.csv$", full.names = TRUE)
orth_summary_files <- list.files(eval_dir, "^orthologue_(h2m|m2h)_summary_fold([0-9]+)\\.csv$", full.names = TRUE)
mmd_files <- list.files(eval_dir, "^mmd_fold([0-9]+)\\.csv$", full.names = TRUE)
expression_mmd_files <- list.files(eval_dir, "^expression_mmd_fold([0-9]+)\\.csv$", full.names = TRUE)

cycle_ps <- read_many(cycle_ps_files, "^cycle_(human|mouse)_persample_fold([0-9]+)\\.csv$", "cycle_persample") %>%
  mutate(
    species = str_to_title(species_or_direction),
    fold = coalesce(fold, fold_from_file),
    model_type = as.character(model_type),
    model_label = factor(model_labels[model_type], levels = model_labels[model_levels])
  )

orth <- read_many(orth_files, "^orthologue_(h2m|m2h)_fold([0-9]+)\\.csv$", "orthologue") %>%
  mutate(
    direction = recode(species_or_direction, h2m = "Human to mouse", m2h = "Mouse to human"),
    species = recode(species_or_direction, h2m = "Mouse", m2h = "Human"),
    fold = coalesce(fold, fold_from_file),
    model_type = as.character(model_type),
    model_label = factor(model_labels[model_type], levels = model_labels[model_levels])
  )

orth_summary_raw <- read_many(orth_summary_files, "^orthologue_(h2m|m2h)_summary_fold([0-9]+)\\.csv$", "orthologue_summary")
orth_summary_available <- nrow(orth_summary_raw) > 0 &&
  all(c("gene_marginal_r_mean", "gene_marginal_r_var") %in% names(orth_summary_raw))
orth_summary_wide <- if (orth_summary_available) {
  orth_summary_raw %>%
    mutate(
      direction = recode(species_or_direction, h2m = "Human to mouse", m2h = "Mouse to human"),
      species = recode(species_or_direction, h2m = "Mouse", m2h = "Human"),
      fold = coalesce(fold, fold_from_file),
      model_type = as.character(model_type),
      model_label = factor(model_labels[model_type], levels = model_labels[model_levels])
    )
} else {
  tibble()
}

mmd <- if (length(mmd_files) == 0) {
  tibble()
} else {
  bind_rows(lapply(mmd_files, read_csv, show_col_types = FALSE)) %>%
    mutate(
      direction = recode(direction, h2m = "Human to mouse", m2h = "Mouse to human"),
      species = recode(direction, `Human to mouse` = "Mouse", `Mouse to human` = "Human"),
      model_type = as.character(model_type),
      model_label = factor(model_labels[model_type], levels = model_labels[c("no_translation", model_levels)])
    )
}

expression_mmd <- if (length(expression_mmd_files) == 0) {
  tibble()
} else {
  expr_raw <- bind_rows(lapply(expression_mmd_files, read_csv, show_col_types = FALSE))
  expr_extra_baseline <- expr_raw %>%
    filter(model_type == "validation_orthologues", feature_set == "orthologues") %>%
    mutate(
      feature_set = "all_target_genes",
      comparison = "source_validation_vs_target_validation_orthologue_reference"
    )
  bind_rows(expr_raw, expr_extra_baseline) %>%
    mutate(
      direction = recode(direction, h2m = "Human to mouse", m2h = "Mouse to human"),
      species = recode(direction, `Human to mouse` = "Mouse", `Mouse to human` = "Human"),
      feature_set = recode(feature_set, all_target_genes = "All target genes", orthologues = "Orthologues"),
      model_type = as.character(model_type)
    )
}

cycle_ps_summary <- cycle_ps %>%
  transmute(
    family = "Cycle consistency",
    metric = "Per-sample",
    metric_order = 1,
    species,
    fold,
    model_type,
    model_label,
    score = per_sample_mean,
    score_name = "Mean per-sample Pearson r"
  )

cycle_gm_long <- cycle_ps %>%
  select(species, fold, model_type, model_label, gene_marginal_r_mean, gene_marginal_r_var) %>%
  pivot_longer(c(gene_marginal_r_mean, gene_marginal_r_var),
               names_to = "metric_raw", values_to = "score") %>%
  mutate(
    family = "Cycle consistency",
    metric = recode(metric_raw,
                    gene_marginal_r_mean = "Gene-marginal mean",
                    gene_marginal_r_var = "Gene-marginal variance"),
    metric_order = recode(metric_raw,
                          gene_marginal_r_mean = 2L,
                          gene_marginal_r_var = 3L),
    score_name = recode(metric_raw,
                        gene_marginal_r_mean = "Gene-marginal mean Pearson r",
                        gene_marginal_r_var = "Gene-marginal variance prediction Pearson r")
  ) %>%
  select(family, metric, metric_order, species, fold, model_type, model_label, score, score_name)

orth_summary <- orth %>%
  group_by(species, direction, fold, model_type, model_label) %>%
  summarise(score = mean(pearson, na.rm = TRUE), .groups = "drop") %>%
  transmute(
    family = "Orthologue-mediated",
    metric = paste0("Orthologue\n", direction),
    metric_order = 4,
    species,
    fold,
    model_type,
    model_label,
    score,
    score_name = "Mean per-sample orthologue Pearson r"
  )

orth_gm_long <- if (nrow(orth_summary_wide) > 0) {
  orth_summary_wide %>%
    select(species, direction, fold, model_type, model_label,
           gene_marginal_r_mean, gene_marginal_r_var) %>%
    pivot_longer(c(gene_marginal_r_mean, gene_marginal_r_var),
                 names_to = "metric_raw", values_to = "score") %>%
    transmute(
      family = "Orthologue-mediated",
      metric = recode(metric_raw,
                      gene_marginal_r_mean = paste0("Orthologue\n", direction, "\ngene-marginal mean"),
                      gene_marginal_r_var = paste0("Orthologue\n", direction, "\ngene-marginal variance")),
      metric_order = recode(metric_raw,
                            gene_marginal_r_mean = 5L,
                            gene_marginal_r_var = 6L),
      species,
      fold,
      model_type,
      model_label,
      score,
      score_name = recode(metric_raw,
                          gene_marginal_r_mean = "Orthologue gene-marginal mean Pearson r",
                          gene_marginal_r_var = "Orthologue gene-marginal variance Pearson r")
    )
} else {
  tibble()
}

cycle_performance <- bind_rows(cycle_ps_summary, cycle_gm_long)

orth_performance <- bind_rows(orth_summary, orth_gm_long)

performance <- bind_rows(cycle_performance, orth_performance) %>%
  filter(!is.na(score), !is.na(model_label)) %>%
  mutate(
    species = factor(species, levels = c("Human", "Mouse")),
    metric = factor(metric, levels = unique(metric[order(metric_order, metric)])),
    model_type = factor(model_type, levels = model_levels),
    model_label = factor(model_label, levels = model_labels[model_levels])
  )

if (nrow(performance) == 0) {
  stop("No supported evaluation CSVs were found in: ", eval_dir)
}

cycle_plot_data <- cycle_performance %>%
  filter(!is.na(score), !is.na(model_label)) %>%
  mutate(
    species = factor(species, levels = c("Human", "Mouse")),
    metric = factor(metric, levels = unique(metric[order(metric_order, metric)])),
    model_type = factor(model_type, levels = model_levels),
    model_label = factor(model_label, levels = model_labels[model_levels])
  )

orth_plot_data <- orth_performance %>%
  filter(!is.na(score), !is.na(model_label)) %>%
  mutate(
    species = factor(species, levels = c("Human", "Mouse")),
    metric = factor(metric, levels = unique(metric[order(metric_order, metric)])),
    model_type = factor(model_type, levels = model_levels),
    model_label = factor(model_label, levels = model_labels[model_levels])
  )

stats_by_panel <- performance %>%
  group_by(family, metric, metric_order, species) %>%
  group_modify(~ paired_wilcox(.x, "score", lower_better = FALSE)) %>%
  ungroup()

cycle_stats_labels <- stats_by_panel %>%
  filter(family == "Cycle consistency") %>%
  filter(!is.na(p_adj_holm)) %>%
  left_join(
    cycle_plot_data %>%
      group_by(family, metric, metric_order, species) %>%
      summarise(
        y_min = min(score, na.rm = TRUE),
        y_max = max(score, na.rm = TRUE),
        .groups = "drop"
      ),
    by = c("family", "metric", "metric_order", "species")
  ) %>%
  group_by(family, metric, metric_order, species) %>%
  arrange(baseline, .by_group = TRUE) %>%
  mutate(
    n_refs = n(),
    xmin = match(model_labels["FlowTransOP"], levels(cycle_plot_data$model_label)),
    xmax = match(as.character(baseline_label), levels(cycle_plot_data$model_label)),
    x = xmax,
    y = pmin(y_max + (row_number() * 0.10 * pmax(y_max - y_min, 0.1)), 1.17),
    y_tip = y - 0.022 * pmax(y_max - y_min, 0.1),
    xmid = (xmin + xmax) / 2,
    label = p_stars(p_adj_holm)
  ) %>%
  ungroup()

orth_stats_labels <- stats_by_panel %>%
  filter(family == "Orthologue-mediated", !is.na(p_adj_holm)) %>%
  left_join(
    orth_plot_data %>%
      group_by(family, metric, metric_order, species) %>%
      summarise(y_min = min(score, na.rm = TRUE), y_max = max(score, na.rm = TRUE), .groups = "drop"),
    by = c("family", "metric", "metric_order", "species")
  ) %>%
  group_by(family, metric, metric_order, species) %>%
  arrange(baseline, .by_group = TRUE) %>%
  mutate(
    n_refs = n(),
    xmin = match(model_labels["FlowTransOP"], levels(orth_plot_data$model_label)),
    xmax = match(as.character(baseline_label), levels(orth_plot_data$model_label)),
    x = xmax,
    y = pmin(y_max + (row_number() * 0.12 * pmax(y_max - y_min, 0.1)), 1.17),
    y_tip = y - 0.025 * pmax(y_max - y_min, 0.1),
    xmid = (xmin + xmax) / 2,
    label = p_stars(p_adj_holm)
  ) %>%
  ungroup()

summary_table <- performance %>%
  group_by(family, metric, species, model_label) %>%
  summarise(
    n_folds = n_distinct(fold),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE),
    median = median(score, na.rm = TRUE),
    q25 = quantile(score, 0.25, na.rm = TRUE),
    q75 = quantile(score, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(performance, file.path(out_dir, "archs4_performance_for_boxplots.csv"))
write_csv(summary_table, file.path(out_dir, "archs4_performance_summary.csv"))
write_csv(stats_by_panel, file.path(out_dir, "archs4_paired_wilcoxon_stats.csv"))

caption <- paste(
  "Statistics: paired one-sided Wilcoxon signed-rank tests on fold-level scores, testing FlowTransOP better than each baseline; Holm-adjusted within each panel.",
  paste0("Cycle per-sample files: ", length(cycle_ps_files),
         " | orthologue files: ", length(orth_files),
         " | orthologue summary files: ", length(orth_summary_files),
         " | MMD files: ", length(mmd_files))
)

p_main <- ggplot(cycle_plot_data, aes(x = model_label, y = score, fill = as.character(model_type))) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_boxplot(width = 0.66, outlier.shape = NA, alpha = 0.84, color = "grey20") +
  geom_point(
    position = position_jitter(width = 0.10, height = 0, seed = 11),
    size = 1.45,
    alpha = 0.75,
    color = "grey15"
  ) +
  geom_segment(
    data = cycle_stats_labels %>% filter(n_refs > 1),
    aes(x = xmin, xend = xmax, y = y, yend = y),
    inherit.aes = FALSE,
    linewidth = 0.28,
    color = "grey25"
  ) +
  geom_segment(
    data = cycle_stats_labels %>% filter(n_refs > 1),
    aes(x = xmin, xend = xmin, y = y, yend = y_tip),
    inherit.aes = FALSE,
    linewidth = 0.28,
    color = "grey25"
  ) +
  geom_segment(
    data = cycle_stats_labels %>% filter(n_refs > 1),
    aes(x = xmax, xend = xmax, y = y, yend = y_tip),
    inherit.aes = FALSE,
    linewidth = 0.28,
    color = "grey25"
  ) +
  geom_text(
    data = cycle_stats_labels %>% filter(n_refs > 1),
    aes(x = xmid, y = y, label = label),
    inherit.aes = FALSE,
    vjust = -0.18,
    size = 3.6,
    color = "grey20"
  ) +
  geom_text(
    data = cycle_stats_labels %>% filter(n_refs == 1),
    aes(x = x, y = y, label = label),
    inherit.aes = FALSE,
    vjust = -0.18,
    size = 4.5,
    color = "grey20"
  ) +
  facet_grid(metric ~ species, scales = "free_y") +
  scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels], drop = FALSE) +
  scale_y_continuous(limits = c(NA, 1.2), n.breaks = 5, expand = expansion(mult = c(0.06, 0.20))) +
  labs(
    title = "ARCHS4 FlowTransOP Cycle Consistency",
    subtitle = "All permutation ablations are shown; through-species-only permutations can preserve the home-space encoder/decoder reconstruction path",
    x = NULL,
    y = "Pearson correlation",
    fill = NULL,
    caption = paste(caption, "Significance: * p <= 0.05, ** p <= 0.01, *** p <= 0.001; ns otherwise.")
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 17),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    plot.caption = element_text(hjust = 0, size = 9.5, color = "grey35"),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    strip.text = element_text(face = "bold", size = 13),
    axis.title = element_text(size = 15),
    axis.text = element_text(size = 12),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 12),
    legend.position = "bottom",
    plot.margin = margin(8, 22, 8, 8)
  )

ggsave(file.path(out_dir, "archs4_flowtransop_cycle_boxplots.png"), p_main, width = 14, height = 12, dpi = 300)
ggsave(file.path(out_dir, "archs4_flowtransop_cycle_boxplots.pdf"), p_main, width = 14, height = 12)

if (nrow(orth_plot_data) > 0) {
  p_orth <- ggplot(orth_plot_data, aes(x = model_label, y = score, fill = as.character(model_type))) +
    geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
    geom_boxplot(width = 0.66, outlier.shape = NA, alpha = 0.84, color = "grey20") +
    geom_point(
      position = position_jitter(width = 0.10, height = 0, seed = 12),
      size = 1.45,
      alpha = 0.75,
      color = "grey15"
    ) +
    geom_segment(
      data = orth_stats_labels %>% filter(n_refs > 1),
    aes(x = xmin, xend = xmax, y = y, yend = y),
    inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_segment(
      data = orth_stats_labels %>% filter(n_refs > 1),
      aes(x = xmin, xend = xmin, y = y, yend = y_tip),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_segment(
      data = orth_stats_labels %>% filter(n_refs > 1),
      aes(x = xmax, xend = xmax, y = y, yend = y_tip),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_text(
      data = orth_stats_labels %>% filter(n_refs > 1),
      aes(x = xmid, y = y, label = label),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 4,
      color = "grey20"
    ) +
    geom_text(
      data = orth_stats_labels %>% filter(n_refs == 1),
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 5,
      color = "grey20"
    ) +
    facet_grid(metric ~ species, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels], drop = FALSE) +
    scale_y_continuous(limits = c(NA, 1.2), n.breaks = 5) +
    labs(
      title = "ARCHS4 FlowTransOP Orthologue-Mediated Evaluation",
      subtitle = "All permutation ablations are shown; partial source/target permutations isolate one component and may not fully destroy orthologue signal",
      x = NULL,
      y = "Pearson correlation",
      fill = NULL,
      caption = "Statistics: paired one-sided Wilcoxon signed-rank tests on fold-level scores, testing higher FlowTransOP performance; Holm-adjusted within panel. * p <= 0.05, ** p <= 0.01, *** p <= 0.001; ns otherwise."
    ) +
    coord_cartesian(clip = "off") +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(size = 10, color = "grey30"),
      plot.caption = element_text(hjust = 0, size = 8, color = "grey35"),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      strip.text = element_text(face = "bold"),
      axis.title = element_text(size = 13.7),
      axis.text = element_text(size = 11.1),
      axis.text.x = element_text(angle = 30, hjust = 1, size = 11.1),
      legend.position = "bottom",
      plot.margin = margin(8, 22, 8, 8)
    )
  ggsave(file.path(out_dir, "orthologue_evaluation_boxplots.png"), p_orth, width = 11, height = 5.5, dpi = 300)
  ggsave(file.path(out_dir, "orthologue_evaluation_boxplots.pdf"), p_orth, width = 11, height = 5.5)
}

if (nrow(mmd) > 0) {
  mmd_stats <- mmd %>%
    filter(model_type %in% c("FlowTransOP", "permuted_both")) %>%
    group_by(direction, species) %>%
    group_modify(~ {
      flow <- .x %>% filter(model_type == "FlowTransOP") %>% select(fold, flow = mmd)
      bind_rows(lapply(c("permuted_both"), function(base_name) {
        base <- .x %>% filter(model_type == base_name) %>% select(fold, baseline_score = mmd)
        paired <- inner_join(flow, base, by = "fold")
        p_val <- if (nrow(paired) >= 3) {
          tryCatch(
            wilcox.test(paired$flow, paired$baseline_score, paired = TRUE,
                        alternative = "less", exact = FALSE)$p.value,
            error = function(e) NA_real_
          )
        } else NA_real_
        tibble(
          baseline = base_name,
          n_folds = nrow(paired),
          mean_flow = mean(paired$flow, na.rm = TRUE),
          mean_baseline = mean(paired$baseline_score, na.rm = TRUE),
          mean_diff = mean(paired$flow - paired$baseline_score, na.rm = TRUE),
          p_value = p_val
        )
      })) %>% mutate(p_adj_holm = p.adjust(p_value, method = "holm"))
    }) %>%
    ungroup()
  write_csv(mmd_stats, file.path(out_dir, "mmd_paired_wilcoxon_stats.csv"))

  mmd_labels <- mmd_stats %>%
    left_join(
      mmd %>%
        filter(model_type %in% c("FlowTransOP", "permuted_both")) %>%
        group_by(direction, species) %>%
        summarise(y_min = min(mmd, na.rm = TRUE), y_max = max(mmd, na.rm = TRUE), .groups = "drop"),
      by = c("direction", "species")
    ) %>%
    group_by(direction, species) %>%
    mutate(
      baseline_label = factor(model_labels[baseline], levels = model_labels[c("FlowTransOP", "permuted_both")]),
      xmin = match(model_labels["FlowTransOP"], model_labels[c("FlowTransOP", "permuted_both")]),
      xmax = match(as.character(baseline_label), model_labels[c("FlowTransOP", "permuted_both")]),
      y = y_max + (row_number() * 0.12 * pmax(y_max - y_min, 0.1)),
      y_tip = y - 0.025 * pmax(y_max - y_min, 0.1),
      xmid = (xmin + xmax) / 2,
      label = p_stars(p_adj_holm)
    ) %>%
    ungroup()

  mmd_plot_data <- mmd %>%
    filter(model_type %in% c("FlowTransOP", "permuted_both")) %>%
    mutate(
      model_label = factor(model_labels[model_type], levels = model_labels[c("FlowTransOP", "permuted_both")]),
      model_type = factor(model_type, levels = c("FlowTransOP", "permuted_both"))
    )

  p_mmd <- ggplot(mmd_plot_data, aes(x = model_label, y = mmd, fill = model_type)) +
    geom_boxplot(width = 0.66, outlier.shape = NA, alpha = 0.84, color = "grey20") +
    geom_point(
      position = position_jitter(width = 0.10, height = 0, seed = 21),
      size = 1.8,
      alpha = 0.8,
      color = "grey15"
    ) +
    geom_segment(
      data = mmd_labels,
      aes(x = xmin, xend = xmax, y = y, yend = y),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_segment(
      data = mmd_labels,
      aes(x = xmin, xend = xmin, y = y, yend = y_tip),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_segment(
      data = mmd_labels,
      aes(x = xmax, xend = xmax, y = y, yend = y_tip),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_text(
      data = mmd_labels,
      aes(x = xmid, y = y, label = label),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 4.5,
      color = "grey20"
    ) +
    facet_wrap(~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = c("FlowTransOP", "permuted_both"),
                      labels = model_labels[c("FlowTransOP", "permuted_both")], drop = FALSE) +
    scale_y_continuous(breaks = mmd_breaks, expand = expansion(mult = c(0.08, 0.34))) +
    labs(
      title = "Latent-Space MMD",
      subtitle = "Lower MMD is better: translated source latents should match target species latents",
      x = NULL,
      y = expression(MMD^2),
      fill = NULL,
      caption = "Latent-space MMD excludes the untranslated-source comparison because FlowTransOP uses direction-specific, non-global latent spaces. Statistics: paired one-sided Wilcoxon tests for lower FlowTransOP MMD with Holm correction; * p <= 0.05, ** p <= 0.01, *** p <= 0.001."
    ) +
    coord_cartesian(clip = "off") +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      plot.caption = element_text(hjust = 0, size = 8, color = "grey35"),
      axis.title = element_text(size = 13.7),
      axis.text = element_text(size = 11.1),
      axis.text.x = element_text(angle = 25, hjust = 1, size = 11.1),
      legend.position = "bottom",
      plot.margin = margin(8, 22, 8, 8)
    )
  ggsave(file.path(out_dir, "mmd_summary_lower_is_better.png"), p_mmd, width = 9.5, height = 6.2, dpi = 300)
  ggsave(file.path(out_dir, "mmd_summary_lower_is_better.pdf"), p_mmd, width = 9.5, height = 6.2)
}

if (nrow(expression_mmd) > 0) {
  expr_levels <- c("validation_orthologues", "FlowTransOP", "permuted_human", "permuted_mouse", "permuted_both")
  expr_plot_data <- expression_mmd %>%
    filter(model_type %in% expr_levels) %>%
    mutate(
      direction = factor(direction, levels = c("Mouse to human", "Human to mouse")),
      feature_set = factor(feature_set, levels = c("Orthologues", "All target genes")),
      model_type = factor(model_type, levels = expr_levels),
      model_label = factor(model_labels[as.character(model_type)], levels = model_labels[expr_levels])
    )

  expr_stats <- expr_plot_data %>%
    group_by(direction, species, feature_set) %>%
    group_modify(~ {
      available <- as.character(unique(.x$model_type))
      baselines <- setdiff(intersect(expr_levels, available), "FlowTransOP")
      paired_wilcox_against_flow(.x, baselines = baselines, value_col = "mmd", lower_better = TRUE)
    }) %>%
    ungroup() %>%
    mutate(
      baseline_label = factor(model_labels[baseline], levels = model_labels[expr_levels])
    )
  write_csv(expr_stats, file.path(out_dir, "expression_mmd_paired_wilcoxon_stats.csv"))

  expr_labels <- expr_stats %>%
    filter(!is.na(p_adj_holm)) %>%
    left_join(
      expr_plot_data %>%
        group_by(direction, species, feature_set) %>%
        summarise(y_min = min(mmd, na.rm = TRUE), y_max = max(mmd, na.rm = TRUE), .groups = "drop"),
      by = c("direction", "species", "feature_set")
    ) %>%
    group_by(direction, species, feature_set) %>%
    arrange(baseline, .by_group = TRUE) %>%
    mutate(
      xmin = match(model_labels["FlowTransOP"], model_labels[expr_levels]),
      xmax = match(as.character(baseline_label), model_labels[expr_levels]),
      y = y_max + (row_number() * 0.12 * pmax(y_max - y_min, 0.1)),
      y_tip = y - 0.025 * pmax(y_max - y_min, 0.1),
      xmid = (xmin + xmax) / 2,
      label = p_stars(p_adj_holm)
    ) %>%
    ungroup()

  p_expr_mmd <- ggplot(expr_plot_data, aes(x = model_label, y = mmd, fill = as.character(model_type))) +
    geom_boxplot(width = 0.66, outlier.shape = NA, alpha = 0.84, color = "grey20") +
    geom_point(
      position = position_jitter(width = 0.10, height = 0, seed = 31),
      size = 1.8,
      alpha = 0.8,
      color = "grey15"
    ) +
    geom_segment(
      data = expr_labels,
      aes(x = xmin, xend = xmax, y = y, yend = y),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_segment(
      data = expr_labels,
      aes(x = xmin, xend = xmin, y = y, yend = y_tip),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_segment(
      data = expr_labels,
      aes(x = xmax, xend = xmax, y = y, yend = y_tip),
      inherit.aes = FALSE,
      linewidth = 0.35,
      color = "grey25"
    ) +
    geom_text(
      data = expr_labels,
      aes(x = xmid, y = y, label = label),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 4,
      color = "grey20"
    ) +
    facet_grid(feature_set ~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = expr_levels, labels = model_labels[expr_levels], drop = FALSE) +
    scale_y_continuous(breaks = mmd_breaks, expand = expansion(mult = c(0.08, 0.42))) +
    labs(
      title = "Expression-Space MMD",
      subtitle = "Lower MMD is better: translated source expression should match target validation expression",
      x = NULL,
      y = expression(MMD^2),
      fill = NULL,
      caption = "Validation orthologues is the same uncorrected source-vs-target validation baseline on matched orthologue genes; it is repeated in all-target-gene panels only as a visual reference. Statistics compare FlowTransOP to each available baseline within panel using paired Wilcoxon tests with Holm correction."
    ) +
    coord_cartesian(clip = "off") +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12, color = "grey30"),
      plot.caption = element_text(hjust = 0, size = 9.5, color = "grey35"),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      strip.text = element_text(face = "bold", size = 12),
      axis.title = element_text(size = 13.7),
      axis.text = element_text(size = 11.1),
      axis.text.x = element_text(angle = 30, hjust = 1, size = 11.1),
      legend.position = "bottom",
      plot.margin = margin(8, 26, 8, 8)
    )
  ggsave(file.path(out_dir, "expression_mmd_lower_is_better.png"), p_expr_mmd, width = 13, height = 8.5, dpi = 300)
  ggsave(file.path(out_dir, "expression_mmd_lower_is_better.pdf"), p_expr_mmd, width = 13, height = 8.5)
}

message("Wrote figures and summaries to: ", out_dir)
message("Rows plotted in main performance figure: ", nrow(performance))
message("Orthologue panels included: ", ifelse(length(orth_files) > 0, "yes", "no"))
message("MMD panels included: ", ifelse(length(mmd_files) > 0, "yes", "no"))
message("Expression-space MMD panels included: ", ifelse(length(expression_mmd_files) > 0, "yes", "no"))
