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

score_dir <- get_arg(
  "--score_dir",
  file.path("..", "archs4", "evaluation", "liver_mas_fibrosis_final_expression_mean")
)
score_dir <- normalizePath(score_dir, mustWork = TRUE)
out_dir <- get_arg("--out_dir", file.path(score_dir, "figures"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

endpoint_labels <- c(
  nas_score = "NAS/MAS score",
  fibrosis_stage = "Fibrosis stage"
)

treatment_cols <- c(
  `WT Control chow+placebo` = "#59A14F",
  `Nlrp3 Control chow+placebo` = "#4C78A8",
  `GS-444217` = "#E15759",
  `CDAA-HFD vehicle` = "#4C78A8",
  Chow = "#59A14F",
  Lanifibranor = "#E15759"
)

treatment_labels <- c(
  `WT Control chow+placebo` = "WT control\nchow + placebo",
  `Nlrp3 Control chow+placebo` = "Nlrp3 control\nchow + placebo",
  `GS-444217` = "GS-444217",
  `CDAA-HFD vehicle` = "CDAA-HFD\nvehicle",
  Chow = "Chow",
  Lanifibranor = "Lanifibranor"
)

dataset_specs <- list(
  mouse_nlrp3 = list(
    label = "GSE140742 Nlrp3A350V",
    file_slug = "gse140742_nlrp3a350v",
    treatment_levels = c("WT Control chow+placebo", "Nlrp3 Control chow+placebo", "GS-444217"),
    healthy = "WT Control chow+placebo",
    control = "Nlrp3 Control chow+placebo",
    treatment = "GS-444217"
  ),
  mouse_cdaa_lanifibranor = list(
    label = "GSE269493 CDAA-HFD",
    file_slug = "gse269493_cdaa_hfd",
    treatment_levels = c("Chow", "CDAA-HFD vehicle", "Lanifibranor"),
    healthy = "Chow",
    control = "CDAA-HFD vehicle",
    treatment = "Lanifibranor"
  )
)

figure_specs <- list(
  hybrid = list(
    label = "Reconstructed translated PLSR with raw-orthologue reference",
    file_slug = "hybrid_reconstructed_translated_with_raw_orthologues",
    description = paste(
      "Hybrid figure: translated expression is scored with PLSR trained on",
      "ensemble-mean reconstructed human expression; raw mouse orthologues are",
      "scored with the observed-human orthologue PLSR."
    ),
    rows = tibble(
      model_family = c("raw_human_plsr", "reconstructed_human_plsr", "reconstructed_human_plsr"),
      scoring_space = c("raw_mouse_orthologues", "translated_all_genes", "translated_orthologues"),
      panel = c(
        "Raw mouse orthologues\nraw-human PLSR",
        "Translated all genes\nreconstructed-human PLSR",
        "Translated orthologues\nreconstructed-human PLSR"
      )
    )
  ),
  raw_human_plsr = list(
    label = "Raw-human PLSR",
    file_slug = "raw_human_plsr",
    description = "All panels are scored using PLSR models trained on observed human Govaere expression.",
    rows = tibble(
      model_family = "raw_human_plsr",
      scoring_space = c("translated_all_genes", "translated_orthologues", "raw_mouse_orthologues"),
      panel = c(
        "Translated all genes\nraw-human PLSR",
        "Translated orthologues\nraw-human PLSR",
        "Raw mouse orthologues\nraw-human PLSR"
      )
    )
  ),
  reconstructed_human_plsr = list(
    label = "Reconstructed-human PLSR",
    file_slug = "reconstructed_human_plsr",
    description = paste(
      "All panels are scored using PLSR models trained on ensemble-mean",
      "human autoencoder reconstructions."
    ),
    rows = tibble(
      model_family = "reconstructed_human_plsr",
      scoring_space = c("translated_all_genes", "translated_orthologues", "raw_mouse_orthologues"),
      panel = c(
        "Translated all genes\nreconstructed-human PLSR",
        "Translated orthologues\nreconstructed-human PLSR",
        "Raw mouse orthologues\nreconstructed-human PLSR"
      )
    )
  )
)

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

read_required_csv <- function(path) {
  if (!file.exists(path)) stop("Missing required file: ", path)
  read_csv(path, show_col_types = FALSE)
}

mouse_long_from_csv <- function(cohort_name) {
  path <- file.path(score_dir, paste0(cohort_name, "_final_expression_mean_plsr_scores.csv"))
  read_required_csv(path) %>%
    mutate(
      cohort = cohort_name,
      mouse_treatment = case_when(
        cohort == "mouse_nlrp3" & mouse_treatment == "Control chow+placebo" ~ "Nlrp3 Control chow+placebo",
        TRUE ~ mouse_treatment
      )
    ) %>%
    pivot_longer(
      cols = c(predicted_nas_score, predicted_fibrosis_stage),
      names_to = "endpoint",
      values_to = "score",
      names_pattern = "predicted_(nas_score|fibrosis_stage)"
    ) %>%
    mutate(endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels))
}

mouse_scores <- bind_rows(lapply(names(dataset_specs), mouse_long_from_csv))

figure_mouse_data <- function(figure_name, score_data = mouse_scores) {
  spec <- figure_specs[[figure_name]]
  score_data %>%
    inner_join(spec$rows, by = c("model_family", "scoring_space")) %>%
    mutate(
      figure_category = figure_name,
      figure_label = spec$label,
      figure_description = spec$description,
      panel = factor(panel, levels = spec$rows$panel)
    )
}

minmax_scale <- function(x, min_value, max_value) {
  ifelse(
    is.na(min_value) | is.na(max_value) | max_value <= min_value,
    0.5,
    (x - min_value) / (max_value - min_value)
  )
}

score_plot_data <- bind_rows(lapply(names(figure_specs), figure_mouse_data)) %>%
  group_by(figure_category, cohort, endpoint, panel) %>%
  mutate(
    score_type = "score",
    raw_plot_value = score,
    panel_min_score = min(score, na.rm = TRUE),
    panel_max_score = max(score, na.rm = TRUE),
    plot_value = minmax_scale(score, panel_min_score, panel_max_score),
    value_scale = "min_max_normalized_within_panel"
  ) %>%
  ungroup()

delta_plot_data <- score_plot_data %>%
  group_by(figure_category, cohort, endpoint, panel) %>%
  group_modify(~ {
    ds <- dataset_specs[[.y$cohort]]
    healthy_scores <- .x$score[.x$mouse_treatment == ds$healthy]
    healthy_mean <- if (length(healthy_scores) > 0 && any(!is.na(healthy_scores))) {
      mean(healthy_scores, na.rm = TRUE)
    } else {
      NA_real_
    }
    .x %>%
      mutate(
        score_type = "delta_from_healthy",
        healthy_baseline = ds$healthy,
        healthy_baseline_mean = healthy_mean,
        raw_plot_value = score - healthy_mean,
        plot_value = score - healthy_mean,
        value_scale = "original_score_delta_from_healthy"
      )
  }) %>%
  ungroup()

all_plot_data <- bind_rows(score_plot_data, delta_plot_data)

stats_for <- function(data, score_type_value) {
  data %>%
    filter(score_type == score_type_value) %>%
    group_by(figure_category, figure_label, figure_description, cohort, endpoint, panel) %>%
    group_modify(~ {
      ds <- dataset_specs[[.y$cohort]]
      panel <- .x %>% filter(!is.na(plot_value), !is.na(mouse_treatment))
      control_df <- panel %>% filter(mouse_treatment == ds$control)
      treatment_df <- panel %>% filter(mouse_treatment == ds$treatment)
      p_val <- if (nrow(control_df) > 0 && nrow(treatment_df) > 0) {
        tryCatch(
          wilcox.test(
            treatment_df$plot_value,
            control_df$plot_value,
            alternative = "less",
            exact = FALSE
          )$p.value,
          error = function(e) NA_real_
        )
      } else {
        NA_real_
      }
      y_min <- min(panel$plot_value, na.rm = TRUE)
      y_max <- max(panel$plot_value, na.rm = TRUE)
      span <- max(y_max - y_min, 0.25)
      if (score_type_value == "score") {
        y_bracket <- 1.08
        y_tip <- 1.03
        y_label <- 1.18
      } else {
        y_bracket <- y_max + 0.20 * span
        y_tip <- y_max + 0.10 * span
        y_label <- y_max + 0.36 * span
      }
      levels_x <- ds$treatment_levels
      description <- paste0(
        "One-sided Wilcoxon rank-sum test; alternative: ",
        ds$treatment, " score < ", ds$control,
        " score. Healthy group ", ds$healthy,
        ifelse(score_type_value == "delta_from_healthy",
               " is used to center deltas.",
        " is shown as a visual reference only. Scores are min-max normalized within each endpoint-by-panel before plotting and testing."),
        " Figure category: ", .y$figure_description
      )
      tibble(
        dataset_label = ds$label,
        score_type = score_type_value,
        healthy = ds$healthy,
        control = ds$control,
        treatment = ds$treatment,
        comparison = paste(ds$control, "vs", ds$treatment),
        alternative = paste(ds$treatment, "<", ds$control),
        n_control = nrow(control_df),
        n_treated = nrow(treatment_df),
        mean_control = mean(control_df$plot_value, na.rm = TRUE),
        mean_treated = mean(treatment_df$plot_value, na.rm = TRUE),
        mean_diff_treated_minus_control = mean_treated - mean_control,
        p_value = p_val,
        label = pretty_p_less(p_val),
        x_start = match(ds$control, levels_x),
        x_end = match(ds$treatment, levels_x),
        x_mid = mean(c(x_start, x_end)),
        y_bracket = y_bracket,
        y_tip = y_tip,
        y_label = y_label,
        value_scale = first(panel$value_scale),
        description = description
      )
    }) %>%
    ungroup()
}

mouse_stats <- bind_rows(
  stats_for(all_plot_data, "score"),
  stats_for(all_plot_data, "delta_from_healthy")
)

mouse_group_summary <- all_plot_data %>%
  group_by(
    figure_category, figure_label, figure_description, cohort, endpoint, panel,
    score_type, mouse_treatment
  ) %>%
  summarise(
    n_samples = n_distinct(sample_id),
    mean = mean(plot_value, na.rm = TRUE),
    median = median(plot_value, na.rm = TRUE),
    sd = sd(plot_value, na.rm = TRUE),
    q25 = quantile(plot_value, 0.25, na.rm = TRUE),
    q75 = quantile(plot_value, 0.75, na.rm = TRUE),
    value_scale = first(value_scale),
    raw_mean = mean(raw_plot_value, na.rm = TRUE),
    raw_median = median(raw_plot_value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    dataset_label = vapply(cohort, function(x) dataset_specs[[x]]$label, character(1)),
    description = paste0(
      "Group-level summary for plotted boxplot values. Score type: ", score_type,
      "; value scale: ", value_scale,
      "; figure category: ", figure_description,
      "; statistical comparisons are drug versus diseased control, with healthy as visual reference",
      ifelse(score_type == "delta_from_healthy", " and delta baseline.", ".")
    )
  )

write_csv(mouse_stats, file.path(out_dir, "final_mouse_boxplot_stat_tests.csv"))
write_csv(mouse_group_summary, file.path(out_dir, "final_mouse_boxplot_group_summaries.csv"))

plot_mouse_category <- function(cohort_name, figure_name, score_type_value) {
  ds <- dataset_specs[[cohort_name]]
  fig <- figure_specs[[figure_name]]
  plot_data <- all_plot_data %>%
    filter(cohort == cohort_name, figure_category == figure_name, score_type == score_type_value) %>%
    mutate(mouse_treatment = factor(mouse_treatment, levels = ds$treatment_levels))
  stat_data <- mouse_stats %>%
    filter(cohort == cohort_name, figure_category == figure_name, score_type == score_type_value)
  if (nrow(plot_data) == 0) return(invisible(NULL))

  y_label <- ifelse(
    score_type_value == "delta_from_healthy",
    "Predicted score delta from healthy control",
    "Min-max normalized predicted score"
  )
  title_suffix <- ifelse(score_type_value == "delta_from_healthy", "deltas", "scores")
  subtitle <- paste0(
    "Drug comparison: ", ds$treatment, " vs ", ds$control,
    ". Healthy group ", ds$healthy,
    ifelse(score_type_value == "delta_from_healthy",
           " centers the delta values.",
           " is a visual reference.")
  )

  facet_layer <- if (score_type_value == "delta_from_healthy") {
    facet_wrap(vars(endpoint, panel), scales = "free_y", ncol = n_distinct(plot_data$panel))
  } else {
    facet_grid(endpoint ~ panel, scales = "fixed")
  }
  y_scale_layer <- if (score_type_value == "delta_from_healthy") {
    scale_y_continuous(expand = expansion(mult = c(0.08, 0.46)))
  } else {
    scale_y_continuous(breaks = c(0, 0.5, 1), expand = expansion(mult = c(0.06, 0.26)))
  }

  p <- ggplot(plot_data, aes(x = mouse_treatment, y = plot_value, fill = mouse_treatment)) +
    {if (score_type_value == "delta_from_healthy") {
      geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", color = "grey50")
    }} +
    geom_boxplot(width = 0.58, outlier.shape = NA, alpha = 0.75, color = "grey25") +
    geom_jitter(width = 0.12, size = 1.85, alpha = 0.78, color = "grey15") +
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
    facet_layer +
    y_scale_layer +
    scale_fill_manual(values = treatment_cols, guide = "none", drop = FALSE) +
    scale_x_discrete(labels = treatment_labels, drop = FALSE) +
    coord_cartesian(clip = "off") +
    labs(
      x = NULL,
      y = y_label,
      title = paste0(ds$label, ": ", fig$label, " ", title_suffix),
      subtitle = subtitle
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 8.5),
      strip.text.x = element_text(size = 8.3),
      plot.margin = margin(8, 12, 10, 8)
    )

  file_name <- paste0(ds$file_slug, "_", fig$file_slug, "_", score_type_value, ".png")
  ggsave(file.path(out_dir, file_name), p, width = 12.2, height = 7.2, dpi = 300)
}

for (cohort_name in names(dataset_specs)) {
  for (figure_name in names(figure_specs)) {
    plot_mouse_category(cohort_name, figure_name, "score")
    plot_mouse_category(cohort_name, figure_name, "delta_from_healthy")
  }
}

read_loocv <- function(model_family, all_path, orth_path) {
  bind_rows(
    read_required_csv(file.path(score_dir, all_path)),
    read_required_csv(file.path(score_dir, orth_path))
  ) %>%
    mutate(
      model_family = model_family,
      feature_space = recode(
        feature_space,
        all_human_genes = "All genes",
        orthologues = "Orthologues",
        .default = feature_space
      )
    ) %>%
    pivot_longer(
      cols = c(observed_nas_score, observed_fibrosis_stage,
               predicted_loocv_nas_score, predicted_loocv_fibrosis_stage),
      names_to = c(".value", "endpoint"),
      names_pattern = "(observed|predicted_loocv)_(nas_score|fibrosis_stage)"
    ) %>%
    rename(predicted = predicted_loocv) %>%
    mutate(endpoint = factor(endpoint, levels = names(endpoint_labels), labels = endpoint_labels))
}

loocv_raw <- read_loocv(
  "raw_human_plsr",
  "human_govaere_raw_plsr_loocv_all_genes.csv",
  "human_govaere_raw_plsr_loocv_orthologues.csv"
)
loocv_recon <- read_loocv(
  "reconstructed_human_plsr",
  "human_govaere_reconstructed_plsr_loocv_all_genes.csv",
  "human_govaere_reconstructed_plsr_loocv_orthologues.csv"
)
loocv_all <- bind_rows(loocv_raw, loocv_recon)

spearman_stats <- function(data) {
  data %>%
    group_by(model_family, feature_space, endpoint) %>%
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
      label = paste0("Spearman rho = ", sprintf("%.2f", rho), "\n", pretty_p(p_value)),
      description = case_when(
        model_family == "raw_human_plsr" ~ paste(
          "LOOCV performance for PLSR trained on observed Govaere human expression;",
          "each held-out sample is predicted by a model trained without that sample."
        ),
        TRUE ~ paste(
          "LOOCV performance for PLSR trained on ensemble-mean reconstructed Govaere human expression;",
          "each held-out sample is predicted by a model trained without that sample."
        )
      )
    )
}

loocv_summary <- spearman_stats(loocv_all)
write_csv(loocv_summary, file.path(out_dir, "final_human_loocv_performance_summary.csv"))

plot_loocv <- function(model_family, title, file_name, color) {
  plot_data <- loocv_all %>% filter(model_family == .env$model_family)
  stat_data <- loocv_summary %>% filter(model_family == .env$model_family)
  p <- ggplot(plot_data, aes(x = observed, y = predicted)) +
    geom_abline(slope = 1, intercept = 0, linewidth = 0.45, linetype = "dashed", color = "grey45") +
    geom_point(size = 1.95, alpha = 0.82, color = color) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, color = "grey20") +
    geom_text(
      data = stat_data,
      aes(x = -Inf, y = Inf, label = label),
      inherit.aes = FALSE,
      hjust = -0.06,
      vjust = 1.12,
      size = 3.2,
      lineheight = 0.95,
      color = "grey15"
    ) +
    facet_grid(endpoint ~ feature_space, scales = "free") +
    labs(
      x = "Observed human score",
      y = "LOOCV predicted score",
      title = title,
      subtitle = "Each point is predicted by a PLSR model trained without that sample"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      panel.grid.minor = element_blank(),
      aspect.ratio = 0.95
    )
  ggsave(file.path(out_dir, file_name), p, width = 8.4, height = 7.2, dpi = 300)
}

plot_loocv(
  "raw_human_plsr",
  "Human Govaere raw-human PLSR leave-one-out performance",
  "human_govaere_raw_human_plsr_loocv_scatter.png",
  "#2C7FB8"
)
plot_loocv(
  "reconstructed_human_plsr",
  "Human Govaere reconstructed-human PLSR leave-one-out performance",
  "human_govaere_reconstructed_human_plsr_loocv_scatter.png",
  "#1B9E77"
)

message("Wrote final expression-mean PNG figures and CSV summaries to: ", out_dir)
