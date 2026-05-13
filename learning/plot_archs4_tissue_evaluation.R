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
out_dir <- file.path(eval_dir, "figures_tissue")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

model_levels <- c("FlowTransOP", "permuted_both")
model_labels <- c(
  FlowTransOP = "FlowTransOP",
  permuted_both = "Permuted both"
)
model_cols <- c(
  FlowTransOP = "#1B9E77",
  permuted_both = "#666666"
)
resolution_labels <- c(
  fine_tissue = "Fine tissues",
  tissue_group = "Tissue groups"
)

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

fmt_common <- function(x) {
  if (!"label_resolution" %in% names(x)) {
    x$label_resolution <- "tissue_group"
  }
  x %>%
    mutate(
      direction = recode(direction, h2m = "Human to mouse", m2h = "Mouse to human"),
      direction = factor(direction, levels = c("Mouse to human", "Human to mouse")),
      label_resolution = coalesce(as.character(label_resolution), "tissue_group"),
      label_resolution_label = factor(
        resolution_labels[label_resolution],
        levels = resolution_labels[c("tissue_group", "fine_tissue")]
      ),
      model_type = as.character(model_type),
      model_label = factor(model_labels[model_type], levels = model_labels[model_levels])
    ) %>%
    filter(model_type %in% model_levels, !is.na(model_label))
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

paired_flow_stats <- function(data, group_cols) {
  if (nrow(data) == 0) return(tibble())
  data %>%
    group_by_at(group_cols) %>%
    group_modify(~ {
      flow <- .x %>% filter(model_type == "FlowTransOP") %>% select(fold, flow = value)
      base <- .x %>% filter(model_type == "permuted_both") %>% select(fold, baseline = value)
      paired <- inner_join(flow, base, by = "fold")
      lower_better <- isTRUE(.x$lower_better[1])
      alt <- if (lower_better) "less" else "greater"
      p_val <- if (nrow(paired) >= 3) {
        tryCatch(
          wilcox.test(paired$flow, paired$baseline, paired = TRUE,
                      alternative = alt, exact = FALSE)$p.value,
          error = function(e) NA_real_
        )
      } else NA_real_
      tibble(
        n_folds = nrow(paired),
        mean_flow = mean(paired$flow, na.rm = TRUE),
        mean_permuted_both = mean(paired$baseline, na.rm = TRUE),
        mean_diff = mean(paired$flow - paired$baseline, na.rm = TRUE),
        median_diff = median(paired$flow - paired$baseline, na.rm = TRUE),
        lower_better = lower_better,
        p_value = p_val
      )
    }) %>%
    ungroup() %>%
    group_by_at(group_cols) %>%
    mutate(p_adj_holm = p.adjust(p_value, method = "holm"),
           stars = p_stars(p_adj_holm)) %>%
    ungroup()
}

single_ref_labels <- function(data, group_cols) {
  stats <- paired_flow_stats(data, group_cols)
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
      model_label = factor(model_labels["permuted_both"], levels = model_labels[model_levels]),
      y = y_max + 0.12 * pmax(y_max - y_min, 0.1)
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
      axis.text.x = element_text(angle = 25, hjust = 1),
      legend.position = "bottom"
    )
}

tissue_summary <- read_folded("^tissue_summary_fold[0-9]+\\.csv$") %>% fmt_common() %>%
  filter(label_resolution == "tissue_group")
reference <- read_folded("^tissue_reference_mapping_fold[0-9]+\\.csv$") %>% fmt_common() %>%
  filter(label_resolution == "tissue_group")
linear <- read_folded("^tissue_linear_probe_fold[0-9]+\\.csv$") %>% fmt_common() %>%
  filter(label_resolution == "tissue_group")
purity <- read_folded("^tissue_knn_purity_fold[0-9]+\\.csv$") %>% fmt_common() %>%
  filter(label_resolution == "tissue_group")
variance_summary <- read_folded("^tissue_variance_summary_fold[0-9]+\\.csv$") %>% fmt_common()
expr_centroid <- read_folded("^tissue_expression_centroid_summary_fold[0-9]+\\.csv$") %>% fmt_common() %>%
  filter(label_resolution == "tissue_group")
expr_knn <- read_folded("^tissue_expression_knn_summary_fold[0-9]+\\.csv$") %>% fmt_common() %>%
  filter(label_resolution == "tissue_group")

if (nrow(tissue_summary) == 0) {
  stop("No tissue evaluation CSVs found in ", eval_dir)
}

latent_centroid_long <- tissue_summary %>%
  select(fold, direction, model_type, model_label,
         mean_distance, mean_cosine_to_correct, cosine_nn_accuracy, centroid_pearson_mean) %>%
  pivot_longer(c(mean_distance, mean_cosine_to_correct, cosine_nn_accuracy, centroid_pearson_mean),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = recode(metric,
      mean_distance = "Centroid distance",
      mean_cosine_to_correct = "Centroid cosine",
      cosine_nn_accuracy = "Cosine top-1 tissue accuracy",
      centroid_pearson_mean = "Mean latent-dim Pearson"
    ),
    metric_label = factor(metric_label, levels = c(
      "Centroid distance", "Centroid cosine", "Cosine top-1 tissue accuracy", "Mean latent-dim Pearson"
    )),
    lower_better = metric == "mean_distance"
  )

latent_centroid_stats <- paired_flow_stats(latent_centroid_long, c("direction", "metric_label"))
latent_centroid_labels <- single_ref_labels(latent_centroid_long, c("direction", "metric_label"))
write_csv(latent_centroid_long, file.path(out_dir, "tissue_latent_centroid_values.csv"))
write_csv(latent_centroid_stats, file.path(out_dir, "tissue_latent_centroid_paired_stats.csv"))

p_latent_centroid <- ggplot(latent_centroid_long, aes(model_label, value, fill = model_type)) +
  geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
  geom_point(position = position_jitter(width = 0.08, height = 0, seed = 41),
             size = 1.7, alpha = 0.78, color = "grey15") +
  geom_text(
    data = latent_centroid_labels,
    aes(x = model_label, y = y, label = stars),
    inherit.aes = FALSE,
    vjust = -0.15,
    size = 5,
    color = "grey20"
  ) +
  facet_grid(metric_label ~ direction, scales = "free_y") +
  scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels]) +
  scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
  labs(
    title = "Tissue Diagnostics in Latent Space",
    subtitle = "Centroid metrics summarize whether translated source tissue centroids land near real target tissue centroids",
    x = NULL,
    y = "Metric value",
    fill = NULL,
    caption = "Lower is better only for centroid distance; higher is better for the remaining panels."
  ) +
  coord_cartesian(clip = "off") +
  theme_archs4(12)
save_both(p_latent_centroid, "tissue_latent_centroid_summary", 11.5, 9)

classifier_long <- bind_rows(
  reference %>% mutate(method_label = "kNN reference mapping"),
  linear %>% mutate(method_label = "Linear probe")
) %>%
  select(fold, direction, model_type, model_label, method_label,
         macro_f1, balanced_accuracy, accuracy) %>%
  pivot_longer(c(macro_f1, balanced_accuracy, accuracy), names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = recode(metric,
      macro_f1 = "Macro F1",
      balanced_accuracy = "Balanced accuracy",
      accuracy = "Accuracy"
    ),
    metric_label = factor(metric_label, levels = c("Macro F1", "Balanced accuracy", "Accuracy")),
    method_label = factor(method_label, levels = c("kNN reference mapping", "Linear probe")),
    lower_better = FALSE
  )

if (nrow(classifier_long) > 0) {
  classifier_stats <- paired_flow_stats(classifier_long, c("direction", "method_label", "metric_label"))
  classifier_labels <- single_ref_labels(classifier_long, c("direction", "method_label", "metric_label"))
  write_csv(classifier_long, file.path(out_dir, "tissue_classifier_values.csv"))
  write_csv(classifier_stats, file.path(out_dir, "tissue_classifier_paired_stats.csv"))
  p_classifier <- ggplot(classifier_long, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 42),
               size = 1.6, alpha = 0.78, color = "grey15") +
    geom_text(
      data = classifier_labels,
      aes(x = model_label, y = y, label = stars),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 4.8,
      color = "grey20"
    ) +
    facet_grid(metric_label ~ method_label + direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "Cross-Species Tissue Label Transfer in Latent Space",
      subtitle = "Classifiers are trained on real target latents and evaluated on translated source latents",
      x = NULL,
      y = "Score",
      fill = NULL,
      caption = "Higher is better. Chance balanced accuracy is roughly 1 / number of common tissues."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(11)
  save_both(p_classifier, "tissue_latent_label_transfer", 15.5, 8.8)
}

purity_long <- purity %>%
  select(fold, direction, model_type, model_label,
         mean_same_tissue_fraction, top1_accuracy, topk_accuracy) %>%
  pivot_longer(c(mean_same_tissue_fraction, top1_accuracy, topk_accuracy),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = recode(metric,
      mean_same_tissue_fraction = "Mean same-tissue fraction",
      top1_accuracy = "Top-1 tissue accuracy",
      topk_accuracy = "Top-k tissue accuracy"
    ),
    metric_label = factor(metric_label, levels = c(
      "Mean same-tissue fraction", "Top-1 tissue accuracy", "Top-k tissue accuracy"
    )),
    lower_better = FALSE
  )

if (nrow(purity_long) > 0) {
  purity_stats <- paired_flow_stats(purity_long, c("direction", "metric_label"))
  purity_labels <- single_ref_labels(purity_long, c("direction", "metric_label"))
  write_csv(purity_long, file.path(out_dir, "tissue_latent_knn_purity_values.csv"))
  write_csv(purity_stats, file.path(out_dir, "tissue_latent_knn_purity_paired_stats.csv"))
  p_purity <- ggplot(purity_long, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 43),
               size = 1.7, alpha = 0.78, color = "grey15") +
    geom_text(
      data = purity_labels,
      aes(x = model_label, y = y, label = stars),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 5,
      color = "grey20"
    ) +
    facet_grid(metric_label ~ direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "kNN Tissue Purity in Latent Space",
      subtitle = "For each translated sample, nearest real target neighbors should share the same tissue label",
      x = NULL,
      y = "Score",
      fill = NULL,
      caption = "Higher is better."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(12)
  save_both(p_purity, "tissue_latent_knn_purity", 11.5, 7.5)
}

variance_long <- variance_summary %>%
  select(fold, direction, label_resolution, label_resolution_label, model_type, model_label,
         weighted_partial_r2_tissue, weighted_partial_r2_domain) %>%
  pivot_longer(c(weighted_partial_r2_tissue, weighted_partial_r2_domain),
               names_to = "component", values_to = "value") %>%
  mutate(
    component_label = recode(component,
      weighted_partial_r2_tissue = "Tissue",
      weighted_partial_r2_domain = "Source/target domain"
    ),
    component_label = factor(component_label, levels = c("Tissue", "Source/target domain")),
    lower_better = component == "weighted_partial_r2_domain"
  )

if (nrow(variance_long) > 0) {
  variance_stats <- paired_flow_stats(
    variance_long %>% rename(metric_label = component_label),
    c("direction", "label_resolution_label", "metric_label")
  )
  variance_labels <- single_ref_labels(
    variance_long %>% rename(metric_label = component_label),
    c("direction", "label_resolution_label", "metric_label")
  ) %>%
    rename(component_label = metric_label)
  write_csv(variance_long, file.path(out_dir, "tissue_variance_partition_values.csv"))
  write_csv(variance_stats, file.path(out_dir, "tissue_variance_partition_paired_stats.csv"))
  p_variance <- ggplot(variance_long, aes(model_label, value, fill = component_label)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20",
                 position = position_dodge(width = 0.75)) +
    geom_point(aes(group = component_label),
               position = position_jitterdodge(jitter.width = 0.06, dodge.width = 0.75, seed = 44),
               size = 1.6, alpha = 0.78, color = "grey15") +
    geom_text(
      data = variance_labels,
      aes(x = model_label, y = y, label = stars, group = component_label),
      inherit.aes = FALSE,
      position = position_dodge(width = 0.75),
      vjust = -0.15,
      size = 4.8,
      color = "grey20"
    ) +
    facet_grid(label_resolution_label ~ direction) +
    scale_fill_manual(values = c("Tissue" = "#4C78A8", "Source/target domain" = "#E15759")) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "Variance Partitioning of Latent PCs",
      subtitle = "Good translation preserves tissue signal while reducing source-vs-target/domain signal",
      x = NULL,
      y = "Weighted partial R2",
      fill = NULL
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(12)
  save_both(p_variance, "tissue_variance_partitioning", 11.5, 7.4)
}

expr_centroid_long <- expr_centroid %>%
  filter(metric == "cosine") %>%
  mutate(
    feature_set = recode(feature_set,
      all_target_genes = "All target genes",
      orthologues = "Orthologues"
    ),
    feature_set = factor(feature_set, levels = c("Orthologues", "All target genes"))
  ) %>%
  select(fold, direction, model_type, model_label, feature_set,
         top1_accuracy, mean_reciprocal_rank, mean_similarity_z) %>%
  pivot_longer(c(top1_accuracy, mean_reciprocal_rank, mean_similarity_z),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = recode(metric,
      top1_accuracy = "Top-1 tissue centroid accuracy",
      mean_reciprocal_rank = "Mean reciprocal rank",
      mean_similarity_z = "Correct-centroid z score"
    ),
    metric_label = factor(metric_label, levels = c(
      "Top-1 tissue centroid accuracy", "Mean reciprocal rank", "Correct-centroid z score"
    )),
    lower_better = FALSE
  )

if (nrow(expr_centroid_long) > 0) {
  expr_centroid_stats <- paired_flow_stats(expr_centroid_long, c("direction", "feature_set", "metric_label"))
  expr_centroid_labels <- single_ref_labels(expr_centroid_long, c("direction", "feature_set", "metric_label"))
  write_csv(expr_centroid_long, file.path(out_dir, "tissue_expression_centroid_cosine_values.csv"))
  write_csv(expr_centroid_stats, file.path(out_dir, "tissue_expression_centroid_cosine_paired_stats.csv"))
  p_expr_centroid <- ggplot(expr_centroid_long, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 45),
               size = 1.55, alpha = 0.78, color = "grey15") +
    geom_text(
      data = expr_centroid_labels,
      aes(x = model_label, y = y, label = stars),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 4.5,
      color = "grey20"
    ) +
    facet_grid(metric_label ~ feature_set + direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "Expression-Space Tissue Centroid Retrieval",
      subtitle = "Cosine similarity between translated tissue centroids and real target tissue centroids",
      x = NULL,
      y = "Score",
      fill = NULL,
      caption = "Higher is better. Top-1 chance is approximately 1 / number of common tissues."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(10.5)
  save_both(p_expr_centroid, "tissue_expression_centroid_retrieval", 16, 8.8)
}

expr_knn_long <- expr_knn %>%
  mutate(
    feature_set = recode(feature_set,
      all_target_genes = "All target genes",
      orthologues = "Orthologues"
    ),
    feature_set = factor(feature_set, levels = c("Orthologues", "All target genes"))
  ) %>%
  select(fold, direction, model_type, model_label, feature_set,
         mean_same_tissue_fraction, top1_accuracy, topk_accuracy) %>%
  pivot_longer(c(mean_same_tissue_fraction, top1_accuracy, topk_accuracy),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = recode(metric,
      mean_same_tissue_fraction = "Mean same-tissue fraction",
      top1_accuracy = "Top-1 tissue accuracy",
      topk_accuracy = "Top-k tissue accuracy"
    ),
    metric_label = factor(metric_label, levels = c(
      "Mean same-tissue fraction", "Top-1 tissue accuracy", "Top-k tissue accuracy"
    )),
    lower_better = FALSE
  )

if (nrow(expr_knn_long) > 0) {
  expr_knn_stats <- paired_flow_stats(expr_knn_long, c("direction", "feature_set", "metric_label"))
  expr_knn_labels <- single_ref_labels(expr_knn_long, c("direction", "feature_set", "metric_label"))
  write_csv(expr_knn_long, file.path(out_dir, "tissue_expression_knn_values.csv"))
  write_csv(expr_knn_stats, file.path(out_dir, "tissue_expression_knn_paired_stats.csv"))
  p_expr_knn <- ggplot(expr_knn_long, aes(model_label, value, fill = model_type)) +
    geom_boxplot(width = 0.62, outlier.shape = NA, alpha = 0.86, color = "grey20") +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 46),
               size = 1.55, alpha = 0.78, color = "grey15") +
    geom_text(
      data = expr_knn_labels,
      aes(x = model_label, y = y, label = stars),
      inherit.aes = FALSE,
      vjust = -0.15,
      size = 4.5,
      color = "grey20"
    ) +
    facet_grid(metric_label ~ feature_set + direction, scales = "free_y") +
    scale_fill_manual(values = model_cols, breaks = model_levels, labels = model_labels[model_levels]) +
    scale_y_continuous(n.breaks = 5, expand = expansion(mult = c(0.08, 0.18))) +
    labs(
      title = "Expression-Space kNN Tissue Neighborhoods",
      subtitle = "Nearest real target samples to translated source samples should share the source tissue label",
      x = NULL,
      y = "Score",
      fill = NULL,
      caption = "Higher is better."
    ) +
    coord_cartesian(clip = "off") +
    theme_archs4(10.5)
  save_both(p_expr_knn, "tissue_expression_knn_neighborhoods", 16, 8.8)
}

message("Wrote tissue figures and summaries to: ", out_dir)
