library(tidyverse)
library(ggplot2)
library(ggpubr)
library(cowplot)

dir.create("../figures", showWarnings = FALSE, recursive = TRUE)

read_translation <- function(result_dir, pattern) {
  files <- list.files(result_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) {
    stop("No matching files found in ", result_dir)
  }

  purrr::map_dfr(files, function(file) {
    readr::read_csv(file, show_col_types = FALSE,
                    name_repair = "unique_quiet") %>%
      select(-any_of(c("...1", "V1")))
  })
}

structural <- read_translation(
  "../results/AutoTransOP_CellPairs",
  "flow[0-9]+_TransAct_GeneralizedTransOP_translation_eval\\.csv$"
) %>%
  select(fold, translation,
         structural_guidance = test,
         direct_translation = Direct_pearson,
         shuffled_x = `shuffled X`)

no_structural <- read_translation(
  "../results/FlowMatch_no_structural_guidance",
  "flow[0-9]+_TransAct_GeneralizedTransOP_translation_latent_dim30_eval\\.csv$"
) %>%
  select(fold, translation,
         no_structural_guidance = test)

matched_results <- inner_join(
  no_structural,
  structural,
  by = c("fold", "translation")
)

violin_values <- bind_rows(
  structural %>%
    pivot_longer(c(structural_guidance, direct_translation, shuffled_x),
                 names_to = "approach",
                 values_to = "r"),
  no_structural %>%
    pivot_longer(no_structural_guidance,
                 names_to = "approach",
                 values_to = "r")
) %>%
  mutate(
    approach = recode(
      approach,
      structural_guidance = "Structural guidance",
      no_structural_guidance = "No structural guidance",
      direct_translation = "Direct translation",
      shuffled_x = "Shuffled X"
    ),
    approach = factor(
      approach,
      levels = c("Structural guidance", "No structural guidance",
                 "Direct translation", "Shuffled X")
    ),
    translation = factor(translation, levels = sort(unique(translation)))
  )

# The dotplot is a paired comparison, so it keeps only translation
# directions/folds present in the no-structural-guidance run.
dot_values <- matched_results %>%
  pivot_longer(
    c(structural_guidance, no_structural_guidance,
      direct_translation, shuffled_x),
    names_to = "approach",
    values_to = "r"
  ) %>%
  mutate(
    approach = recode(
      approach,
      structural_guidance = "Structural guidance",
      no_structural_guidance = "No structural guidance",
      direct_translation = "Direct translation",
      shuffled_x = "Shuffled X"
    ),
    approach = factor(
      approach,
      levels = c("Structural guidance", "No structural guidance",
                 "Direct translation", "Shuffled X")
    ),
    translation = factor(translation, levels = sort(unique(translation)))
  )

comparisons <- list(
  c("Structural guidance", "No structural guidance"),
  c("No structural guidance", "Direct translation"),
  c("No structural guidance", "Shuffled X")
)

min_val <- min(min(violin_values$r, dot_values$r, na.rm = TRUE), 0)

pv <- ggviolin(violin_values %>% arrange(translation, fold),
               x = "approach", y = "r",
               fill = "approach",
               add = "jitter",
               palette = "jco",
               draw_quantiles = 0.5,
               size = 2) +
  ylab("Pearson's r") +
  scale_y_continuous(n.breaks = 10, limits = c(min_val, 1)) +
  stat_compare_means(
    comparisons = comparisons,
    label.y = c(0.72, 0.80, 0.88),
    method = "wilcox",
    label = "p.signif",
    size = 5
  ) +
  theme(
    text = element_text(size = 16),
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.position = "none",
    plot.margin = margin(5.5, 5.5, 14, 5.5)
  )

plot_values_mean <- dot_values %>%
  group_by(approach, translation) %>%
  mutate(r = mean(r, na.rm = TRUE)) %>%
  ungroup() %>%
  select(-fold) %>%
  unique()

pd <- ggdotplot(plot_values_mean,
                x = "approach", y = "r",
                fill = "translation",
                position = position_dodge(0.025)) +
  geom_line(aes(x = approach, y = r, group = translation),
            color = "gray50", size = 0.75, alpha = 0.35) +
  stat_summary(fun = mean, geom = "crossbar",
               width = 0.3, color = "black", linetype = "dashed",
               fatten = 2) +
  scale_y_continuous(n.breaks = 10, limits = c(min_val, 1)) +
  guides(fill = guide_legend(ncol = 4)) +
  ylab("Pearson's r averaged across folds") +
  stat_compare_means(
    comparisons = comparisons,
    label.y = c(0.72, 0.80, 0.88),
    method = "wilcox",
    paired = TRUE,
    label = "p.signif",
    size = 5
  ) +
  theme(
    text = element_text(size = 16),
    axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 15, hjust = 1, vjust = 1),
    axis.title.y = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.position = "bottom",
    legend.text = element_text(size = 11),
    legend.title = element_blank(),
    plot.margin = margin(14, 5.5, 5.5, 5.5)
  )

figure_s1 <- cowplot::plot_grid(
  pv,
  pd,
  labels = c("a", "b"),
  ncol = 1,
  rel_heights = c(1, 1),
  align = "v",
  axis = "lr",
  label_fontfamily = "sans",
  label_fontface = "bold",
  label_size = 14,
  label_x = 0.045,
  label_y = 0.99,
  hjust = 0,
  vjust = 1
)

ggsave("../figures/Supplementary_Figure_S1_no_structural_guidance.png",
       plot = figure_s1,
       width = 36,
       height = 24,
       units = "cm",
       dpi = 600)
