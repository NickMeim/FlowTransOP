library(tidyverse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(cowplot)
library(ggpubr)
library(patchwork)
library(rstatix)

### Load  bracketed data---------------------
files_dec_diff_in <- list.files('../results/Decoders_only_diffenetInputs_bracketed/',full.names = TRUE)
files_dec_diff_in <- files_dec_diff_in[grepl('translation',files_dec_diff_in)]
results_dec_diff_ins <- data.frame()
for (file in files_dec_diff_in){
  tmp <- data.table::fread(file)
  results_dec_diff_ins <- rbind(results_dec_diff_ins,tmp)
}
results_dec_diff_ins <- results_dec_diff_ins %>% mutate(approach='Consensus space decoders') %>%
  gather('set','r',-fold,-cell,-iteration,-approach)

## Load generalizedtransop results
files <- list.files('../results/AutoTransOP_CellPairs_diffenetInputs_bracketed/',full.names = TRUE)
files <- files[grepl('translation',files)]
results_diff_ins <- data.frame()
for (file in files){
  tmp <- data.table::fread(file)
  if ('shuffled X' %in% colnames(tmp)){
    tmp <- tmp %>% select(-`shuffled X`)
  }
  results_diff_ins <- rbind(results_diff_ins,tmp)
}
results_diff_ins <- results_diff_ins %>% mutate(approach='GeneralizedTransOP') %>%
  gather('set','r',-fold,-cell,-iteration,-approach)
results_diff_ins <- results_diff_ins %>% select(all_of(colnames(results_dec_diff_ins)))

## load brackets average similarity
similarity <- data.table::fread('MeanSubsetCorrelationsWithBrackets.csv')
colnames(similarity)[2] <- 'input_correlation'

## Load uncorrelated data---------
files_dec_diff_in_uncorr <- list.files('../results/DecodersOnly_differentInputs/',full.names = TRUE)
files_dec_diff_in_uncorr <- files_dec_diff_in_uncorr[grepl('translation',files_dec_diff_in_uncorr)]
results_dec_diff_ins_uncorr <- data.frame()
for (file in files_dec_diff_in_uncorr){
  tmp <- data.table::fread(file)
  results_dec_diff_ins_uncorr <- rbind(results_dec_diff_ins_uncorr,tmp)
}
results_dec_diff_ins_uncorr <- results_dec_diff_ins_uncorr %>% mutate(approach='Consensus space decoders') %>%
  gather('set','r',-fold,-cell,-iteration,-approach)
## Load generalizedtransop results
files_uncorr <- list.files('../results/AutoTransOP_CellPairs_diffenetInputs/',full.names = TRUE)
files_uncorr <- files_uncorr[grepl('translation',files_uncorr)]
results_diff_ins_uncorr <- data.frame()
for (file in files_uncorr){
  tmp <- data.table::fread(file) %>% select(-`shuffled X`)
  results_diff_ins_uncorr <- rbind(results_diff_ins_uncorr,tmp)
}
results_diff_ins_uncorr <- results_diff_ins_uncorr %>% mutate(approach='GeneralizedTransOP') %>%
  gather('set','r',-fold,-cell,-iteration,-approach)
results_diff_ins_uncorr <- results_diff_ins_uncorr %>% select(all_of(colnames(results_dec_diff_ins)))

## Process uncorrelated data for iteration 9 only
results_diff_ins_uncorr_processed <- results_diff_ins_uncorr %>%
  group_by(fold,set,cell) %>% mutate(r = mean(r)) %>%
  ungroup() %>% mutate(iteration=9) %>% unique()
results_dec_diff_ins_uncorr_processed <- results_dec_diff_ins_uncorr %>%
  group_by(fold,set,cell) %>% mutate(r = mean(r)) %>%
  ungroup() %>% mutate(iteration=9)%>% unique()

similarity <- rbind(similarity,
                    data.frame(iteration = rep(9,length(unique(similarity$cell))),
                               input_correlation=rep(0,length(unique(similarity$cell))),
                               cell = unique(similarity$cell)))

## Visualize------------------------------------------------
compare_diffIns_plt <- rbind(results_dec_diff_ins,
                             results_diff_ins,
                             results_diff_ins_uncorr_processed,
                             results_dec_diff_ins_uncorr_processed)
compare_diffIns_plt <- left_join(similarity,compare_diffIns_plt)
colnames(compare_diffIns_plt)[1] <- 'group_id'

# Create grouped data by averaging groups 0-4
df_grouped <- compare_diffIns_plt %>%
  mutate(group_category = ifelse(group_id %in% 0:4, "Easy_Tasks", paste0("group_", group_id))) %>%
  group_by(group_category, cell, fold, approach, set) %>%
  summarize(
    input_correlation = mean(input_correlation),
    r = mean(r),
    .groups = "drop"
  ) %>%
  mutate(group_id = group_category)
df_grouped <- df_grouped %>%
  mutate(difficulty_category = case_when(
    group_id == "group_9" ~ "Uncorrelated",
    group_id == "group_8" ~ "Very Difficult",
    group_id == "group_7" ~ "Difficult",
    group_id == "group_6" ~ "Moderate",
    TRUE ~ "Easy"
  ))

# Filter test set for analysis
test_df <- df_grouped %>% filter(set == "test")

# Calculate mean, SEM for each difficulty category and approach
category_stats <- test_df %>%
  group_by(difficulty_category, approach) %>%
  summarize(
    r_mean = mean(r),
    r_sd = sd(r),
    r_sem = sd(r) / sqrt(n()),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    difficulty_category = factor(difficulty_category, 
                                 levels = c("Uncorrelated","Very Difficult", "Difficult", "Moderate", "Easy"))
  )

# Perform t-tests for each difficulty category
difficulty_levels <- c("Uncorrelated","Very Difficult", "Difficult", "Moderate", "Easy")
sig_results <- data.frame()

for (cat in difficulty_levels) {
  cat_data <- test_df %>% filter(difficulty_category == cat)
  
  consensus_vals <- cat_data %>% filter(approach == "Consensus space decoders") %>% pull(r)
  generalized_vals <- cat_data %>% filter(approach == "GeneralizedTransOP") %>% pull(r)
  
  # Perform t-test
  test_result <- t.test(generalized_vals, consensus_vals)
  p_val <- test_result$p.value
  
  # Determine significance stars
  sig_stars <- case_when(
    p_val < 0.001 ~ "***",
    p_val < 0.01 ~ "**",
    p_val < 0.05 ~ "*",
    TRUE ~ "ns"
  )
  
  sig_results <- rbind(sig_results, data.frame(
    category = cat,
    p_value = p_val,
    significance = sig_stars
  ))
}

sig_results$category <- factor(sig_results$category, 
                               levels = c("Uncorrelated","Very Difficult", "Difficult", "Moderate", "Easy"))

print("Statistical significance results:")
print(sig_results)

# Calculate summary statistics by group_id
plot_data_A <- test_df %>%
  group_by(approach,group_id,cell) %>%
  mutate(mu_cell = mean(r)) %>%
  ungroup() %>%
  group_by(group_id, approach) %>%
  summarize(
    r_mean = mean(r),
    r_sem = sd(r) / sqrt(n()),
    min_r = min(r),
    max_r = max(r),
    # mu_cell = mu_cell,
    ci_lower = mean(r) + qt(0.025, df = n() - 1) * sd(r)/sqrt(n()),
    ci_upper = mean(r) + qt(0.975, df = n() - 1) * sd(r)/sqrt(n()),
    # cell=cell,
    input_correlation = mean(input_correlation),
    .groups = "drop"
  ) %>%
  arrange(input_correlation)
# fit <- lm(r ~ input_correlation, data = test_df %>% filter(approach=='GeneralizedTransOP'))
# b <- coef(fit)
# label_form <- sprintf("r = %.2f + %.2f · input_corr", b[1], b[2])
panel_A <- ggplot(plot_data_A, aes(x = input_correlation, y = r_mean, 
                                   color = approach, shape = approach, fill = approach)) +
  # Shaded background regions
  annotate("rect", xmin = 0, xmax = 0.35, ymin = -Inf, ymax = Inf, 
           fill = "red", alpha = 0.05) +
  annotate("text", x = 0.1, y = 0.55, 
           color = "darkred", label = 'Difficult bracket') +
  annotate("rect", xmin = 0.55, xmax = Inf, ymin = -Inf, ymax = Inf, 
           fill = "green", alpha = 0.05) +
  annotate("text", x = 0.56, y = 0.55, 
           color = "darkgreen", label = 'Easy bracket') +
  # Main lines with error bars
  geom_line(linewidth = 1.2) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              alpha = 0.2, colour = NA) +
  geom_point(size = 3) +
  
  # Styling
  scale_color_manual(values = c("Consensus space decoders" = "#2E86AB",
                                "GeneralizedTransOP" = "#E63946")) +
  scale_fill_manual(values = c("Consensus space decoders" = "#2E86AB",
                               "GeneralizedTransOP" = "#E63946")) +
  scale_shape_manual(values = c("Consensus space decoders" = 16,
                                "GeneralizedTransOP" = 15)) +
  
  labs(
    title = "A) Models` Performance Across Task Difficulty Spectrum",
    x = "Features` Correlation (Task Difficulty)",
    y = "Performance (pearson`s r)",
    color = "Approach",
    shape = "Approach",
    fill = "Approach"  # Add this
  ) +
  
  guides(fill = guide_legend(override.aes = list(alpha = 0.2)),  # Add this
         color = guide_legend(override.aes = list(alpha = 1))) +  # Add this
  
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(face = "bold", size = 24, hjust = 0),
    plot.subtitle = element_text(size = 14, hjust = 0),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )
panel_A
# panel_A <- ggplot(
#   test_df %>%
#     arrange(cell, approach, input_correlation),
#   aes(
#     x = input_correlation,
#     y = r,
#     color = approach,
#     group = interaction(cell, approach),   # one line per cell × approach
#     lty = approach
#   )
# ) +
#   annotate("rect", xmin = 0, xmax = 0.35, ymin = -Inf, ymax = Inf,
#            fill = "red", alpha = 0.05) +
#   annotate("rect", xmin = 0.55, xmax = Inf, ymin = -Inf, ymax = Inf,
#            fill = "green", alpha = 0.05) +
#   geom_smooth(
#     method = "lm",       # linear fit instead of loess
#     se = FALSE,
#     linewidth = 0.6,
#     alpha = 0.7,
#     color = 'grey'
#   ) +
#   geom_smooth(aes(x = input_correlation,
#                   y = r,
#                   color = approach,
#                   lty = approach),
#               inherit.aes = FALSE,
#               method = "lm",       # linear fit instead of loess
#               se = TRUE,
#               linewidth = 1.5
#   ) +
#   # if you want a bit of curvature:
#   # geom_smooth(method = "lm", formula = y ~ poly(x, 2), ...)
#   
#   scale_color_manual(values = c(
#     "Consensus space decoders" = "#2E86AB",
#     "GeneralizedTransOP"       = "#E63946"
#   )) +
#   coord_cartesian(ylim = c(0, 1)) +
#   labs(
#     title = "A) Models' Performance Across Task Difficulty Spectrum",
#     x = "Features' Correlation (Task Difficulty)",
#     y = "Performance (Pearson's r)",
#     color = "Approach"
#   ) +
#   theme_minimal(base_size = 16) +
#   theme(
#     plot.title = element_text(face = "bold", size = 24, hjust = 0),
#     axis.title = element_text(face = "bold"),
#     legend.position = "bottom",
#     legend.title = element_text(face = "bold"),
#     panel.grid.minor = element_blank()

# Calculate y-positions for significance stars - place above the bars
## Wilcoxon (paired) per difficulty category
stat_w <- test_df %>%
  group_by(difficulty_category) %>%
  t_test(r ~ approach, paired = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%   # adds p.adj.signif
  ungroup()

## Rank-biserial effect size per difficulty
test_df$approach <- factor(test_df$approach,
                           levels=c("GeneralizedTransOP",
                                    "Consensus space decoders"))
eff_w <- test_df %>% 
  group_by(difficulty_category) %>%
  cohens_d(r ~ approach, paired = TRUE) %>%  # effsize = rank-biserial r
  ungroup()

## Mean difference: GeneralizedTransOP - Consensus space decoders
mean_diff <- test_df %>%
  group_by(difficulty_category, approach) %>%
  summarise(mean_r = mean(r), .groups = "drop") %>%
  tidyr::pivot_wider(
    names_from  = approach,
    values_from = mean_r
  ) %>%
  mutate(
    diff_mean = `GeneralizedTransOP` - `Consensus space decoders`
  )

## Combine, build multi-line label
stat_labels <- stat_w %>%
  left_join(eff_w %>% select(difficulty_category, effsize),
            by = "difficulty_category") %>%
  left_join(mean_diff %>% select(difficulty_category, diff_mean),
            by = "difficulty_category") %>%
  mutate(
    group1 = difficulty_category,
    group2 = difficulty_category,
    y.position = 0.9,   # adjust as needed
    label = paste0(
      p.adj.signif,
      "\nΔmean r = ", sprintf("%+.3f", diff_mean),
      "\nCohen`s d = ", sprintf("%+.2f", effsize)
    )
  )
panel_B <- ggviolin(test_df %>%
                      mutate(
                        difficulty_category = factor(difficulty_category, 
                                                     levels = c("Uncorrelated","Very Difficult", "Difficult", "Moderate", "Easy"))
                      ), 
                    x = 'difficulty_category', y = 'r', fill = 'approach',width = 0.8) +
  geom_boxplot(
    aes(x = difficulty_category, y = r, fill = approach),
    width = 0.15,                          # narrower
    position = position_dodge(0.8),        # align with violins
    outlier.size = 0.5
  )+
  stat_pvalue_manual(
    stat_labels,
    label       = "label",
    xmin        = "group1",
    xmax        = "group2",
    y.position  = "y.position",
    bracket.size = 0,    # no brackets
    tip.length   = 0,
    size         = 5
  )  +
  scale_fill_manual(values = c("Consensus space decoders" = "#2E86AB",
                               "GeneralizedTransOP" = "#E63946")) +
  
  labs(
    title = "B) Performance by Difficulty Category",
    subtitle = "Bonferroni adjusted p-value (p): *** p<0.001, ** p<0.01, * p<0.05",
    x = "Difficulty Category",
    y = "pearson`s r",
    fill = "Approach"
  ) +
  scale_y_continuous(n.breaks = 10)+
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(face = "bold", size = 24, hjust = 0),
    plot.subtitle = element_text(size = 14, hjust = 0),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  ) +
  coord_cartesian(ylim = c(0, 1.2))

final_figure <- panel_A/panel_B
print(final_figure)

# Save the final figure
ggsave("../figures/decoders_autotransop_bracheted_performance.png", final_figure, 
       width = 28, height = 28,units = 'cm', dpi = 600, bg = "white")

