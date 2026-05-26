library(tidyverse)
library(ggplot2)
library(ggpubr)
library(patchwork)
library(ggridges)

dir.create("../figures", showWarnings = FALSE, recursive = TRUE)

panel_tag_theme <- theme(
  plot.tag = element_text(family = "Arial", face = "bold", size = 14)
)
real_data_panel_tag_theme <- theme(
  plot.tag = element_text(family = "Arial", face = "bold", size = 14),
  plot.tag.position = c(-0.012, 1)
)
theme_set(theme_get() + panel_tag_theme)

with_panel_tags <- function(plot) {
  plot + panel_tag_theme
}

with_real_data_panel_tags <- function(plot) {
  plot + real_data_panel_tag_theme
}

save_png_pdf <- function(plot, stem, width, height, units = "cm", dpi = 600) {
  ggsave(file.path("../figures", paste0(stem, ".png")),
         plot = plot, width = width, height = height, units = units, dpi = dpi)
  ggsave(file.path("../figures", paste0(stem, ".pdf")),
         plot = plot, width = width, height = height, units = units,
         device = cairo_pdf)
}

make_random_plots <- function(detailed_results_file) {
  distribution <- stringr::str_remove(basename(detailed_results_file), "_detailed_results\\.csv$")
  all_results <- data.table::fread(detailed_results_file)
  all_results <- all_results %>% 
    gather('matrix','r',-feature_size,-sample_size,-iteration,-cpu_time,-gpu_time,-speedup) %>%
    mutate(metric = ifelse(grepl('min',matrix),'minimum','mean')) %>%
    mutate(matrix = ifelse(grepl('_1_val',matrix),'validation x1',
                           ifelse(grepl('_2_val',matrix),'validation x2',
                                  ifelse(grepl('_1',matrix),'x1','x2'))))
  
  p_sample <- ggplot(all_results,
                     aes(x=sample_size,y=r,color=matrix))+
    geom_smooth()+
    scale_y_continuous(limits = c(0,1.05))+
    facet_grid(metric ~ feature_size,
               labeller = labeller(
                 feature_size = function(x) paste("# of features:", x),
                 metric = function(x) paste(x,"of r")
               ))
  
  p_dimension <- ggplot(all_results %>% mutate(dimension_ratio = feature_size/sample_size) %>%
                          group_by(metric,matrix,dimension_ratio) %>%
                          mutate(mu = mean(r)) %>%
                          mutate(std = sd(r)) %>%
                          mutate(se = std/sqrt(n())) %>%
                          ungroup() %>%
                          select(-iteration) %>%
                          unique(),
                        aes(x=dimension_ratio,y=mu,color=matrix,fill=matrix))+
    scale_y_continuous(limits= c(0,1.099),
                       breaks = seq(0,1,0.1))+
    scale_x_continuous(n.breaks = 10)+
    geom_line()+
    geom_point()+
    geom_ribbon(aes(ymin = mu - std,ymax=mu+std),alpha=0.4,color = NA)+
    geom_hline(yintercept = 0.9,linetype='dashed',color='black',lwd=0.5)+
    geom_vline(xintercept = 1,linetype='dashed',color='black',lwd=0.5)+
    annotate(
      "text",
      x = 23,
      y = 0.8,
      label = "dimensions` ratio = 1"
    )+
    facet_wrap(~metric,
               labeller = labeller(
                 metric = function(x) paste(x,"of r")
               ))+
    xlab('# features / # samples')+
    ylab('Pearson`s r')+
    theme_pubr()
  
  p_sample <- with_panel_tags(p_sample)
  p_dimension <- with_panel_tags(p_dimension)
  combined <- p_sample / p_dimension + plot_annotation(tag_levels = "a")
  
  # Keep the historical individual PNG outputs and add one merged PNG/PDF
  # pair per random distribution for Supplementary Figure S12 assembly.
  save_png_pdf(p_sample, paste0("CPU_vs_GPU_random_", distribution), 24, 12)
  save_png_pdf(p_dimension, paste0("CPU_vs_GPU_random_", distribution, "_dim_effect"), 16, 12)
  save_png_pdf(combined,
               paste0("Supplementary_Figure_S12_random_", distribution, "_combined"),
               24,
               24)
}

## Compare random matrices results================
random_detailed_files <- list.files("../results/GPU_vs_CPU_random",
                                    pattern = "_detailed_results\\.csv$",
                                    full.names = TRUE)
purrr::walk(random_detailed_files, make_random_plots)

#### Compare results in my actual cells (imputed genes)================
files_cells <- list.files('../results/GPU_vs_CPU/',full.names = TRUE)
files_cells <- files_cells[grepl('same',files_cells)]
files_cells <- files_cells[!grepl('time',files_cells)]
all_results_cells <- data.frame()
for (file in files_cells) {
  tmp <- data.table::fread(file) %>% select(-V1)
  if (grepl('_1_val',file)){
    mat <- 'validation x1'
  }else if (grepl('_2_val',file)){
    mat <- 'validation x2'
  }else if (grepl('_1',file)) {
    mat <- 'x1'
  }else{
    mat <- 'x2'
  }
  tmp <- tmp %>% mutate(matrix = mat)
  all_results_cells <- rbind(all_results_cells,tmp)
}

p_imputed_cells <- ggplot(all_results_cells, aes(x = value, y = as.factor(matrix))) +
  geom_density_ridges(alpha = 0.75,
                      color='black') +
  scale_x_continuous(n.breaks = 10 )+
  ggtitle('Distribution of Pearson`s r between CPU and GPU constructed aligned latent variables')+
  xlab('Pearson`s r') + ylab('matrix')+ theme(base_family = "Arial") + 
  theme_pubr(base_family = "Arial",base_size = 14) + 
  theme(plot.title = element_text(hjust = 1)) +
  panel_tag_theme
save_png_pdf(p_imputed_cells, "CPU_vs_GPU_in_imputed_cells", 24, 12)

#### Compare results in my actual cell line pairs================
files_pairs <- list.files('../results/GPU_vs_CPU/',full.names = TRUE)
files_pairs <- files_pairs[grepl('cellLinePairs',files_pairs)]
files_pairs <- files_pairs[!grepl('time',files_pairs)]
all_results_pairs <- data.frame()
for (file in files_pairs) {
  tmp <- data.table::fread(file) %>% select(-V1)
  if (grepl('_1_val',file)){
    mat <- 'validation x1'
  }else if (grepl('_2_val',file)){
    mat <- 'validation x2'
  }else if (grepl('_1.csv',file)) {
    mat <- 'x1'
  }else{
    mat <- 'x2'
  }
  tmp <- tmp %>% mutate(matrix = mat)
  all_results_pairs <- rbind(all_results_pairs,tmp)
}

ggviolin(all_results_pairs,
       x='folder',y='value',fill='matrix',width = 2)+
  scale_y_continuous(limits = c(0.96,1.01))

p_cell_line_pairs <- ggplot(all_results_pairs, aes(x = value, y = as.factor(folder),fill=matrix)) +
  geom_density_ridges(alpha = 0.75,
                      color='black') +
  scale_x_continuous(n.breaks = 10 )+
  ggtitle('Distribution of Pearson`s r between CPU and GPU constructed aligned latent variables')+
  xlab('Pearson`s r') + ylab('cell line pair')+ theme(base_family = "Arial") + 
  theme_pubr(base_family = "Arial",base_size = 14) + 
  theme(plot.title = element_text(hjust = 1)) +
  panel_tag_theme
save_png_pdf(p_cell_line_pairs, "CPU_vs_GPU_in_cell_line_pairs", 24, 12)


#### Compare results in my actual cell line pairs that are subsetted================
files_pairs_sub <- list.files('../results/GPU_vs_CPU/',full.names = TRUE)
files_pairs_sub <- files_pairs_sub[grepl('subSampled',files_pairs_sub)]
files_pairs_sub <- files_pairs_sub[!grepl('time',files_pairs_sub)]
all_results_pairs_sub <- data.frame()
for (file in files_pairs_sub) {
  tmp <- data.table::fread(file) %>% select(-V1)
  if (grepl('_1_val',file)){
    mat <- 'validation x1'
  }else if (grepl('_2_val',file)){
    mat <- 'validation x2'
  }else if (grepl('_1.csv',file)) {
    mat <- 'x1'
  }else{
    mat <- 'x2'
  }
  tmp <- tmp %>% mutate(matrix = mat)
  all_results_pairs_sub <- rbind(all_results_pairs_sub,tmp)
}

ggplot(all_results_pairs_sub,
       aes(x=sample_size,y=value,color=variable))+
  geom_smooth()+
  scale_y_continuous(limits = c(0,1.05))+
  facet_wrap(~matrix)

p_subsampled <- ggplot(all_results_pairs_sub %>% 
         group_by(variable,matrix,sample_size) %>%
         mutate(mu = mean(value)) %>%
         mutate(std = sd(value)) %>%
         mutate(minimum = min(value)) %>%
         mutate(maximum = max(value)) %>%
         ungroup() %>%
         mutate(se = std/sqrt(30)) %>%
         select(-itearation,-value) %>%
         unique(),
       aes(x=sample_size,y=mu,color=variable,fill=variable))+
  scale_y_continuous(limits= c(0,1.099),
                     breaks = seq(0,1,0.1))+
  scale_x_continuous(n.breaks = 10)+
  geom_line(lwd=1)+
  geom_point(size=3)+
  geom_ribbon(aes(ymin = minimum,ymax=maximum),alpha=0.2,color = NA)+
  geom_hline(yintercept = 0.9,linetype='dashed',color='black',lwd=0.5)+
  geom_vline(xintercept = 128,linetype='dashed',color='black',lwd=0.5)+
  guides(
    color = guide_legend(ncol = 10),
    fill  = guide_legend(ncol = 10)
  )+
  facet_wrap(~matrix)+
  xlab('# of samples')+
  ylab('Pearson`s r')+
  theme_pubr() +
  panel_tag_theme
save_png_pdf(p_subsampled, "CPU_vs_GPU_subsampled", 24, 14)

p_real_data_baselines <- with_real_data_panel_tags(p_cell_line_pairs) /
  with_real_data_panel_tags(p_imputed_cells) /
  with_real_data_panel_tags(p_subsampled) +
  plot_annotation(tag_levels = "a")
save_png_pdf(p_real_data_baselines,
             "Supplementary_Figure_S12_real_data_baselines_combined",
             24,
             36)
