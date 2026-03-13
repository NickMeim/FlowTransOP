library(tidyverse)
library(ggplot2)
library(ggpubr)
library(patchwork)
library(lme4)
library(emmeans)

## Load example data----------
results <- data.table::fread('../results/AutoTransOP_CellPairs/A375_HT29_flow12_TransAct_GeneralizedTransOP_translation_eval.csv') %>%
  select(-V1)
results <- results %>% gather('set','r',-fold,-translation)

## Visualize-----------
min_val <- min(min(results$r),0)
results <- results %>% mutate(set=ifelse(set=='Direct_pearson','Direct translation',set))
results$set <- factor(results$set,levels = c('AutoTransOP','train','test','Direct translation','shuffled X'))
ggboxplot(results %>% arrange(fold),x='set',y='r',
          color='set',
          add='jitter',palette = "jco")+
  ylab('pearson`s correlation')+
  scale_y_continuous(n.breaks = 10,limits = c(min_val,1))+
  stat_compare_means(comparisons = list(c('AutoTransOP','train'),
                                        c('AutoTransOP','test'),
                                        c('train','Direct translation'),
                                        c('test','Direct translation'),
                                        c('test','shuffled X'),
                                        c('train','shuffled X'),
                                        c('train','test'),
                                        c('Direct translation','shuffled X')),
                     method = 'wilcox',
                     label.y = c(0.73,0.78,0.73,0.6,0.67,0.83,0.6,0.6))+
  facet_wrap(~translation)+
  theme(text = element_text(size=16),
        axis.title.x = element_blank(),
        legend.position = 'none')
ggsave('../flow12_transact_performance_a375_ht29.png',
       width = 20,
       height = 14,
       units = 'cm',
       dpi = 600)

# ggpaired(results %>% arrange(fold), x = "set", y = "r", id = "fold",
#          color = "set", line.color = "gray", line.size = 0.4,
#          palette = "jco") +
#   stat_compare_means(comparisons = list(c('train','Direct_pearson'),
#                                         c('test','Direct_pearson'),
#                                         c('test','shuffled X'),
#                                         c('train','shuffled X'),
#                                         c('train','test'),
#                                         c('Direct_pearson','shuffled X')),
#                      method = 'wilcox',
#                      label.y = c(0.75,0.65,0.7,0.8,0.6,0.6))+
#   theme(text = element_text(size=24),
#         axis.title.x = element_blank())

## Load all data----------
files <- list.files('../results/AutoTransOP_CellPairs/',full.names = TRUE)
results <- data.frame()
for (file in files){
  tmp <- data.table::fread(file) %>%
    select(-V1)
  results <- rbind(results,tmp)
}
results <- results %>% gather('set','r',-fold,-translation)
statistics_results_train <- results %>% filter(set %in% c('train','Direct_pearson')) %>%
  group_by(translation) %>%
  rstatix::wilcox_test(r~set,alternative = 'less',ref.group = 'Direct_pearson') %>%
  rstatix::adjust_pvalue(method = 'BH') %>% ungroup() %>%
  mutate(better=ifelse(p.adj<0.05,1,0))
statistics_results_test <- results %>% filter(set %in% c('test','Direct_pearson')) %>%
  group_by(translation) %>%
  rstatix::wilcox_test(r~set,alternative = 'less',ref.group = 'Direct_pearson') %>%
  rstatix::adjust_pvalue(method = 'BH') %>% ungroup()%>%
  mutate(better=ifelse(p.adj<0.05,1,0))

## summary
### Think perhas how making a contigency table to do a fisher exact test
## higher than direct, lower than direct, statistically significant, no statistically significant
print(paste0('Training performance better than direct translation in ',
             round(sum(statistics_results_train$better)/16*100,2),'%'))
print(paste0('Test performance better than direct translation in ',
             round(sum(statistics_results_test$better)/16*100,2),'%'))

## Visualize-----------
min_val <- min(min(results$r),0)
results <- results %>% mutate(set=ifelse(set=='Direct_pearson','Direct translation',set))
results$set <- factor(results$set,levels = c('AutoTransOP','train','test','Direct translation','shuffled X'))
ggboxplot(results %>% arrange(fold),x='set',y='r',
          color='set',
          add='jitter',palette = "jco")+
  ylab('pearson`s correlation')+
  scale_y_continuous(n.breaks = 10,limits = c(min_val,1))+
  stat_compare_means(comparisons = list(c('AutoTransOP','train'),
                                        c('AutoTransOP','test'),
                                        c('train','Direct translation'),
                                        c('test','Direct translation'),
                                        c('test','shuffled X'),
                                        c('train','shuffled X'),
                                        c('train','test'),
                                        c('Direct translation','shuffled X')),
                     method = 'wilcox',
                     label = 'p.signif',
                     label.y = c(0.73,0.78,0.73,0.6,0.67,0.83,0.6,0.6))+
  facet_wrap(~translation)+
  theme(text = element_text(size=10),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        legend.position = 'top')
results <- results %>%
  mutate(z = atanh(r))
lmm_fit <- lmer(z ~ set + (1 | translation) + (1 | fold),
                data = results)
emm <- emmeans(lmm_fit, ~ set)
contrasts_res <- contrast(emm, method = list(
  "AutoTransOP vs train"            = c(1, -1,  0,  0,  0),
  "AutoTransOP vs test"             = c(1,  0, -1,  0,  0),
  "train vs Direct translation"     = c(0,  1,  0, -1,  0),
  "test vs Direct translation"      = c(0,  0,  1, -1,  0),
  "test vs shuffled X"              = c(0,  0,  1,  0, -1),
  "train vs shuffled X"             = c(0,  1,  0,  0, -1),
  "train vs test"                   = c(0,  1, -1,  0,  0),
  "Direct translation vs shuffled X"= c(0,  0,  0,  1, -1)
  # adjust vector order to match levels(factor(results$set))
), adjust = "bonferroni")
summary(contrasts_res)
get_sig <- function(p) ifelse(p < 0.001, "***",
                              ifelse(p < 0.01,  "**",
                                     ifelse(p < 0.05,  "*", "ns")))

contrast_df <- as.data.frame(summary(contrasts_res)) %>%
  mutate(label = get_sig(p.value))
# Pull labels in the same order as comparisons below
sig_labels <- contrast_df$label  # order matches the contrast list above
pv <- ggviolin(results %>% arrange(fold),
         x = 'set', y = 'r',
         fill = 'set',
         add = 'jitter',
         palette = "jco",
         draw_quantiles = 0.5,
         size = 2) +
  ylab("Pearson's r") +
  scale_y_continuous(n.breaks = 10, limits = c(min_val, 1)) +
  geom_signif(
    comparisons = list(c('AutoTransOP', 'train'),
                       c('AutoTransOP', 'test'),
                       c('train', 'Direct translation'),
                       c('test', 'Direct translation'),
                       c('test', 'shuffled X'),
                       c('train', 'shuffled X'),
                       c('train', 'test'),
                       c('Direct translation', 'shuffled X')),
    annotations = sig_labels,        # <-- LMM Bonferroni labels
    y_position  = c(0.75, 0.83, 0.75, 0.65, 0.70, 0.80, 0.63, 0.60),
    tip_length  = 0.01,
    textsize    = 7
  ) +
  theme(text = element_text(size = 24),
        axis.title.x = element_blank(),
        axis.text.x  = element_blank(),
        legend.text  = element_text(size = 28),
        legend.title = element_text(size = 28),
        legend.position = 'top')

results_mean <- results %>%
  group_by(set, translation) %>%
  mutate(r = mean(r)) %>%
  ungroup() %>%
  select(-fold,-z) %>%
  unique()
pd <- ggdotplot(results_mean,
          x = 'set', y = 'r',
          fill = 'translation',
          position = position_dodge(0.025)) +
  geom_line(aes(x = set, y = r, group = translation),
            color = 'gray50', size = 0.75, alpha = 0.5) +
  stat_summary(fun = mean, geom = "crossbar",
               width = 0.3, color = "black", linetype = 'dashed',
               fatten = 2) +
  stat_compare_means(
    comparisons = list(c('AutoTransOP', 'train'),
                       c('AutoTransOP', 'test'),
                       c('train', 'Direct translation'),
                       c('test', 'Direct translation'),
                       c('test', 'shuffled X'),
                       c('train', 'shuffled X'),
                       c('train', 'test'),
                       c('Direct translation', 'shuffled X')),
    label.y  = c(0.75, 0.83, 0.75, 0.65, 0.70, 0.80, 0.63, 0.60),
    method='wilcox',
    paired=TRUE,
    label = 'p.signif' ,
    size=7
  ) +
  scale_y_continuous(n.breaks = 10, limits = c(min_val, 1)) +
  guides(fill = guide_legend(nrow = 2)) +
  ylab("Pearson's r averaged across folds") +
  theme(text= element_text(size = 24),
        axis.title.x  = element_blank(),
        legend.position = 'top',
        legend.text   = element_text(size = 16),
        legend.title  = element_blank(),
        axis.text     = element_text(size = 24))
p <- pv+pd
print(p)

ggsave('../figure3a.png',
       plot = p,
       width = 82.5,
       height = 24,
       units = 'cm',
       dpi = 600)
ggsave('../figure3a.eps',
       device = cairo_ps,
       plot = p,
       width = 82.5,
       height = 24,
       units = 'cm',
       dpi = 600)

## Load decoders only ---------------
files_dec_trans <- list.files('../results/DecodersOnly/',full.names = TRUE)
files_dec_trans <-  files_dec_trans[grepl('translation',files_dec_trans)]
results <- data.frame()
for (file in files_dec_trans){
  if (grepl('sources',file)){
    approach <- 'separate consensus space<-->source space'
  }else if (grepl('targets',file)){
    approach <- 'separate consensus space<-->target space'
  }else if (grepl('source_1',file)){
    approach <- 'consencus space: 1 -> 2'
  }else{
    approach <- 'consencus space: 2 -> 1'
  }
  tmp <- data.table::fread(file) %>%
    select(-V1) %>% mutate(approach=approach)
  if (!('subset_size' %in% colnames(tmp))){
    results <- rbind(results,tmp)
  }
}
results <- results %>% gather('set','r',-fold,-translation,-approach)

print(results %>% filter(approach %in% c('consencus space: 1 -> 2','consencus space: 2 -> 1')) %>%
        group_by(set,approach) %>% 
        summarise(mu = mean(r),med=median(r),min = min(r),max=max(r)))

ggboxplot(results,x='approach',y='r',color='approach',add='jitter')+
  facet_wrap(~set)+
  stat_compare_means(comparisons = list(c('consencus space: 1 -> 2','consencus space: 2 -> 1')),
                     method = 'wilcox',paired = TRUE)+
  theme(axis.text.x = element_blank())
ggsave('../transact_decoders_all_approaches.png',
       width = 28,
       height = 10,
       units = 'cm',
       dpi = 600)

## Now compare decoders only using only paired and random conditions
files_dec_trans <- list.files('../results/DecodersOnly/',full.names = TRUE)
files_dec_trans <-  files_dec_trans[grepl('translation',files_dec_trans)]
files_dec_trans <-  files_dec_trans[grepl('TransActPaired',files_dec_trans)]
files_dec_trans <-  files_dec_trans[!grepl('with_source_1',files_dec_trans)]
results_sub <- data.frame()
for (file in files_dec_trans){
  if (grepl('sources',file)){
    approach <- 'separate consensus space<-->source space'
  }else if (grepl('targets',file)){
    approach <- 'separate consensus space<-->target space'
  }else if (grepl('source_1',file)){
    approach <- 'consencus space: 1 -> 2'
  }else{
    approach <- 'consencus space: 2 -> 1'
  }
  tmp <- data.table::fread(file) %>%
    select(-V1) %>% mutate(approach=approach)
  if ('subset_size' %in% colnames(tmp)){
    results_sub <- rbind(results_sub,tmp)
  }
}

results_sub <- results_sub %>% gather('set','r',-subset_size,-iteration,-fold,-translation,-approach)
ggplot(results_sub ,
       aes(x=subset_size,y=r,color=set))+
  geom_smooth()+
  facet_wrap(~translation,scales = 'free_y')
ggsave('../transact_decoders_subsets_convergence.png',
       width = 20,
       height = 16,
       units = 'cm',
       dpi = 600)

# ggplot(results_sub %>% group_by(subset_size,set,translation,fold) %>% mutate(r=mean(r)) %>%
#          ungroup() %>% select(-iteration) %>% unique() %>%
#          group_by(subset_size,set,translation) %>% mutate(r=mean(r)) %>%
#          ungroup() %>% select(-fold) %>% unique(),
#        aes(x=subset_size,y=r,color=set))+
#   geom_line(lwd=1)+
#   geom_point(size=2.5)+
#   facet_wrap(~translation,scales = 'free_y')

ggboxplot(results_sub ,x='subset_size',y='r',add='jitter')+
  scale_y_continuous(n.breaks = 10,limits = c(0,1))+
  stat_compare_means(method='kruskal')+
  stat_compare_means(comparisons = list(c('512','384'),
                                        c('384','256'),
                                        c('256','128'),
                                        c('512','64')),
                     method='wilcox')+
  facet_wrap(~set)
ggsave('../transact_decoders_subsets_boxplots.png',
       width = 16,
       height = 12,
       units = 'cm',
       dpi = 600)
# ggboxplot(results_sub %>% filter(set=='test') ,x='subset_size',y='r',add='jitter')+
#   scale_y_continuous(n.breaks = 10,limits = c(0,1))+
#   stat_compare_means(method='kruskal')+
#   stat_compare_means(comparisons = list(c('512','384'),
#                                         c('384','256'),
#                                         c('256','128'),
#                                         c('512','64')),
#                      method='wilcox')+
#   facet_wrap(~translation)

compare_df <- results_sub %>% filter(subset_size==512) %>% select(-iteration,-subset_size) %>%
  mutate(`consensus space constructed with` = 'random subsets') %>%
  group_by(translation,fold,set) %>% mutate(r= mean(r)) %>% ungroup() %>% unique()
compare_df <- rbind(results %>% filter(approach=='consencus space: 2 -> 1') %>%
                      mutate(`consensus space constructed with` = 'pairs'),
                    compare_df)
ggboxplot(compare_df,
          x='consensus space constructed with',y='r',add = 'jitter')+
  facet_wrap(~set)+
  stat_compare_means(method = 'wilcox')
ggsave('../transact_decoders_paired_vs_subset.png',
       width = 18,
       height = 10,
       units = 'cm',
       dpi = 600)

### Compare my approach with decoders only-----
files <- list.files('../results/AutoTransOP_CellPairs/',full.names = TRUE)
results <- data.frame()
for (file in files){
  tmp <- data.table::fread(file) %>%
    select(-V1)
  results <- rbind(results,tmp)
}
results <- results %>% gather('set','r',-fold,-translation)
results <- results %>% mutate(set=ifelse(set=='Direct_pearson','Direct translation',set))
autotransop <- results %>% filter(set=='AutoTransOP') %>%
  mutate(approach=set) %>% select(-set)
results <- results %>% filter(set %in% c('train','test'))

# merge
compare_df_sub <- compare_df %>% filter(`consensus space constructed with` != 'pairs') %>%
  select(-`consensus space constructed with`) %>%
  mutate(approach = 'Consensus space decoders')
compare_df <- compare_df %>% filter(`consensus space constructed with` == 'pairs') %>%
  select(-`consensus space constructed with`) %>%
  mutate(approach = 'Consensus space decoders')
compare_df_autotransop <- rbind(compare_df %>% 
                                  filter(set=='test') %>% 
                                  select(all_of(colnames(autotransop))),
                                autotransop)
## VS AutoTransOP
p1 <- ggpaired(compare_df_autotransop,
          x='approach',y='r',
         line.size = 0.1,line.color = 'lightgrey',linetype = 'solid')+
  scale_y_continuous(n.breaks = 10,limits = c(0,1))+
  stat_compare_means(method = 'wilcox',paired=TRUE,size=6)+
  ylab('r') + xlab('approach')+
  theme(axis.title.x = element_blank())
ggsave('../transact_decoders_vs_autotransop.png',
       plot=p1,
       width = 12,
       height = 12,
       units = 'cm',
       dpi = 600)

## VS FlowTransOP
compare_df_plt <- rbind(compare_df,
                    results %>% mutate(approach='FlowTransOP') %>% 
                      select(all_of(colnames(compare_df))))
p2 <-ggpaired(compare_df_plt ,
         x='approach',y='r',
         line.size = 0.1,line.color = 'lightgrey',linetype = 'solid')+
  scale_y_continuous(n.breaks = 10,limits = c(0,1))+
  facet_wrap(~set)+
  stat_compare_means(method = 'wilcox')+
  ylab('r') + xlab('approach') + 
  theme(axis.title.x = element_blank())
ggsave('../transact_decoders_paired_vs_generalizedtransop.png',
       plot=p2,
       width = 12,
       height = 12,
       units = 'cm',
       dpi = 600)
p <- p2 + p1 + plot_layout(widths = c(2, 1))
print(p)
ggsave('../transact_decoders_vs_autotransop_vs_flowtransop.png',
       plot=p,
       width = 32,
       height = 12,
       units = 'cm',
       dpi = 600)
ggsave('../transact_decoders_vs_autotransop_vs_flowtransop.eps',
       device = cairo_ps,
       plot=p,
       width = 32,
       height = 12,
       units = 'cm',
       dpi = 600)

## Compare FlowTransOP when we have pairs------------------
files <- list.files('../results/AutoTransOP_CellPairs_withPairs/',full.names = TRUE)
files <- files[grepl('translation',files)]
results_paired_general <- data.frame()
for (file in files){
  tmp <- data.table::fread(file) %>%
    select(-V1)
  results_paired_general <- rbind(results_paired_general,tmp)
}
results_paired_general <- results_paired_general %>% gather('set','r',-fold,-translation)
results_paired_general <- results_paired_general %>% mutate(approach='Paired FlowTransOP')
results_paired_general <- results_paired_general %>% select(all_of(colnames(compare_df)))

## plot comparioson
stat.test <- rbind(results_paired_general,
                   compare_df %>% filter(translation %in% results_paired_general$translation)) %>%
  filter(set != 'shuffled X') %>%
  group_by(set) %>%
  rstatix::wilcox_test(r ~ approach) %>%
  rstatix::add_y_position() 
ggboxplot(
  rbind(results_paired_general,
        compare_df %>% filter(translation %in% results_paired_general$translation)),
  x = 'set', y = 'r', color = 'approach', add = 'jitter'
) +
  scale_y_continuous(n.breaks = 10,limits = c(0,1)) + 
  stat_pvalue_manual(
    stat.test,
    x = "set", 
    label = "Wilcox test p = {p}",
    position = position_dodge(0.8), 
    tip.length = 0.01
  )
ggsave('../transact_decoders_paired_vs_PairedGeneralizedtransop.png',
       width = 18,
       height = 12,
       units = 'cm',
       dpi = 600)

### Comare in different input spaces----------------
files_dec_diff_in <- list.files('../results/DecodersOnly_differentInputs/',full.names = TRUE)
files_dec_diff_in <- files_dec_diff_in[grepl('translation',files_dec_diff_in)]
results_dec_diff_ins <- data.frame()
for (file in files_dec_diff_in){
  tmp <- data.table::fread(file)
  results_dec_diff_ins <- rbind(results_dec_diff_ins,tmp)
}
results_dec_diff_ins <- results_dec_diff_ins %>% mutate(approach='Consensus space decoders') %>%
  gather('set','r',-fold,-cell,-iteration,-approach)

## Load FlowTransOP results
files <- list.files('../results/AutoTransOP_CellPairs_diffenetInputs/',full.names = TRUE)
files <- files[grepl('translation',files)]
results_diff_ins <- data.frame()
for (file in files){
  tmp <- data.table::fread(file)
  results_diff_ins <- rbind(results_diff_ins,tmp)
}
results_diff_ins <- results_diff_ins %>% mutate(approach='FlowTransOP') %>%
  gather('set','r',-fold,-cell,-iteration,-approach)
results_diff_ins <- results_diff_ins %>% select(all_of(colnames(results_dec_diff_ins)))

## Visualize
compare_diffIns_plt <- rbind(results_dec_diff_ins %>% filter(cell %in% results_diff_ins$cell),
                        results_diff_ins)
compare_diffIns_plt <- compare_diffIns_plt %>%
  mutate(approach = ifelse(set=='shuffled X','shuffled X',approach)) %>%
  mutate(set = ifelse(set=='shuffled X','test',set))
# compare_diffIns_plt <- compare_diffIns_plt %>%
#   mutate(id = paste0(set,'_',approach,'_',cell,'_',iteration,'_',fold))
min_val <- floor(min(compare_diffIns_plt$r))
pv <- ggviolin(compare_diffIns_plt,
               x = 'approach', y = 'r',fill='grey') +
  geom_boxplot(width=0.2)+
  scale_y_continuous(n.breaks = 10,limits = c(NA,1))+
  ylab('Pearson`s r')+
  ## facet where all three approaches exist
  stat_compare_means(
    data = subset(compare_diffIns_plt, set == "test"),
    comparisons = list(c('FlowTransOP','Consensus space decoders'),
                       c('FlowTransOP','shuffled X')),
    label.y = 0.65,
    method = 'wilcox',
  ) +
  ## facet where "shuffled X" is absent
  stat_compare_means(
    data = subset(compare_diffIns_plt, set == "train"),
    comparisons = list(c('FlowTransOP','Consensus space decoders')),
    label.y = 0.65,method = 'wilcox',
  ) +
  facet_wrap(~set,scales = 'free_x')+
  theme(axis.title.x = element_blank(),
        axis.text = element_text(size=14),
        axis.title = element_text(size=16),
        axis.text.x = element_blank())
min_val <- floor(min(compare_diffIns_plt %>% 
                       group_by(cell,approach) %>% mutate(r=mean(r)) %>%
                       ungroup() %>% select(-fold) %>% unique() %>% select(r)))
pd <- ggdotplot(compare_diffIns_plt %>% 
                  group_by(cell,approach,set) %>% mutate(r=mean(r)) %>%
                  ungroup() %>% select(-fold,-iteration) %>% unique(),
                x = 'approach', y = 'r',fill='cell',
                position = position_dodge(0.025))+
  geom_line(aes(x=approach,y=r,group = cell),
            color = 'gray50', size = 0.75, alpha = 0.25)+
  scale_y_continuous(n.breaks = 10,limits = c(min(min_val,0),1))+
  guides(fill = guide_legend(ncol = 8)) +
  ylab('Pearson`s r averaged across folds/iterations')+
  ## facet where all three approaches exist
  stat_compare_means(
    data = subset(compare_diffIns_plt, set == "test"),
    comparisons = list(c('FlowTransOP','Consensus space decoders'),
                       c('FlowTransOP','shuffled X')),
    label.y = 0.65,
    method = 'wilcox'
  ) +
  ## facet where "shuffled X" is absent
  stat_compare_means(
    data = subset(compare_diffIns_plt, set == "train"),
    comparisons = list(c('FlowTransOP','Consensus space decoders')),
    label.y = 0.65,
    method = 'wilcox'
  ) +
  stat_summary(fun = mean, geom = "crossbar",
               width = 0.3, color = "black",linetype='dashed',
               fatten = 2) +
  facet_wrap(~set,scales = 'free_x')+
  theme(axis.title.x = element_blank(),
        legend.position = 'bottom',
        legend.text = element_text(size=14),
        legend.title = element_blank(),
        axis.text = element_text(size=14),
        axis.title = element_text(size=16),
        axis.text.x = element_text(angle = 15),
        legend.box.margin = margin(t = -30, r = 0, b = 0, l = 0))
p <- pv/pd
print(p)

ggsave('../transact_decoders_paired_vs_generalizedtransop_diffInputs.png',
       plot=p,
       width = 28,
       height = 20,
       units = 'cm',
       dpi=600)


## Compare using TRANSACT and STRUCTURE to drive the flow constraining------------------
files <- list.files('../results/AutoTransOP_CellPairs/',full.names = TRUE)
results <- data.frame()
for (file in files){
  tmp <- data.table::fread(file) %>%
    select(-V1)
  results <- rbind(results,tmp)
}
results <- results %>% gather('set','r',-fold,-translation)
results <- results %>% filter(!(set %in% c('AutoTransOP','shuffled X','Direct_pearson'))) %>%
  mutate(approach = 'TRANSACT')

## Load STRUCTURE results
files_dec_structure <- list.files('../results/AutoTransOP_withSTRUCTURE/',full.names = TRUE)
files_dec_structure <-  files_dec_structure[grepl('translation',files_dec_structure)]
results_structure <- data.frame()
for (file in files_dec_structure){
  tmp <- data.table::fread(file) %>%
    mutate(approach='STRUCTURE') %>%
    select(-folder,-fold_id)
  results_structure <- rbind(results_structure,tmp)
}
results_structure <- results_structure %>% gather('set','r',-fold,-translation,-approach)
results_structure <- results_structure %>% select(all_of(colnames(results))) %>%
  mutate(approach = ifelse(set=='shuffled X','shuffled X',approach)) %>%
  mutate(set = ifelse(set=='shuffled X','test',set))

## plot comparison
combined_results_plotting <-  rbind(results,results_structure)
combined_results_plotting$approach <- factor(combined_results_plotting$approach,
                                             levels = c('TRANSACT','STRUCTURE','shuffled X'))
combined_results_plotting <- combined_results_plotting %>%
  mutate(id = paste0(set,'_',approach,'_',translation))

pv <- ggviolin(combined_results_plotting,
          x = 'approach', y = 'r',fill='grey') +
  geom_boxplot(width=0.2)+
  scale_y_continuous(n.breaks = 10,limits = c(NA,1))+
  ylab('Pearson`s r')+
  ## facet where all three approaches exist
  stat_compare_means(
    data = subset(combined_results_plotting, set == "test"),
    comparisons = list(
      c("TRANSACT", "STRUCTURE"),
      c("STRUCTURE", "shuffled X")
    ),
    label.y = 0.65
  ) +
  ## facet where "shuffled X" is absent
  stat_compare_means(
    data = subset(combined_results_plotting, set == "train"),
    comparisons = list(
      c("TRANSACT", "STRUCTURE")
    ),
    label.y = 0.65
  ) +
  facet_wrap(~set,scales = 'free_x')+
  theme(axis.title.x = element_blank(),
        axis.text = element_text(size=14),
        axis.title = element_text(size=16))

min_val <- floor(min(combined_results_plotting %>% 
  group_by(translation,approach) %>% mutate(r=mean(r)) %>%
  ungroup() %>% select(-fold) %>% unique() %>% select(r)))
pd <- ggdotplot(combined_results_plotting %>% 
            group_by(set,translation,approach) %>% mutate(r=mean(r)) %>%
            ungroup() %>% select(-fold) %>% unique(),
          x = 'approach', y = 'r',fill='translation',
         position = position_dodge(0.025))+
  geom_line(aes(x=approach,y=r,group = translation),
            color = 'gray50', size = 0.75, alpha = 0.25)+
  scale_y_continuous(n.breaks = 10,limits = c(min(min_val,0),1))+
  guides(fill = guide_legend(ncol = 6)) +
  ylab('Pearson`s r averaged across folds')+
  ## facet where all three approaches exist
  stat_compare_means(
    data = subset(combined_results_plotting, set == "test"),
    comparisons = list(
      c("TRANSACT", "STRUCTURE"),
      c("STRUCTURE", "shuffled X")
    ),
    label.y = 0.65
  ) +
  ## facet where "shuffled X" is absent
  stat_compare_means(
    data = subset(combined_results_plotting, set == "train"),
    comparisons = list(
      c("TRANSACT", "STRUCTURE")
    ),
    label.y = 0.65
  ) +
  stat_summary(fun = mean, geom = "crossbar",
               width = 0.3, color = "black",linetype='dashed',
               fatten = 2) +
  facet_wrap(~set,scales = 'free_x')+
  theme(axis.title.x = element_blank(),
        legend.position = 'bottom',
        legend.text = element_text(size=14),
        legend.title = element_blank(),
        axis.text = element_text(size=14),
        axis.title = element_text(size=16))

p <- pv/pd
print(p)

ggsave(filename = '../structure_vs_transact.png',
       plot=p,
       width = 28,
       height = 20,
       units = 'cm',
       dpi=600)
