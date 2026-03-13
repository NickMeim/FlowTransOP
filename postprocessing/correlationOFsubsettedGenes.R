library(tidyverse)
library(ggplot2)
library(ggpubr)

## Load example data----------
initial_corr <- readRDS('InitialCorrelationOfGenesSubsets.rds') %>% 
  filter(set=='validation')  %>% filter(gene_1!=gene_2)
gc()
# initial_corr <- data.table::fread('InitialCorrelationOfGenesSubsets.csv') %>%
#   select(-V1)
# colnames(initial_corr)
initial_corr <- initial_corr %>% 
  mutate(p1 = paste(gene_1,gene_2,sep=' , ')) %>%
  mutate(p2 = paste(gene_2,gene_1,sep=' , ')) %>%
  filter(p1!=p2) %>%
  select(-p1,-p2)
initial_corr <- distinct(initial_corr %>% group_by(gene_1,gene_2,set,cell,fold) %>%
  mutate(r = mean(r)) %>% select(-iteration))

## Visualize
p <- ggviolin(initial_corr,
          x='cell',y='r')+
  geom_boxplot(outliers = FALSE,width=0.25)+
  scale_y_continuous(n.breaks = 20,limits = c(-1,1))+
  facet_wrap(~set,ncol = 1)+
  ylab('r') + xlab('cell line')+
  theme(axis.text.x = element_text(angle = 20))
ggsave('../figures/initial_corr_test_subGenes.png',
       plot = p,
       width = 20,
       height = 12,
       units = 'cm',
       dpi = 600)
