# Postprocessing and Plotting

This folder contains R and Python scripts used to summarize model outputs,
compute statistics, and generate manuscript/supplementary figures.

Run scripts from this folder unless otherwise noted.

## R Dependencies

Install the packages used across the plotting scripts:

```r
install.packages(c(
  "tidyverse", "ggplot2", "ggpubr", "patchwork", "cowplot",
  "rstatix", "lme4", "emmeans", "ggridges", "ggsignif"
))
```

## L1000 Benchmark Figures

These scripts summarize outputs from `../results/` and create the L1000
benchmark figures used for the manuscript.

```bash
Rscript evaluate5folds.R
Rscript LowPairsPerformance.R
Rscript DifferentInputsPerformanceBracketed.R
Rscript GPU_vs_CPU_implementation.R
```

Related helper scripts:

```text
CalculateCorrelationOfGenesSubsets.py
CalculateSimilarityInBrackets.py
CreateHighCorrelationSets.py
BracketedSubsettingOfGenes.py
correlationOFsubsettedGenes.R
```

Typical figure outputs are written to `../figures/` or the repository root,
depending on the historical script.

## ARCHS4 Evaluation Figures

Use these scripts after the ARCHS4 evaluation CSVs have been generated in
`../archs4/evaluation/`.

```bash
Rscript plot_archs4_evaluation.R
Rscript plot_archs4_liver_evaluation.R
Rscript plot_liver_mas_fibrosis_final_expression_mean.R
```

Expected inputs:

```text
../archs4/evaluation/cycle_*_persample_fold*.csv
../archs4/evaluation/orthologue_*_fold*.csv
../archs4/evaluation/expression_mmd_fold*.csv
../archs4/evaluation/liver_*_fold*.csv
../archs4/evaluation/liver_mas_fibrosis_final_expression_mean/*.csv
```

Main outputs:

```text
../archs4/evaluation/figures_flowtransop/
../archs4/evaluation/figures_liver/
../archs4/evaluation/liver_mas_fibrosis_final_expression_mean/figures/
```

The plotting scripts save both PNG and PDF outputs where applicable.

## Statistical Tests Used in Plots

- L1000 paired task comparisons use paired tests or mixed-effects models as
  described in the figure legends.
- ARCHS4 cycle, orthologue, and MMD comparisons use paired one-sided Wilcoxon
  signed-rank tests across folds.
- MASH drug-treatment boxplots use one-sided Wilcoxon rank-sum tests comparing
  drug-treated samples against disease controls.

## Replotting Checklist

1. Confirm all expected CSVs exist under `../results/` or `../archs4/evaluation/`.
2. Run the relevant R script from this folder.
3. Check newly written PNG/PDF files under the output folders above.
4. When updating manuscript panels, use the PDF versions for vector-quality
   assembly where possible.
