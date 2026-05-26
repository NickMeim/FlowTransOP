# Results Folder

This folder stores intermediate and final CSV outputs from the L1000 benchmark
experiments. The plotting scripts in `../postprocessing/` read from these
directories to generate benchmark figures and statistical summaries.

The folder is organized by experiment family. Directory names are intentionally
close to the training script names so that outputs can be traced back to the
workflow that produced them.

## Main Result Groups

```text
AutoTransOP_CellPairs/
AutoTransOP_CellPairs_withPairs/
AutoTransOP_CellPairs_diffenetInputs/
AutoTransOP_CellPairs_diffenetInputs_bracketed/
AutoTransOP_withSTRUCTURE/
AutoTransOP_extremely_fewPairs_A375_HT29/

FlowMatch_lowPairsPercentage/
FlowMatch_lowPairsPercentage_withPairs/
FlowMatch_fewPairs_A375_HT29_PairAndSimilarity/
FlowMatch_fewPairs_A375_HT29_PairAndSimilarity_meanAgg/
FlowMatch_fewPairs_A375_HT29_PairAndSimilarity_sumAgg/
FlowMatch_extremely_fewPairs_A375_HT29/
FlowMatch_extremely_fewPairs_A375_HT29_withPairs/
FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity/
FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity_meanAgg/
FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity_sumAgg/

DecodersOnly/
DecodersOnly_differentInputs/
Decoders_only_diffenetInputs_bracketed/

GPU_vs_CPU/
GPU_vs_CPU_random/
LatentDim_30/
```

## How to Recreate

Run the relevant scripts from `../learning/`. Examples:

```bash
cd ../learning
bash cell_pairs_benchmark.sh
bash low_percentage_of_pairs.sh
bash extremely_low_percentage_of_pairs.sh
bash pairedFlow_low_percentage_of_pairs.sh
bash pairedFlow_low_percentage_of_pairs_extreme.sh
bash OneCell_differentInputs_benchmark.sh
bash decoders_only_imputedGenes.sh
bash subsetting_decoders_only.sh
```

The Python scripts write CSV files into the subdirectories listed above. The
exact naming convention usually includes the cell-line pair, direction, model
type, fold, or paired-sample count.

## Plotting Outputs

After result CSVs are present, run the postprocessing scripts:

```bash
cd ../postprocessing
Rscript evaluate5folds.R
Rscript LowPairsPerformance.R
Rscript DifferentInputsPerformanceBracketed.R
Rscript GPU_vs_CPU_implementation.R
```

These scripts create figures in `../figures/` or the repository root, depending
on the historical output path used by each script.

## Notes for New Experiments

- Add new result directories with descriptive names matching the training
  script or model variant.
- Keep fold-level CSV outputs rather than only summary tables; downstream
  statistics often require fold-level or task-level values.
- Do not overwrite manuscript result directories unless intentionally
  regenerating the full analysis.
