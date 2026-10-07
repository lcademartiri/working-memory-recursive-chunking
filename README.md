# Working-memory-gated recursive chunking: minimal reproduction

This repository contains the original MATLAB V8.2 learner and nested-source generator used for the paper's synthetic capacity result. `run_reproduction.m` is a small driver for two experiments:

1. The capacity gate at a fixed input interval (`C = 4:10`, `Δt = τ = 1`, 120 source realizations per capacity).
2. The slow-input check (`C = 4:7`, `Δt = [1 2 4 8 12]`, 50 source realizations per cell).

The two files in `src/` are copied byte-for-byte from the retained V8.2 production handover. The driver uses the same settings and seed formulas as `run_v8_production_core.m` for these *structured-source* conditions. It omits hierarchy-null, grammar-robustness, first-divergence, source-cutoff, and maternal-speech analyses. It is a reproduction of the central synthetic result, not the complete paper pipeline.

## Requirements and commands

MATLAB with `table`, `containers.Map`, and `rng` support. No add-on toolbox is required by these three files. From the repository folder, run:

```matlab
T = run_reproduction('smoke'); % 2 paired realizations, 4 learner runs
T = run_reproduction('gate');  % 840 learner runs, 120 paired realizations
T = run_reproduction('slow');  % 1000 learner runs, 50 paired realizations
% T = run_reproduction('all'); % both full experiments
```

The full experiments can take substantial time. After each source realization, the driver writes a MATLAB checkpoint and a CSV in `results/`. Rerunning the same command resumes completed cells. Do not combine output files from different versions of the source code.

The returned table and CSV contain one row per run, including capacity, input interval, exact source depth reached, maximum learned depth, maximum learned concept span, and mean context amplification. `expected_counts.csv` gives the retained production result. A full `gate` or `slow` run checks the achieved level-5 counts against that file at completion.

## Expected result

At `Δt = 1`, the number of runs reaching an exact level-5 source motif is **2/120 at C = 6** and **120/120 at C = 7**. These counts concern the *maximum exact source level*, not the learner's unconstrained construction level or the raw length of its longest chunk. Across the separate 50-realization slow-input experiment, the corresponding counts at `C = 6` are **2/50 at every tested interval from 1 through 12**; at `C = 7`, they are **50/50**. The two experiments use different source seeds and must not be pooled.

## What the code does

The source consists of 360 top-level episodes expanded through five binary levels, using eight motif types per level, eight raw symbols, recurrence lag four, and one fixed grammar seed. A source realization is generated once per replicate and supplied to every capacity or interval condition within that replicate.

The learner has a FIFO active buffer of `C` symbolic items. An unknown adjacent pair is learned only when two nonoverlapping copies coexist and remain active for processing time `τ = 1`. Confirmed chunks occupy one active slot and can be constituents of new pairs. A known pair also needs one processing unit to be recognized and compressed. Confirmed chunk definitions persist in a separate unbounded dictionary; unknown pairs leave no usable candidate trace after evidence is lost. Pending jobs are tied to their supporting active items. The source hierarchy is used to evaluate exact recovery and is not given to the learner.

Every new pair has the same four-item local evidence requirement. The gate at six to seven slots measures whether the sequence of lower-level discoveries allows the required items to become available at later levels; it does not reflect an increase in the arity or active footprint of the learning operation.

## Provenance and sharing

The source files are `wm_bootstrap_model_v8.m` (header: `Patch level: V8.2`) and `generate_nested_source_v8.m` from the paper handover dated 2026-09-15. `expected_counts.csv` was calculated from its retained `capacity_gate_runs.csv` and `slow_input_runs.csv` for the structured condition. The runner was added for this repository. No MATLAB executable was available in the preparation environment, so execution was checked against the production configuration and archived CSVs, but the packaged driver was not run here.

No license has been added. The copyright holders should choose a license before making a public GitHub repository if they want others to be able to reuse or modify the code.
