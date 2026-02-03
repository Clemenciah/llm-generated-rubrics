# GER-Eval Metrics


This repository provides reference implementations of the evaluation metrics introduced in Learning to Judge: LLMs Designing and Applying Evaluation Rubrics.

Large language models (LLMs) are increasingly used as evaluators for natural language generation, typically applying human-defined rubrics to assess system outputs. GER-Eval investigates whether LLMs can instead design and apply their own evaluation rubrics, and how such LLM-defined criteria compare to human-defined ones in terms of reliability and alignment.

The README is intentionally focused on two things:
1) **[How to reproduce / rerun the pipeline](#1-reproducing-the-experiments-rerun-end-to-end)**
2) **[How to plug in your own datasets and LLMs](#2-extending-the-framework-add-your-datasets--llms)**


> Tip: If you only want to analyze existing outputs, jump to **[`final_results/`](final_results/)** (precomputed runs + summaries) and **[`datasets/`](datasets/)** (data used for experiments).

---

## 1) Reproducing the experiments (rerun end-to-end)

### 1.1 Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install python-dotenv
```

### 1.2 Set API keys
Create a `.env` file in the repo root:
```bash
OPENAI_API_KEY=...
# optional: OPENAI_MODEL=...

MISTRALAI_API_KEY=...
DEEPINFRA_API_KEY=...
```

### 1.3 Run the full pipeline (paper-style)
The pipeline runs in **five stages**:
1) metric/rubric generation (**definition**)
2) instruction generation (**instructions**)
3) scoring (**scoring**)
4) tagging student metrics vs teacher metrics (**tagging**)
5) computing tag-based summary scores (**tag_scores**)

If you want the easiest “run it all again” option, use the provided runners:
```bash
python definement_total_runner.py
python instruction_total_runner.py
python scoring_total_runner.py
python tagging_total_runner.py
python tag_scores_runner.py
```

These runners iterate over datasets and settings, and write outputs into `results/`.

### 1.4 Run a single experiment (quick sanity check)
If you want to validate your environment before running everything, run one dataset + one model:

**A) Generate a rubric (“definition”)**
```bash
python scripts/definement.py --data SummEval --api gpt-4o-mini --number_of_shots 3 --use_restriction
```

**B) Generate instructions for that rubric**
```bash
python scripts/instructions.py --data SummEval --api gpt-4o-mini --setting_name <DEFINITION_FILE.json>
```

**C) Score dataset outputs**
```bash
python scripts/scoring.py --data SummEval --api gpt-4o-mini --setting_name <DEFINITION_FILE.json> --use_sample --number_of_shots 5
```

**D) Tag student metrics to teacher metrics**
```bash
python scripts/tagging.py --data SummEval --setting_name <DEFINITION_FILE.json>
```

**E) Compute summary tag scores (coverage/diversity/unseen)**
```bash
python scripts/tag_scores.py --data SummEval --setting_name <DEFINITION_FILE.json> --use_similarity True --similarity_threshold 0.82 --device cpu
```

### 1.5 Where outputs go
By default, outputs follow stage-specific folders under `results/` (created automatically). Typical paths:
- `results/definition/<DATASET>/...json`
- `results/tag/<DATASET>/...json`
- `results/tag/<DATASET>/scores/{embedding|name}/...json`
- plus summary CSVs in the `scores/` folders

### 1.6 Using `final_results/` for analysis (no rerun required)
The repo includes a `final_results/` directory containing **precomputed outputs** (and often aggregated summaries) that you can use to:
- compare settings without rerunning the LLM calls
- compute additional statistics
- build plots/tables for your own analysis

If you’re doing new analyses, start from `final_results/` and/or copy artifacts into your own analysis scripts/notebooks.

---

## 2) Extending the framework (add your datasets & LLMs)

This codebase is dataset-driven: most scripts assume that each dataset has a corresponding **`data_information/`** package (pickled artifacts such as prompts, metric definitions, and other cached metadata). So the core workflow is:

1) **Check compatibility** → 2) **Create `data_information/` pickles** → 3) **Run the pipeline**.

### 2.1 Add your own dataset

#### Step 0 — Check whether your dataset is compatible
Before you touch anything, compare your dataset to the existing ones (in `datasets/` + `data_information/`) and confirm you can provide the same *logical fields* the pipeline relies on, typically:
- an input prompt (or instruction)
- a model output / candidate response to be judged
- optionally: a reference / gold text (for some tasks)
- a **teacher rubric** (ground-truth metrics) if you want alignment/coverage analyses

If your dataset can be represented in the same “prompt → response (→ reference)” structure as existing datasets, it’s usually compatible.

If not, you can still add it, but you’ll likely need to **adapt the dataset loader + prompt construction** (see “When things don’t match” below).

#### Step 1 — Add raw data under `datasets/`
Create:
```
datasets/<YourDatasetName>/
```

Place your raw files there (often `.csv`). Many runners also expect a smaller subset:
- `test.csv` (recommended) for quick runs and controlled evaluation.

#### Step 2 — Create the required `data_information/` artifacts (pickles)
The pipeline expects preprocessed dataset “information” to exist under:
```
data_information/<YourDatasetName>/
```

These files are **pickle** objects containing the prompts, metrics/teacher rubric objects, and other dataset-specific cached structures the scripts read at runtime.

To create them, use the provided notebook:
- **`create_dataset_informations.ipynb`** (canonical way to generate the pickles)


#### Step 3 — (Optional but recommended) Write instructions / extract metric info
If you have a teacher rubric (human-defined metrics), make sure you can:
- represent the metrics in the same format as the existing datasets (names + descriptions + scale)
- generate or provide evaluation instructions per metric (used by the `instructions` stage)

This is important because the tagging + tag-score stages compare student rubrics to teacher metrics.

#### Step 4 — Register the dataset in the loader / CLI
Once your `data_information/<YourDatasetName>/` exists, wire the dataset name into the code so it can be selected via CLI:
- add `<YourDatasetName>` where dataset choices are defined in `scripts/*.py`
- add a corresponding loader entry in the dataset loader (e.g., `data/data_loader.py`) pointing to:
  - `datasets/<YourDatasetName>/...`
  - `data_information/<YourDatasetName>/...`

#### Step 5 — Smoke test
Run a minimal end-to-end test:
```bash
python scripts/definement.py --data <YourDatasetName> --api gpt-4o-mini --number_of_shots 1
python scripts/tag_scores.py --data <YourDatasetName> --setting_name <DEFINITION_FILE.json> --device cpu
```

If this works, your dataset is correctly wired.

#### When things don’t match (expected & normal)
Even if your dataset is “compatible,” you may still need light adaptation, for example:
- renaming columns / fields to match what the loaders expect
- tweaking prompt formatting (so the LLM sees the same structure used in the paper runs)
- adjusting sampling (few-shot selection) logic for your dataset size or splits

If your dataset differs substantially (multi-turn dialogues, multiple references, etc.), you may need to:
- add a dataset-specific adapter in the loader
- modify prompt builders in `util/prompting_utils.py`
- update formatting/parsing in `util/formatting_tools.py`

---

### 2.2 Add your own LLM / model backend

LLMs are routed through the API abstraction (see `models/api_loader.py`).

#### Step 1 — Add the model
Add a new backend/client that returns model outputs in the same shape used by the existing backends (text completions + optional structured parsing).

Then register it so it can be used via:
```bash
--api <your_backend_name>
```

#### Step 2 — Add credentials (if needed)
Add any new keys to `.env`, for example:
```bash
MY_PROVIDER_API_KEY=...
```

#### Step 3 — Validate
Run a minimal rubric generation call:
```bash
python scripts/definement.py --data SummEval --api <your_backend_name> --number_of_shots 1
```

If this works, the rest of the pipeline typically works too (instructions → scoring → tagging → tag_scores).

#### Step 4 — Expect minor tuning when swapping models
Different LLMs may require small adjustments to get stable parsing and consistent rubric formatting, e.g.:
- stricter prompt constraints / formatting instructions
- tweaks to output parsing rules (if your model is “creative” with structure)
- temperature / decoding changes
---
## Contact

Questions or ideas? You can reach me at **alianpourya@gmail.com** — I’m happy to help.

If you find an issue (bug, reproducibility problem, unclear docs, etc.), please **open a GitHub Issue** so we can track it and fix it.


