# EHR templates as shortcuts in clinical-note models

Code for the paper **Clinical language models exploit documentation artifacts as hidden shortcuts**. We show that LLMs fine-tuned to predict 30-day
readmission from discharge summaries use the EHR template a note was written in
as a shortcut, and that aggregate evaluation hides it.

## Layout

| Directory | What it does | Paper section |
|---|---|---|
| `template_selection/` | Link notes to templates, build the eligible template pool, build the five training pairs, sample the held-out templates | Methods: template identification, primary pairs, held-out templates |
| `synthetic_notes/` | Extract each note into a standardized 62-field schema with an LLM, verify it, and render it deterministically into two templates with identical content | Methods: synthetic template-controlled datasets |
| `finetuning/` | Patient-level, rate-matched splitting; LoRA fine-tuning and evaluation; the synthetic bias sweep; within/cross-template AUC decomposition | Methods: data splitting, model training, subgroup analysis, balancing |
| `gradient_scaling/` | Two-head model with a gradient-scaled template head (ShorT, Brown et al. 2023) and its analysis | Methods: mitigation 3 |
| `probing/` | Linear probes for template identity on frozen representations | Methods: representation analysis |
| `figures/` | Scripts for Figures 2-5 | |

## Setup

```bash
uv sync
```

Python >= 3.10. Fine-tuning full-length notes needs a GPU with about 40 GB of memory.
The LLM steps (`synthetic_notes/`, `template_selection/classify_template_pool.py`)
call an Azure-OpenAI-compatible gateway; set `LLM_API_BASE_URL` and put the keys
(`LLM_API_KEY_1`, `LLM_API_KEY_2`) in the environment or in a local `.env`.

## Data

The clinical notes are protected health information and are not distributed.
Scripts expect local inputs under `data/` (git-ignored) and print only counts and
rates, never note text. The study used an extract from an Epic Clarity
database; to run on another institution's data, provide these files in
`data/raw/`:

| File | Contents | Required columns |
|---|---|---|
| `filtered_notes.parquet` | One row per discharge summary (deaths excluded) | `note_id`, `note_csn_id`, `pat_enc_csn_id`, `pat_id`, `full_note` |
| `template_ids.pkl` | Pandas DataFrame linking notes to the SmartPhrase templates they were generated from | `note_id`, `smartphrases_id` (list of template ids) |
| `smartphrases.pkl` | Template shells | `smartphrase_id` (or `template_id`) and the shell text in `template` or `full_note` |
| `readmission.pkl` | Pandas DataFrame of encounter outcomes | `pat_enc_csn_id`, `readmission` (number of inpatient readmissions within 30 days; > 0 is a positive) |
| `confounder_spans.csv` | Optional. Templated / copied character spans per note, used only by the non-baseline extraction conditions | `note_csn_id`, `start`, `end`, `type` |

Template ids are the Epic SmartPhrase ids. The ids hard-coded in this repository
(the training templates A-D and held-out templates H1-H10) are specific to the
study institution.

## Pipeline

Run everything from the repository root.

### 1. Templates, pairs and held-out set

```bash
uv run python template_selection/build_note_template_parquet.py --with-text   # notes + template ids
uv run python template_selection/template_pool.py --frame-out template_sampling_frame.csv \
    --dump-shells template_pool_txt                                    # eligible pool (>= 100 notes)
uv run python template_selection/classify_template_pool.py                    # specialty label per template
uv run python template_selection/build_pair_parquets.py                       # data/pairs/<A>_<B>.parquet
uv run python template_selection/build_heldout_parquet.py --with-pat-id      # data/heldout/heldout_notes.parquet
```

The ten held-out templates are fixed in `template_selection/heldout_templates.py`.
`sample_heldout_templates.py` shows how candidates were screened (note count,
shell length, section structure, attestation filter, TF-IDF dissimilarity) and
draws a stratified random set, but it does not regenerate the paper's draw; see
its docstring.

### 2. Splits

```bash
bash finetuning/split_pairs.sh          # 70/15/15 by patient, per-template rates matched
uv run python finetuning/split_report.py data/pairs/*/
```

Splits use seed 42; per-template readmission rates differ by at most 0.1 pp across
the three splits. `finetuning/balanced_group_split.py` is the splitting algorithm; `--audit-only`
prints what any patient-grouped split could achieve on a dataset.

### 3. Fine-tuning and evaluation

```bash
uv run python -m finetuning.train \
    --train-data data/pairs/206596_229657/train.parquet \
    --val-data   data/pairs/206596_229657/val.parquet \
    --test-data  data/pairs/206596_229657/test.parquet \
    --output-dir models/206596_229657
```

Defaults are the paper's settings (Qwen3-0.6B, LoRA r=16, alpha=32, LR 2e-4,
4 epochs, effective batch size 4). Every model in the paper uses LR 2e-4, so the
scale and pretraining comparisons only change `--model-name`:

| Model | `--model-name` | Extra flags |
|---|---|---|
| Qwen3-0.6B (default) | `Qwen/Qwen3-0.6B` | |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | `--max-seq 9000` |
| Qwen3-4B | `Qwen/Qwen3-4B` | `--max-seq 9000` |
| Llama-2 13B | `meta-llama/Llama-2-13b-hf` | `--max-seq 9000` |
| MeLlama 13B | local path to `MeLLaMA-13B-chat` | `--max-seq 9000` |

Every model except the default Qwen3-0.6B truncates notes to 9,000 tokens; the 0.6B
model sees full notes.
MeLlama is distributed through PhysioNet (credentialed access), not the Hugging Face
Hub, so download it there and pass the local directory. Use
`--balance-readmission-rates` for the balancing mitigation. The TRACE arm trains
on `data/pairs/<pair>_removed.parquet`, produced by running TRACE (Cahoon et al.)
on each pair file, then split with `SUFFIX=_removed bash finetuning/split_pairs.sh`.

### 4. Synthetic template sweep

```bash
uv run python synthetic_notes/run_pool_reformat.py --gcs_parquet <notes> --exclude_phrase <phrase> \
    --n 3000 --output pool_reformat.csv --model gpt-5-5 --repair      # step 1: extraction
uv run python synthetic_notes/run_validation.py --reformat_csv pool_reformat.csv ...   # VeriFact-style check
uv run python synthetic_notes/make_exact_arms.py --input pool_reformat.csv --out_prefix pool_exact
uv run python synthetic_notes/build_paired_pool.py --arm-a-csv pool_exact_armA_input.csv \
    --arm-b-csv pool_exact_armB_input.csv ...                        # step 2: render S1 / S2

uv run python -m finetuning.sweep_p_train \
    --template-a-path data/paired_pool/template_a.parquet \
    --template-b-path data/paired_pool/template_b.parquet \
    --p-trains 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --group-column pat_id --match-rates --match-eval-bias \
    --results-dir results/synthetic_sweep
uv run python finetuning/auc_decomposition.py results/synthetic_sweep
```

`uv run python synthetic_notes/restructure.py --selftest` checks, without any patient
data, that the two rendered templates carry identical content.

### 5. Gradient scaling

```bash
sbatch gradient_scaling/run_gradient_scaling.sbatch        # or run_gradient_scaling_local.sh
uv run python -m gradient_scaling.analyze --runs-root ~/gradient_scaling_runs
```

### 6. Linear probe

```bash
uv run python -m probing.linear_probe --model-path models/206596_229657 --data-source presplit ...
```

### 7. Figures

```bash
cd figures
uv run python figure2_pipeline_schematic.py && uv run python figure2_synthetic.py
uv run python figure3_main.py
uv run python figure4_scale_pretraining.py
uv run python figure5_mitigation.py
```

The figures as they appear in the paper are in `figures/figure_1/` ... `figures/figure_5/`
(Figure 1 is a hand-drawn overview). Each script writes `figure_N/figureN.{png,pdf}`.
The scripts draw from stored results rather than from model outputs:

- `figure2_synthetic.py`: the synthetic sweep, as written by `auc_decomposition.py`
  (pass `--csv` to plot a new sweep).
- `auc_data.py`: aggregate and per-template test AUROCs for every pair and model,
  used by Figures 3-5.
- `figure3_main.py`: ROC curves in panels b-c are simulated to match the reported
  AUROCs, and the held-out points in panels f-g were digitized from the original
  analysis plots.
- `figure5_mitigation.py`: the gradient-scaling panels hold placeholder values.

## Not included

- Cohort extraction from Clarity and the 30-day readmission labels.
- TRACE, which is a separate tool (Cahoon et al.).
- Evaluation of the trained models on the held-out templates (Fig. 3e-g,
  Table 2).

## License

MIT, see `LICENSE`.
