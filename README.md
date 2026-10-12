# prompt-sensitivity-hybrid

Initial research foundation for studying prompt sensitivity across large language models, featuring a unified model-loading interface, strict hardware-enforced 4-bit quantization, reproducible seed management, and cluster-based leakage-free dataset splitting.

---

## 1. Project Overview & Architecture

`prompt-sensitivity-hybrid` is designed for reproducible experimental research into model prompt sensitivity and perturbation robustness.

```
prompt-sensitivity-hybrid/
├── config.yaml               # Validated central experiment & hardware configuration
├── requirements.txt          # Minimal pinned dependencies (PyTorch, Transformers, BNB, PEFT)
├── main.py                   # Unified CLI (check-env, dataset-check, smoke-test)
├── model.py                  # Unified model loading interface (Qwen2.5 & LLaMA-3.1)
├── dataset.py                # Schema definition, CSV loader & leakage-free cluster splitting
├── utils.py                  # Seeding, structured logging, schema validation, manifest
├── data/
│   ├── paraphrased_prompts_axis_tagged_qa.csv  # Source benchmark dataset (preserved unchanged)
│   ├── sample_dataset.jsonl                    # Reference unit-testing dataset
│   └── splits/                                 # Generated canonical dataset & zero-leakage splits
│       ├── canonical_dataset.jsonl             # 1,176 canonical prompt records
│       ├── train.jsonl                         # 137 clusters (822 prompt records)
│       ├── val.jsonl                           # 29 clusters (174 prompt records)
│       ├── test.jsonl                          # 30 clusters (180 prompt records)
│       └── split_metadata.json                 # Manifest with cluster assignments & seed
├── tests/
│   ├── __init__.py
│   ├── test_config.py        # Config schema, fallback prohibition & bounds testing
│   ├── test_dataset.py       # CSV loading, variant mapping, counts, and zero-leakage split tests
│   └── test_model.py         # Model interface, dtype resolution & CUDA enforcement
├── outputs/                  # Diagnostic reports and model test outputs
│   ├── logs/
│   │   └── run.log
│   ├── env_check_report.json
│   ├── dataset_split_report.json
│   └── smoke_test_Qwen_Qwen2.5-7B-Instruct.json
└── README.md
```

---

## 2. Hardware Target & Enforcement

- **Target GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Compute Capability**: `sm_89` (Ada Lovelace architecture)
- **Precision**: Native `bfloat16` and 4-bit NormalFloat (`nf4`) quantization with double quantization enabled.
- **Fail-Fast Policy**: Under no circumstances does the system silently fall back to CPU or alter the requested model. If CUDA is unavailable or VRAM is exhausted, an explicit `RuntimeError` is raised immediately.

---

## 3. Supported Model Interface

Both models are instantiated through the identical interface in `model.py` (`load_model_and_tokenizer`):

1. **`Qwen/Qwen2.5-7B-Instruct`**
   - Open-weights via Hugging Face.
   - Tested and verified on RTX 4090 with 4-bit quantization (7.368 GB VRAM).
2. **`meta-llama/Llama-3.1-8B-Instruct`**
   - Gated repository via Meta / Hugging Face.
   - Requires community license agreement approval on Hugging Face and an active user token (`HF_TOKEN`).

---

## 4. PromptSensitivity Dataset Integration & Canonical Schema

The primary dataset is located at `data/paraphrased_prompts_axis_tagged_qa.csv` and contains wide format columns:
`id, original, lexical, syntactic, pragmatic, length, format`

### Conversion & Representation
1. **Source Preservation**: The source CSV is strictly preserved and unmodified.
2. **Cluster Grouping**: Each row `id` (1 through 196) represents a single semantic question cluster.
3. **Canonical Record Schema** (`PromptRecord` in `dataset.py`):
   - `cluster_id` (str): Preserved original cluster ID (`"1"` through `"196"`).
   - `prompt_id` (str): Unique prompt identifier (e.g. `f"{cluster_id}_{variant_type}"`).
   - `prompt_text` (str): Text of the prompt.
   - `variant_type` (str): `"original"` for column `original`, or one of the 5 linguistic variant axes: `"lexical"`, `"syntactic"`, `"pragmatic"`, `"length"`, `"format"`.
   - `reference_answer` (null): Explicitly `null` because the source CSV contains no answers. No synthetic answers are fabricated.
   - `perturbation_metadata` (dict): Tracks metadata such as `{"axis": ..., "is_variant": ...}`.

### Cluster-Based Splitting (70% Train / 15% Val / 15% Test)
Dataset partitioning is performed strictly at the **cluster level** using a configurable seed (default: 42):
- All 6 versions of each question cluster remain together in the exact same split.
- Zero cluster leakage across partitions:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

### Split Breakdown (196 Clusters, 1,176 Prompt Records)
| Split | Unique Clusters | Prompt Records | Percentage |
| :--- | :--- | :--- | :--- |
| **Train** | 137 | 822 | 69.9% |
| **Val** | 29 | 174 | 14.8% |
| **Test** | 30 | 180 | 15.3% |
| **Total** | **196** | **1,176** | **100.0%** |

---

## 5. Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (tested on Python 3.11.9)
- NVIDIA GPU Driver with CUDA 12.1+ support

### Installation (Windows PowerShell)

```powershell
# 1. Clone / navigate to project directory
cd c:\Users\Student5\Desktop\Mitigation

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.1 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## 6. Execution Commands

### A. Regenerate & Validate Dataset Splits
Load the axis-tagged CSV, convert to canonical format, validate counts and 0-leakage, and save under `data/splits/`:
```powershell
python main.py dataset-check --config config.yaml
```
Custom source or splits directory can also be specified:
```powershell
python main.py dataset-check --data-path data/paraphrased_prompts_axis_tagged_qa.csv --splits-dir data/splits
```

### B. Run Automated Unit Tests (24 Tests)
Execute the complete test suite verifying config validation, CSV conversion, variant mapping, zero leakage, and model constraints:
```powershell
pytest -v
```

### C. Hardware & Environment Audit
Verify CUDA, RTX 4090 recognition, VRAM, and bitsandbytes quantization:
```powershell
python main.py check-env --config config.yaml
```

### D. Model Loading & Inference Smoke Tests
Execute 4-bit quantized loading and forward inference generation on GPU:

**1. Qwen 2.5 7B Instruct:**
```powershell
python main.py smoke-test --model Qwen/Qwen2.5-7B-Instruct --prompt "Explain prompt sensitivity in two sentences."
```

**2. Meta LLaMA 3.1 8B Instruct:**
*Requires Hugging Face token with repository access:*
```powershell
$env:HF_TOKEN = "your_hf_token_here"
python main.py smoke-test --model meta-llama/Llama-3.1-8B-Instruct --prompt "Explain prompt sensitivity in two sentences."
```

---

## 7. Known Blockers & Authentication Requirements

1. **Meta LLaMA 3.1 8B Instruct Access**:
   `meta-llama/Llama-3.1-8B-Instruct` is a gated repository. If run without authentication or prior access approval at `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`, Hugging Face raises `401 Client Error: GatedRepoError`. Set `$env:HF_TOKEN = "<token>"` or log in with `huggingface-cli login`.
2. **Windows NTFS Symlinks**:
   Hugging Face Hub issues a benign warning on Windows if Developer Mode is disabled because Windows requires admin rights or Developer Mode to create symlinks. Models download and cache normally.
