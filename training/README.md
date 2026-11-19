# Qwen3-8B QLoRA preparation

This directory prepares supervised ProcessModel 1.0 examples and a Kaggle GPU
QLoRA run. It is isolated from the production API. No training has been run by
these repository changes.

## Audit and representation

The assistant target is the exact compact JSON serialization of a fully
validated `ProcessModel`. The input uses Qwen's native `system`, `user`, and
`assistant` conversation roles. This retains participant and lane ownership,
scope hierarchy, tasks, XOR and AND gateways, sequence-flow conditions and
back-edges, subprocess ownership, and cross-participant message flows.

Every reference is parsed by Pydantic and checked by `validate_process_graph`
before JSONL is written. Generated Qwen output is never accepted as ground
truth. Each seed declares provenance and `synthetic: true`.

The 12 cases in `evaluation/corpus.json`, their descriptions, paraphrases, and
reference processes are reserved exclusively for evaluation. The builder
rejects close description matches and exact semantic fingerprints against this
corpus, as well as duplicate IDs, descriptions, and semantic models inside the
training set.

## Data actually available

`reference_cases.json` currently contains six hand-authored synthetic examples:
linear, exclusive decision, parallel split/join, correction loop, two-party
message flow, and nested subprocess. The generated deterministic split is five
training examples and one validation example. This is enough to test the
pipeline but not enough to claim a useful fine-tune. A serious run requires a
larger independently reviewed corpus with more combinations, depths, actors,
branch counts, loop shapes, and negative ambiguity cases.

The deterministic grammar expansion is documented in
`training/SYNTHETIC_DATASET.md`. It produces the larger training and validation
JSONL files without changing or replacing these six audit seeds.

Build the leakage-checked data locally or on Kaggle:

```powershell
.venv\Scripts\python.exe training\build_supervised_dataset.py
```

## Kaggle commands

Enable a Kaggle GPU accelerator and internet only if model/package download is
needed. Then, from the repository root:

```bash
python -m pip install -r training/requirements-kaggle.txt
python training/build_supervised_dataset.py
python training/train_qlora_kaggle.py --preflight-only
```

Do not proceed unless the preflight reports non-empty train/validation data,
all package versions, `cuda.available: true`, and the actual GPU name and
memory. A typical Transformers run is:

```bash
python training/train_qlora_kaggle.py \
  --backend transformers \
  --model Qwen/Qwen3-8B \
  --max-seq-length 8192 \
  --output /kaggle/working/qwen3-8b-processmodel
```

For Unsloth, first install a release compatible with the current Kaggle
PyTorch/CUDA image, rerun preflight, then select `--backend unsloth`. The script
does not silently fall back when Unsloth is missing. Checkpoints are retained
under the output directory and the final PEFT adapter is saved as
`adapter-final`. Resume with `--resume-from-checkpoint PATH`.

Compare the base and adapter on exactly the 12 reserved cases:

```bash
python evaluation/compare_qlora_models.py \
  --confirm-gpu-evaluation \
  --model Qwen/Qwen3-8B \
  --adapter /kaggle/working/qwen3-8b-processmodel/adapter-final \
  --output /kaggle/working/qlora-comparison.json
```

Add `--with-layout` only when Node.js and the pinned layout worker dependencies
are installed in the Kaggle environment. The comparison records strict JSON
and schema validity, `validate_process_graph` validity, optional BPMN DI result,
and the existing ID-independent semantic fidelity rubric for both models.

## Remaining GPU constraints

QLoRA still requires a CUDA GPU supported by bitsandbytes, enough VRAM for an
8B model in 4-bit plus activations, and compatible PyTorch, CUDA, Transformers,
PEFT, and optionally Unsloth versions. Actual feasibility depends on the GPU
assigned by Kaggle and the token lengths measured from the expanded dataset;
the repository cannot establish either condition in advance.
