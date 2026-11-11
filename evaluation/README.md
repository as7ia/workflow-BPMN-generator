# Qwen3 ProcessModel semantic evaluation

This directory is an isolated benchmark for measuring how faithfully a real
local Qwen model converts English business descriptions into `ProcessModel`
schema 1.0. It is not imported by production code and regular pytest never
performs inference.

## Corpus and rubric

`corpus.json` contains 12 manually authored descriptions covering linear
flows, exclusive decisions, parallel split/join, correction loops,
two-participant messages, nested subprocesses, lanes, three-party
collaboration, and combined processes. Every case links to a valid repository
reference model and declares expected semantic properties separately.

The scorer compares normalized human-readable names and topology rather than
requiring the model to reproduce reference IDs. Eight equally weighted rubric
dimensions are reported from 0 to 1:

1. Required activities preserved
2. Participants and lanes identified
3. Gateway types, counts, and branch conditions preserved
4. Parallel split and join topology preserved
5. Loop back-edge preserved
6. Message-flow direction, endpoints, and label preserved
7. Subprocess ownership and nested activities preserved
8. Missing and invented activity penalty

The report keeps four concepts distinct:

- JSON/schema validity: the candidate matches `ProcessModel` 1.0.
- Graph validity: deterministic domain invariants pass.
- BPMN DI validity: optional `generate_bpmn()` layout succeeds.
- Semantic fidelity: the rubric score against manually declared expectations.

A high structural score is not proof that the English meaning was understood.
Review the saved candidate and missing/invented element lists for every case.

## Offline Windows execution

Prerequisites on the supervisor's computer:

- 64-bit Windows with enough RAM for the selected Qwen model
- Python 3.11+ and this project installed in `.venv`
- Node.js 20.12+ and `npm.cmd ci --ignore-scripts` already completed in
  `app\bpmn\layout_worker` if `--with-layout` is used
- Ollama installed separately
- The selected model weights explicitly installed before disconnecting from
  the network; the benchmark never downloads or starts a model
- Sufficient disk space for weights and Ollama's local cache

Configure the existing adapter through environment variables:

```powershell
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "qwen3:8b"
$env:OLLAMA_TIMEOUT_SECONDS = "120"
$env:OLLAMA_NUM_PREDICT = "4096"
$env:OLLAMA_NUM_CTX = "8192"
```

Run all 12 cases, exactly one attempt per case:

```powershell
.venv\Scripts\python.exe evaluation\run_qwen_evaluation.py --confirm-live-ollama
```

Include real semantic serialization and DI validation:

```powershell
.venv\Scripts\python.exe evaluation\run_qwen_evaluation.py --confirm-live-ollama --with-layout
```

Use another installed model or a smaller bounded sample without editing code:

```powershell
.venv\Scripts\python.exe evaluation\run_qwen_evaluation.py `
  --confirm-live-ollama `
  --model qwen3:14b `
  --timeout 180 `
  --num-predict 4096 `
  --num-ctx 8192 `
  --max-cases 3
```

Results are written under ignored `evaluation/results/` files. Each case saves
latency, categorized errors, the unmodified candidate, structural validity,
optional DI validity, category scores, and missing/invented activities. There
are no retries and no repair step.

## Safe checks without Ollama

```powershell
.venv\Scripts\python.exe -m pytest tests\evaluation\test_evaluation_corpus.py
```

This validates the corpus, all reference models, and the scorer. It does not
load Qwen, call Ollama, start Node, or demonstrate semantic accuracy.
