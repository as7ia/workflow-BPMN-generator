# Fine-tuning preparation audit

## ProcessModel 1.0

The target representation is strict JSON with globally identified participants,
lanes, scopes, flow nodes, sequence flows, and message flows. Supported node
types are start/end events, tasks, subprocesses, exclusive gateways, and
parallel gateways. Conditions remain human-readable labels. Back-edges encode
loops directly. A nested scope links an existing subprocess node to its internal
graph through `parent_scope_id` and `subprocess_node_id`.

Pydantic enforces shape and primitive constraints. `validate_process_graph`
adds global ID uniqueness, participant/lane/scope ownership, subprocess
hierarchy, reference integrity, same-scope sequence flows, per-scope start/end
and reachability, message-flow eligibility, and XOR/AND gateway rules. Training
references must pass both layers without weakening either one.

## Existing inference and evaluation

The production Ollama adapter sends the ProcessModel JSON Schema, uses
deterministic generation options, parses exact JSON, and applies Pydantic plus
graph validation. Optional repair is bounded to one additional call. These
runtime checks remain authoritative after fine-tuning.

The 12-case evaluation corpus measures schema validity, graph validity,
optional serialization/layout validity, and ID-independent business fidelity.
It contains repeated references and paraphrases by design. Every case,
description, paraphrase, expected rubric, and referenced process is held out
from supervised construction.

## Training risks and controls

- Six seed examples validate tooling but are not enough for a useful model.
- Synthetic descriptions and models can encode author bias; all are explicitly
  marked and require review before expansion.
- Exact validation does not prove semantic equivalence. New references need
  independent review of actors, order, branches, joins, loops, messages, and
  subprocess boundaries.
- Duplicate descriptions and semantic model fingerprints are rejected.
- Similarity and fingerprints reduce leakage risk but cannot prove absence of
  conceptual overlap; human review remains necessary.
- Model-generated candidates are never promoted to labels without independent
  reconstruction, validation, provenance, and review.
- QLoRA compatibility and memory depend on the actual Kaggle image and assigned
  GPU. The mandatory preflight reports facts and refuses an unready run.
