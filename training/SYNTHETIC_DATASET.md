# Deterministic synthetic supervised dataset

The generate_synthetic_grammar.py script constructs the English description
and ProcessModel together from one deterministic business-process
specification. It does not paraphrase an independently produced graph and does
not use model output as ground truth.

## Families and controls

The grammar covers variable-length sequences, two-to-five-way XOR decisions
with explicit non-default conditions, two-to-five-branch AND split/join blocks,
back-edge loops, two-party collaborations with task-to-task messages, and
subprocesses whose internal scope can contain a sequence, XOR, or AND block.
Twelve business domains rotate across genuinely different topologies.

Each candidate passes Pydantic and validate_process_graph, explicit
text coverage for every business element, evaluation leakage checks, a
vocabulary-independent topology check, and description deduplication.

The train/validation split holds out complete structural groups, such as an
XOR branch-count group, a loop-body-size group, a message-count group, or a
subprocess-internal pattern. No split group occurs in both JSONL files.
For every BPMN family with at least two structural groups, the builder keeps at
least one complete group in train and one in validation. Families that cannot
be split without leaking a group are reported under `unsplittable_families`;
the builder never copies such a group across both sets.

## Build

    python training/generate_synthetic_grammar.py
    python training/build_supervised_dataset.py --source training/artifacts/synthetic_references.json --output training/artifacts/synthetic

Generated references, reports, JSONL, and summaries remain under
training/artifacts, which is intentionally ignored by Git. The generator never
pads the dataset to a requested count.

## Limitations

Structural validation cannot prove that synthetic business scenarios are
natural or operationally complete. The text-graph check proves explicit
coverage, not deep semantic equivalence. The topology fingerprint is a strong
iterative graph invariant, not a formal graph-isomorphism proof. The domains
and sentence templates can still introduce stylistic bias, and collaborations
currently contain two participants. Human sampling and evaluation on the held
out corpus remain necessary before treating a fine-tune as improved.
