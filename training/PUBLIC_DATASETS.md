# Public BPMN dataset import

This experimental pipeline only admits human-authored text/BPMN pairs whose reuse
terms are explicit and whose BPMN semantics fit ProcessModel 1.0 without loss.
Generated model output is never treated as ground truth.

## Audited sources

| Source | Access verified | Description/model format | Reuse decision |
|---|---|---|---|
| [Zenodo 7783492](https://zenodo.org/records/7783492) | `text_and_bpmn.raw.tgz` downloaded; published MD5 verified | 24 English `.txt` descriptions and 223 scored `.bpmn2.xml` alternatives | Excluded: no license is published in the record or embedded in the archive |
| [PMo dataset](https://zenodo.org/records/15857589) | `pmo-dataset.zip` downloaded; published MD5 verified | 55 English `.txt` descriptions paired with standardized BPMN and eight other representations | Excluded: no license is published in the record or embedded in the archive |
| [Camunda BPMN for Research](https://github.com/camunda/bpmn-for-research) | Repository archive downloaded and English material inspected | Four English exercise PDFs and five reference `.bpmn` solutions | Supported only after explicit acceptance of research-only, non-commercial, attribution-required terms |
| [MaD study](https://link.springer.com/article/10.1186/s43067-025-00288-9) | Article inspected; no public dataset download found | The article describes 30,000 text/model pairs and DOT training targets | Not imported: data is available from the corresponding author on reasonable request; dataset license is unverified |

## Strict conversion policy

The importer rejects the complete pair when a BPMN construct has no faithful
ProcessModel 1.0 representation. Examples include inclusive or event-based gateways,
intermediate and boundary events, event definitions, call activities, annotations,
and data objects. Tasks must have business names, XOR branches must retain explicit
conditions, and message-flow endpoints must be representable. Every converted model
is validated first by Pydantic and then by `validate_process_graph`.

The import also rejects duplicate descriptions, duplicate semantic model fingerprints,
low task-name coverage between text and model, and descriptions or model fingerprints
equivalent to the 12 reserved evaluation cases.

## Usage

Install the isolated import dependency and download the Camunda repository yourself.
Review its terms before setting the acceptance flag:

```powershell
python -m pip install -r training/requirements-public-import.txt
python training/public_bpmn_import.py `
  --camunda-root C:\path\to\bpmn-for-research `
  --accept-camunda-research-only-license `
  --output training/artifacts/public
```

The command writes `camunda_reference_cases.json` and `public_import_report.json`.
Artifacts remain outside the versioned training seeds. The report records candidates,
accepted and rejected counts, rejection reasons, and accepted family/complexity
distributions. Empty output is valid when lossless conversion is impossible.
