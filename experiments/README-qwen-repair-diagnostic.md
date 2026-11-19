# Qwen repair topology diagnostic

This experimental command makes one initial Qwen call and, only when that
candidate is graph-invalid, one repair call. It holds both complete candidates
in memory but prints only an ID-based topology comparison. Production API code,
logs, and errors are not involved.

Put one business description in a local text file, then run:

```powershell
.venv\Scripts\python.exe experiments\qwen_repair_diagnostic.py `
  --confirm-live-ollama `
  --description-file .\local-description.txt
```

The report compares participants, scopes, nodes, and sequence flows; highlights
`nodes[5]` and `scopes[1]`; and flags subprocess nodes added during repair. For
pre-existing subprocesses, the justification check is deliberately conservative:
it requires both an explicit nesting phrase and words from the subprocess name
in the original description. `not_demonstrated` means human review is required,
not that the subprocess is certainly wrong.

Do not redirect the report to a shared location when process names are
sensitive. The tool never downloads a model, starts Ollama, or writes candidates
to disk.
