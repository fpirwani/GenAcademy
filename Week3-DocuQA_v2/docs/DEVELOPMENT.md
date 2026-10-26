# Development and validation

Run commands from `DocuQA_v2`. `uv sync` installs the locked project dependencies;
`uv run streamlit run app.py --server.port 8502` launches the UI.

## Automated checks

```powershell
uv run pytest -q
uv run ruff check .
```

Tests use temporary stores and fake model adapters. They cover document lifecycle,
source isolation, citation validation, numeric rejection, missing evidence, report grouping, saved-pair replacement,
summaries, previews and UI controls.
`test_evidence_pipeline.py` verifies that extraction is independent of display caps
and that numeric validation rejects altered values.

## Live evaluation

Run Ollama with the configured model and compare the sample insurance, course and
mortgage pairs using temporary stores. Check expected values on both sides, source
attribution, qualifications and report grouping. Avoid writing test results into app data.

Before changing a model or prompt, run the same cases and compare expected values,
false absence claims, source attribution, missing answers, latency and summary
quality. Add representative failing examples without adding domain-specific rules
to production code. Training a model is not part of ordinary app maintenance.

## Implementation boundaries

`comparison/workflow.py` routes entry modes and checks selected generations.
`inventory.py` discovers source field names; `holistic.py` validates report facts;
`presentation.py` owns report grouping and contextual difference summaries.
`validation.py` and `independent.py` check source quotations and numbers.

Apply report display limits after extraction. Do not treat missing
extraction as proven absence. Do not accept model-created source identities or
silently approve a fact after review fails. Retain complete source qualifications.
Do not log original documents or model reasoning in routine application logs.

Local data and model weights are not test fixtures and should not be committed.
