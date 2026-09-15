# Agent

Provider adapters, summary learning and robot evaluation for VA-Bench.
See the [root README](../README.md) for setup and full-suite commands.

```bash
python -m agent --help
python -m agent eval robotwin --help
```

Provider configuration uses `AGENT_BASE_URL`, `AGENT_MODEL`, `AGENT_API_KEY`,
and `AGENT_WIRE_API`. `OPENAI_*` and `ANTHROPIC_*` provider variables are also
supported. Run data defaults to `.agent/`; the main launcher uses `outputs/`.
