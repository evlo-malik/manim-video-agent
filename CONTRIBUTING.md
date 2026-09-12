# Contributing

Contributions that improve reliability, isolation, provider portability, or output quality are welcome.

## Set up

Install Manim's system dependencies, then create a Python 3.11–3.13 environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the local checks before opening a pull request:

```bash
ruff check .
ruff format --check .
pytest
python -m build
```

Keep provider calls out of unit tests. Use small fixtures and mocks for code generation, vision review, and speech synthesis. Changes to prompts should explain the failure mode they address and include a reproducible input when possible.

## Security changes

Generated code execution defines the project's main trust boundary. Discuss substantial sandboxing changes in an issue before implementation. Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
