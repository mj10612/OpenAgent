# Contributing

Use Python 3.11 or later. Install development dependencies with `python -m pip install -e ".[dev]"`.

Before submitting a pull request, run:

```sh
python -m pytest -q
python -m ruff check src tests
python -m mypy src tests
```

Add a regression test for behavior changes and keep provider tests offline with mocked HTTP transports. Exercise Windows paths/newlines and POSIX permissions where relevant. Describe the user-visible behavior and validation in your pull request.

Report defects in [GitHub Issues](https://github.com/mj10612/OpenAgent/issues). Follow [SECURITY.md](SECURITY.md) for sensitive vulnerability reports.
