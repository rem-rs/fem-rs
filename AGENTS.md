# Python Environment

This project uses **uv** (https://github.com/astral-sh/uv) to manage Python environments and package management.

## Setup

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or on Windows:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   uv pip install -r pyproject.toml  # if pyproject.toml exists
   # or
   uv pip install <package-name>
   ```

## Key Commands

- `uv venv` — Create a virtual environment in `.venv/`
- `uv pip install <pkg>` — Install a package
- `uv pip install -r requirements.txt` — Install from requirements file
- `uv pip list` — List installed packages
- `uv run <script.py>` — Run a script with the environment's Python
- `uv build` — Build Python packages
- `uv publish` — Publish packages to PyPI

## Notes

- The `.venv/` directory is git-ignored.
- Python bindings for this project are built using **maturin** and installed via `uv pip install`.

## Building Python Bindings

The Python bindings use PyO3 + maturin, located at `crates/python/`.

1. Ensure a Python 3.11+ interpreter is available:
   ```bash
   uv python install 3.11
   ```

2. Build the development version:
   ```bash
   uv venv
   uv pip install maturin numpy scipy
   maturin develop
   ```

3. Or set `PYO3_PYTHON` to point to the uv-managed Python before `cargo build`:
   ```powershell
   $env:PYO3_PYTHON = "$(uv python find 3.11)"
   cargo build -p fem-py
   ```
