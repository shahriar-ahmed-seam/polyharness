# Contributing to PolyHarness

Thank you for your interest in contributing to **PolyHarness**! We welcome contributions to the Agent Data Protocol (ADP) specification, new target harness adapters, anti-overfitting synthesis algorithms, and evaluation benchmarks.

---

## 🛠️ Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/shahriar-ahmed-seam/polyharness.git
   cd polyharness
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies in editable development mode:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. **Verify the test suite and linter:**
   ```bash
   pytest -v
   ruff check src tests examples
   ```

---

## 🏗️ Adding a New Harness Adapter

To implement a new harness adapter (e.g. for a custom framework or agent run format):
1. Create `src/polyharness/adapters/your_harness.py`.
2. Inherit from `BaseAdapter` defined in `src/polyharness/adapters/base.py`.
3. Implement `render()`, `parse()`, and `to_sft_record()`.
4. Register your adapter in `src/polyharness/adapters/__init__.py`.
5. Add unit tests in `tests/test_adapters.py`.

---

## 🧪 Testing Guidelines

- Write thorough tests for any new features or bug fixes.
- Ensure all tests pass with `pytest -v`.
- Adhere to formatting and linting standards using `ruff check` and `ruff format`.

---

## 📄 Pull Request Process

1. Open an issue describing the proposed change or bug.
2. Submit a feature branch PR with descriptive commits.
3. Ensure CI workflows pass on all Python matrix versions (3.10, 3.11, 3.12).
