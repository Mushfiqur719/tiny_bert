# Contributing to TinyBanglaBERT

Thank you for your interest in contributing to **TinyBanglaBERT**! We welcome contributions ranging from bug reports and documentation improvements to new distillation techniques and evaluation benchmarks for Bengali NLP.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/tinybanglabert.git
   cd tinybanglabert
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies in Editable Mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify Test Suite**:
   ```bash
   pytest tests/ -v
   ```
   All tests should pass before making any modifications.

---

## 🧪 Guidelines & Code Standards

- **Determinism**: Always pass seeds via `src.utils.seeding.seed_everything(seed)` for reproducible outputs.
- **Config-Driven**: Hyperparameters must be configured via `configs/*.yaml` and parsed through `src.utils.config_loader.load_config`.
- **Typing & Docstrings**: Ensure all new functions include type hints and clear docstrings explaining inputs, returns, and mathematical formulations.
- **Unit Testing**: Add corresponding tests under `tests/` for any new modules, layers, loss functions, or evaluation routines.
- **Code Quality**: Run formatting and linting checks using `ruff`:
  ```bash
  ruff check .
  ruff format .
  ```

---

## 🚀 Submitting Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with meaningful messages:
   ```bash
   git commit -m "Add feature description"
   ```
3. Ensure all tests pass:
   ```bash
   pytest tests/
   ```
4. Push to your fork and submit a Pull Request against `master` / `main`.

---

## 📄 License
By contributing to TinyBanglaBERT, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
