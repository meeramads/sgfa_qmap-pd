# Documentation

This directory contains the Sphinx documentation for the SGFA qMAP-PD project.

## Building Documentation

### Prerequisites

```bash
pip install -e ".[dev]"
```

This installs Sphinx and all required documentation dependencies.

### Build HTML Documentation

```bash
cd docs
make html
```

The documentation will be built in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view.

### Clean Build

```bash
cd docs
make clean
make html
```

### Auto-rebuild on Changes

For development, you can use sphinx-autobuild:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
```

This will serve the docs at http://localhost:8000 and auto-rebuild on changes.

## Documentation Structure

- `index.rst`: Main documentation index
- `api/`: Auto-generated API documentation
- `examples/`: Usage examples and tutorials
- `experiments/`: Experiment documentation
- `*.md`: Additional documentation (configuration, testing, etc.)

## Adding New Documentation

### Adding a New Module

1. Create a new `.rst` file in the appropriate directory
2. Add it to the `toctree` in `index.rst`
3. Run `make html` to build

### Adding API Documentation

API documentation is auto-generated from docstrings using Sphinx's autodoc extension.

To document a new module, add it to the API documentation:

```rst
.. automodule:: your_module_name
   :members:
   :undoc-members:
   :show-inheritance:
```

## CI/CD Integration

Documentation is automatically built and deployed on pushes to main/master branches via GitHub Actions (see `.github/workflows/ci.yml`).
