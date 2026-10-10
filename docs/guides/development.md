# Development

## Build the documentation

```powershell
pip install -r docs/requirements.txt
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The command treats warnings
as errors and checks cross-references (`-n`).

## Docstring style

Use Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Example:`).
They are rendered by `sphinx.ext.napoleon`. API pages are generated from the
source by `autosummary`, so new modules need only be added to
`docs/api/index.rst`.

## Known documentation gaps

* `main.main` has no `Returns:` section (it returns `None`).
* `_step` and `_execute_action` are private and are not in the API reference.
  Their roles are described in the architecture overview.
