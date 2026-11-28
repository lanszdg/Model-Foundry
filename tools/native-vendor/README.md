# Rebuilding bundled native components

These are optional maintainer tools. Opening the delivered HTML and running the JavaScript regression suite need no package installation.

From this directory, `npm ci && node build.mjs` rebuilds `src/vendor/native-hf.js` and `src/vendor/native-graphql.js` using the committed version lock. `entry.mjs` contains the Jinja `unique` extension. The runtime adapter supplies a Jinja-compatible `dict` function; official template strings remain intact.

The Python scripts run from the repository root. They use `.test-artifacts/v16-evidence` for downloaded source assets. `fetch-tokenizers.py` reads the committed inventory, downloads the exact pinned files and rejects SHA256/Git blob changes; it uses the recorded public equivalent for GR00T. Then `package-tokenizers.py` writes the inline tokenizer bundle and source inventory. `resolve-groot-tokenizer.py` preserves the original independent Git blob verification procedure. `reference-tokenizers.py` generates Rust-tokenizers/Python-Jinja golden results; copy the reviewed output into `test-fixtures/native-tokenizers-v16.json`.

Reference generation uses Python `tokenizers==0.23.3`, `transformers==5.24.1` and `jinja2==3.1.6`, including Transformers' `_compile_jinja_template`. Package these tools in an isolated environment. Normal `npm test` consumes the committed golden data and does not install Python packages or access the network.

`package-mlflow-ui.py` reads the installed MLflow 3.0.1 distribution in `.test-artifacts/v12-external/.venv/Lib/site-packages/mlflow/server/js/build`. It omits source maps, preserves all other bytes, and stores each file's SHA256 and size. `export-mlflow-schema.py`, run with that environment's Python, writes the original Graphene SDL. The runtime copy is `src/vendor/mlflow-schema.js`.

See [third-party notices](../../third-party/NOTICE.md) and [source inventory](../../docs/原生Tokenizer来源.json).
