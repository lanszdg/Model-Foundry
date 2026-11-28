# Bundled third-party components

Model Foundry is independent of the projects listed below. Their names identify the source of bundled components, not endorsement.

| Component | Pinned version | License and source |
| --- | --- | --- |
| Hugging Face Tokenizers.js | 0.2.0 | Apache-2.0; [license](hf-tokenizers-LICENSE), [source](https://github.com/huggingface/tokenizers.js) |
| Hugging Face Jinja | 0.5.13 | MIT; [license](hf-jinja-LICENSE), [source](https://github.com/huggingface/huggingface.js/tree/main/packages/jinja) |
| GraphQL.js | 17.0.2 | MIT; [license](graphql-LICENSE), [source](https://github.com/graphql/graphql-js) |
| MLflow frontend and GraphQL schema | 3.0.1 | Apache-2.0; [license](mlflow-LICENSE), [source](https://github.com/mlflow/mlflow/tree/v3.0.1/mlflow/server/js) |

The MLflow distribution assets are unchanged after decompression. Its distributed JavaScript license notices remain in the asset bundle. The transport adapter, local service implementation, iframe bootstrap and vocabulary projection are Model Foundry code. The Jinja interpreter has one documented extension for `unique(case_sensitive=..., attribute=...)`; upstream chat template text is not rewritten.

Tokenizer data and chat templates retain their model repositories, pinned revisions, original file SHA256 values and download locations in [the source inventory](../docs/原生Tokenizer来源.json). Each upstream repository retains its own model/data licensing terms. GR00T's gated text dependency was resolved only for three tokenizer/template files whose public Qwen Git blob IDs exactly match the dependency metadata; no gated weights are included. The inventory records both locations and the byte-identity evidence.

Canonical JSON serialization is used to pack tokenizer assets. Exported JSON preserves the upstream configuration and vocabulary; whitespace formatting differs from the original files. The HTML does not bundle pretrained model weights.
