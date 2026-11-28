from pathlib import Path
import json
from mlflow.server.graphql.graphql_schema_extensions import schema
Path('.test-artifacts/mlflow-3.0.1-schema.graphql').write_text(str(schema),'utf8')
Path('src/vendor/mlflow-schema.js').write_text('/* Original MLflow 3.0.1 Graphene schema. Apache-2.0. */\nglobalThis.FoundryMLflowSchema='+json.dumps(str(schema))+';\n','utf8')
print('MLflow GraphQL schema exported',len(str(schema)))
