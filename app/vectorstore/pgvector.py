import os
from embedding import embeddings
from langchain_community.vectorstores import PGVector
from util import Config
import warnings

connection = Config.env["POSTGRES_CONNECTION"]
collection_name = "kurikulum"

with warnings.catch_warnings(action="ignore"):
    vectorstore = PGVector(
        connection_string=connection,
        collection_name=collection_name,
        embedding_function=embeddings,
        use_jsonb=True,
    )
