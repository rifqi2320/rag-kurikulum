from langchain_openai import OpenAIEmbeddings
import tiktoken
from util import Config
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

model_name = Config.openai_embedding["model"]
underlying_embeddings = OpenAIEmbeddings(model=model_name)
store = LocalFileStore("cache/")

embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)


if model_name not in [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]:
    raise ValueError(f"Model {model_name} not supported")

encoding = tiktoken.get_encoding("cl100k_base")
encode_function = encoding.encode
encode_batch_function = encoding.encode_batch
