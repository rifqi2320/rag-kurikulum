from langchain_core.runnables.base import RunnableSerializable
from .base import BaseAgent
from langchain.vectorstores.base import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import cohere
from util import Config

class RetrieverAgent(BaseAgent):
    def __init__(
        self,
        vectorstore: VectorStore,
        query_key: str = "summarized_query",
        output_key: str = "documents",
    ):
        super().__init__()
        self.vectorstore = vectorstore
        self.query_key = query_key
        self.output_key = output_key

    def retrieve(self, _payload: dict):
        query = _payload[self.query_key]
        documents = self.vectorstore.similarity_search(query, k=20)
        content = [doc.page_content for doc in documents]

        # rerank the top 20 documents, and return the top one
        co = cohere.Client(Config.env["COHERE_API_KEY"])
        results = co.rerank(query=query, documents=content, top_n=1, model=Config.cohere["model"])
        return documents[results.results[0].index]

    def _get_runnable(self) -> RunnableSerializable:
        return RunnablePassthrough.assign(**{self.output_key: RunnableLambda(self.retrieve)})
