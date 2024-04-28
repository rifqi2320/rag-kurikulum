from langchain_core.runnables.base import RunnableSerializable
from .base import BaseAgent
from langchain.vectorstores.base import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


class RetrieverAgent(BaseAgent):
    def __init__(
        self,
        vectorstore: VectorStore,
        query_key: str = "summarized_query",
    ):
        super().__init__()
        self.vectorstore = vectorstore
        self.query_key = query_key

    def retrieve(self, _payload: dict):
        query = _payload[self.query_key]
        return self.vectorstore.similarity_search(query, k=20)

    def _get_runnable(self) -> RunnableSerializable:
        return RunnablePassthrough.assign(result=RunnableLambda(self.retrieve))
