from langchain_core.runnables.base import RunnableSerializable
from .base import BaseLLMAgent
from langchain.vectorstores.base import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.llms.base import LLM
from util import LLMUtils
from langchain_core.output_parsers import StrOutputParser


class ResponderAgent(BaseLLMAgent):
    def __init__(
        self, llm: LLM | ChatOpenAI, query_key="documents", output_key: str = "response"
    ):
        super().__init__(llm, prompt="")
        self.query_key = query_key
        self.output_key = output_key

    def generate_prompt(self, _payload: dict):
        query = _payload["summarized_query"]
        docs_content = _payload[self.query_key].page_content
        docs_book = _payload[self.query_key].metadata["id"].split("_")[0]
        docs_page = _payload[self.query_key].metadata["id"].split("_")[1]
        relevant_docs = f"Sertakan buku dan halaman yang relevan pada jawaban. Dokumen yang relevan untuk menjawab pertanyaan di atas adalah buku {docs_book} halaman {docs_page}:\n{docs_content}"
        final_query = query + "\n\n" + relevant_docs
        return final_query

    def _get_runnable(self) -> RunnableSerializable:
        return RunnableLambda(self.generate_prompt) | self.llm | StrOutputParser()
