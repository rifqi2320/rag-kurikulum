from langchain_core.runnables.base import RunnableSerializable
from .base import BaseLLMAgent
from langchain.vectorstores.base import VectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.llms.base import LLM
from util import LLMUtils
from langchain_core.output_parsers import StrOutputParser


class ResponderAgent(BaseLLMAgent):
    def __init__(self, llm: LLM | ChatOpenAI, query_key = "documents", output_key: str = "response"):
        super().__init__(llm, prompt="")
        self.query_key = query_key
        self.output_key = output_key

    def generate_prompt(self, _payload: dict):
        query = _payload["summarized_query"]
        relevant_docs = _payload[self.query_key][0].page_content
        final_query = query + '\n\n' + 'Berikut merupakan dokumen yang relevan untuk menjawab pertanyaan di atas:\n' + relevant_docs
        return final_query

    def _get_runnable(self) -> RunnableSerializable:
        return RunnablePassthrough.assign(**{self.output_key: RunnableLambda(self.generate_prompt) | self.llm | StrOutputParser()})