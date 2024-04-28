from .base import BaseLLMAgent
from langchain_openai import ChatOpenAI
from langchain.llms.base import LLM
from util import LLMUtils
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser


class QuerySummarizerAgent(BaseLLMAgent):
    def __init__(self, llm: LLM | ChatOpenAI, output_key: str = "summarized_query"):
        super().__init__(llm, prompt=LLMUtils.get_prompt("query_summarizer"))
        self.output_key = output_key

    def _get_runnable(self) -> RunnableSerializable:
        return RunnablePassthrough.assign(
            **{self.output_key: self.prompt | self.llm | StrOutputParser()}
        )
