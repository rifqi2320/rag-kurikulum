from typing import Any
from langchain.llms.base import LLM
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.tools.base import Tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


# Using Runnable as a base class so we can use it in Langchain library
class BaseAgent(Runnable):
    def _get_runnable(self) -> RunnableSerializable:
        raise NotImplementedError()

    def __call__(self, input: dict, config: RunnableConfig | None = None):
        return self._get_runnable().invoke(input, config)

    # Methods needed to be easily used in the pipeline (LCEL, Langserve, LangSmith, etc.)
    def __or__(self, other: Runnable):
        return self._get_runnable() | other

    def __ror__(self, other: Runnable):
        return other | self._get_runnable()

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        return self._get_runnable().invoke(input, config)

    def stream(self, input: Any, config: RunnableConfig | None = None) -> Any:
        return self._get_runnable().stream(input, config)

    async def astream_log(
        self, input: Any, config: RunnableConfig | None = None
    ) -> Any:
        return self._get_runnable().astream_log(input, config)


class BaseLLMAgent(BaseAgent):
    prompt: PromptTemplate | ChatPromptTemplate
    llm: LLM | ChatOpenAI

    def __init__(
        self,
        llm: LLM | ChatOpenAI,
        prompt: PromptTemplate | ChatPromptTemplate,
    ):
        self.llm = llm
        self.prompt = prompt
