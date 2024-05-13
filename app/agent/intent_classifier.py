from .base import BaseLLMAgent
from util import LLMUtils, RunnableUtils
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import EnumOutputParser
from enum import Enum


class Intent(Enum):
    conversation = "conversation"
    search = "search"


class IntentClassifierAgent(BaseLLMAgent):
    def __init__(self, llm, output_key="intent"):
        super().__init__(llm, prompt=LLMUtils.get_prompt("intent_classifier"))
        self.output_key = output_key

    def _get_runnable(self):
        output_parser = EnumOutputParser(enum=Intent)

        return RunnablePassthrough.assign(
            **{
                self.output_key: self.prompt.partial(
                    format_instructions=output_parser.get_format_instructions()
                )
                | self.llm
                | output_parser
                | RunnableUtils.enum_to_str()
            }
        )
