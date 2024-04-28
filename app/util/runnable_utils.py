from langchain_core.runnables import RunnableLambda
from enum import Enum


class RunnableUtils:
    @staticmethod
    def enum_to_str() -> RunnableLambda:
        def fn(enum: Enum):
            return enum.name

        return RunnableLambda(fn)
