from vectorstore import vectorstore
from agent import QuerySummarizerAgent, RetrieverAgent
from util import LLMUtils


def create_rag_chain():
    llm = LLMUtils.get_llm("haiku")
    return QuerySummarizerAgent(llm=llm) | RetrieverAgent(
        vectorstore
    )  # TODO: Lanjutin bai
