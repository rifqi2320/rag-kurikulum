from vectorstore import vectorstore
from agent import QuerySummarizerAgent, RetrieverAgent, ResponderAgent
from util import LLMUtils


def create_rag_chain():
    llm = LLMUtils.get_llm("haiku")
    response = QuerySummarizerAgent(llm=llm) | RetrieverAgent(vectorstore) | ResponderAgent(llm=llm)
    print(response)
    return response
