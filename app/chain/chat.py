from agent import (
    IntentClassifierAgent,
    RouteAgent,
    QuerySummarizerAgent,
    RetrieverAgent,
    ResponderAgent,
)
from util import LLMUtils
from vectorstore import vectorstore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import MessageLikeRepresentation, AIMessage
from langchain_core.output_parsers import StrOutputParser

store: dict[str, ChatMessageHistory] = {}


class ChatInput(BaseModel):
    query: str
    chat_history: list[MessageLikeRepresentation]


ChatOutput = str


def turn_history_to_string(_payload: dict):
    chat_history = ""
    for message in _payload["chat_history"]:
        chat_history += f"{message.type}: {message.content}\n"

    return {**_payload, "chat_history": chat_history}


def create_chat_chain():
    llm = LLMUtils.get_llm("haiku")
    conversation_prompt = LLMUtils.get_prompt("conversation")

    rag_chain = (
        QuerySummarizerAgent(llm=llm)
        | RetrieverAgent(vectorstore)
        | ResponderAgent(llm=llm)
    )
    route_dict = {
        "search": rag_chain,
        "conversation": conversation_prompt | llm | StrOutputParser(),
    }
    main_chain = IntentClassifierAgent(llm=llm) | RouteAgent(
        route_dict=route_dict, route_key="intent"
    )

    return (
        RunnableLambda(turn_history_to_string)
        | main_chain
        | RunnableLambda(lambda x: x["result"])
    ).with_types(
        input_type=ChatInput,
        output_type=ChatOutput,
    )
