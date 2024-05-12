from agent import IntentClassifierAgent, RouteAgent, QuerySummarizerAgent, RetrieverAgent, ResponderAgent
from util import LLMUtils
from vectorstore import vectorstore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda

store: dict[str, ChatMessageHistory] = {}


class ChatInput(BaseModel):
    query: str


class ChatOutput(BaseModel):
    output: str


def get_session_history(session_id: str) -> str:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def turn_history_to_string(_payload: dict):
    chat_history = ""
    for message in _payload["chat_history"]:
        chat_history += f"{message.type}: {message.content}\n"
    return {**_payload, "chat_history": chat_history}

def extract_response(payload):
    response = payload.get('result', {}).get('response', "Maaf, saya tidak bisa menemukan jawaban untuk pertanyaan ini.")
    # return {'output': response}
    return response

def create_chat_chain():
    llm = LLMUtils.get_llm("haiku")

    rag_chain = QuerySummarizerAgent(llm=llm) | RetrieverAgent(vectorstore) | ResponderAgent(llm=llm)
    route_dict = {
        "search": rag_chain,
        "conversation": llm,
    }
    main_chain = IntentClassifierAgent(llm=llm) | RouteAgent(route_dict=route_dict, route_key="intent")


    return RunnableWithMessageHistory(
        RunnableLambda(turn_history_to_string) | main_chain | RunnableLambda(extract_response),
        get_session_history=get_session_history,
        input_messages_key="query",
        output_messages_key="output",
        history_messages_key="chat_history",
    ).with_types(
        input_type=ChatInput,
        output_type=ChatOutput,
    )
