from fastapi import FastAPI
from langserve import add_routes
from chain import create_chat_chain

app = FastAPI(
    title="RAG Kurikulum Merdeka",
    version="0.1.0",
    description="Chatbot utilizing RAG model for Kurikulum Merdeka",
)

chat_chain = create_chat_chain()


add_routes(app, chat_chain, path="/chat", playground_type="chat")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=1337)
