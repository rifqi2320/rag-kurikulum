from .base import BaseAgent
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


class RouteAgent(BaseAgent):
    def __init__(self, route_dict: dict[str, RunnableSerializable], route_key: str):
        super().__init__()
        self.route_dict = route_dict
        self.route_key = route_key

    def route(self, _payload: dict):
        route_key = _payload[self.route_key]
        return self.route_dict[route_key]

    def _get_runnable(self):
        return RunnablePassthrough.assign(result=RunnableLambda(self.route))
