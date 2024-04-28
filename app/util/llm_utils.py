from .config import Config
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)


class LLMUtils:
    @staticmethod
    def get_prompt(prompt_name):
        prompt_configs = Config.prompts[prompt_name]

        few_shots = FewShotChatMessagePromptTemplate(
            examples=prompt_configs["examples"] if "examples" in prompt_configs else [],
            example_prompt=example_prompt,
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", prompt_configs["system_prompt"]),
                few_shots,
                ("human", prompt_configs["instruction_prompt"]),
            ],
        )

    @staticmethod
    def get_llm(llm_name):
        llm_configs = Config.llms[llm_name]
        if llm_configs["provider"] == "anthropic":
            temp = {**llm_configs}
            del temp["provider"]
            return ChatAnthropic(**temp)
        else:
            raise NotImplementedError("LLM provider not supported yet")
