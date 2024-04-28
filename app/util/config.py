import os
from dotenv import load_dotenv
import yaml


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            raise ValueError(f"Key {key} not found in config")

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            if default is not None:
                return default
            raise ValueError(f"Key {key} not found in config")


class Config:
    CONFIG_PATH = os.path.join(os.curdir, "config")
    LLM_PATH = os.path.join(os.curdir, "config", "llm")
    PROMPT_PATH = os.path.join(os.curdir, "prompt")

    env: ConfigDict
    openai_embedding: ConfigDict

    llms: ConfigDict
    prompts: ConfigDict

    @classmethod
    def init(cls):
        load_dotenv()
        cls.env = ConfigDict(os.environ)
        cls.prompts = ConfigDict()
        cls.llms = ConfigDict()

        for file in os.listdir(cls.CONFIG_PATH):
            if file.endswith(".yaml"):
                config_name = file.split(".")[0]
                setattr(
                    cls,
                    config_name,
                    ConfigDict(cls.__load_yaml(os.path.join(cls.CONFIG_PATH, file))),
                )

        for file in os.listdir(cls.PROMPT_PATH):
            if file.endswith(".yaml"):
                prompt_name = file.split(".")[0]
                cls.prompts[prompt_name] = cls.__load_yaml(
                    os.path.join(cls.PROMPT_PATH, file)
                )

        for file in os.listdir(cls.LLM_PATH):
            if file.endswith(".yaml"):
                llm_name = file.split(".")[0]
                cls.llms[llm_name] = cls.__load_yaml(os.path.join(cls.LLM_PATH, file))

    @staticmethod
    def __load_yaml(path: str):
        with open(path) as f:
            return yaml.safe_load(f)
