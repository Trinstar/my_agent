from langchain_community.chat_models import ChatTongyi
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Optional


def select_model(model_name: str, is_local: bool = False, base_url: Optional[str] = None) -> BaseChatModel:
    """Select and return the langchain's BaseChatModel based on the model_name.

    Args:
        model_name: Name of the model to select
        is_local: Whether to use local model
        base_url: Base URL for the model API (optional)

    Returns:
        BaseChatModel: Selected chat model instance

    Raises:
        ValueError: If the model name is not supported or configuration is invalid
    """

    # Cloud models
    if not is_local:
        if model_name == "qwen3-max":
            return ChatTongyi(model="qwen3-max")
        elif model_name == "deepseek-chat":
            return ChatDeepSeek(model="deepseek-chat")
        else:
            raise ValueError(f"Unsupported cloud model name: {model_name}")

    # Local models
    else:
        if not base_url:
            # Local models without base URL
            if model_name == "qwen3:0.6B":
                return ChatOllama(model="qwen3:0.6B")
            else:
                raise ValueError(
                    f"Unsupported local model name without base_url: {model_name}")
        else:
            # Local models with base URL
            if model_name == "qwen3:0.6B":
                return ChatOllama(model="qwen3:0.6B", base_url=base_url)
            elif model_name == "vllm/qwen3:0.6B":
                return ChatOpenAI(
                    base_url=base_url,
                    api_key="EMPTY",
                    model="/root/autodl-tmp/Qwen3-0.6B",
                )
            else:
                raise ValueError(
                    f"Unsupported local model name with base_url: {model_name}")