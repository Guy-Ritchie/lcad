"""
Module that provides interfaces,
and classes to handle LLMs from different providers.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI as ChatGoogle
from langchain_google_genai import GoogleGenerativeAIEmbeddings as EmbedGoogle

load_dotenv()


class LLMProvider(BaseModel):
    """
    LLMProvider class, acts as the factory for fetching the ChatModel instance.
    We'll use this to instantiate, and access all of our LLMs in project!

    TODO: Later on, focus on things like:
    * fallback mechanisms
    * rate limit logging
    * backoff handling, etc.
    """

    provider: Literal["google", "groq"] = Field(default="groq")
    model_id: str = Field(default=os.environ["DEFAULT_GROQ_MODEL"])

    def __init__(
        self,
        provider: Literal["google", "groq"] = "groq",
        model_id: str = os.environ["DEFAULT_GROQ_MODEL"],
    ):
        super().__init__()
        self.provider = provider
        self.model_id = model_id

    def get_llm(self, temperature=0.2, max_tokens=2024):
        """
        Function that returns LLM instance based on provider!
        Currently accepts temp & max_tokens args.
        """
        if self.provider == "google":
            return ChatGoogle(
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
        if self.provider == "groq":
            return ChatGroq(
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens
            )

        raise ValueError(f"Unknown provider: {self.provider}")

    def get_embed_lm(self):
        """
        method that returns the embedding model.
        By default, we only have access to Google's Embedding model.
        """
        return EmbedGoogle(
            model=os.environ["EMBEDDING_MODEL"]
        )


__all__ = ["LLMProvider"]
