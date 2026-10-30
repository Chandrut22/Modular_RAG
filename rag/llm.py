from langchain.chat_models import init_chat_model 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


def make_llm(temperature: float):
    return init_chat_model(
        model="gpt-oss:20b",
        model_provider="openai",
        api_key=os.getenv("API_KEY"),
        base_url="https://ollama.com/v1",
        temperature=temperature,
        max_retries=2,
    )

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)