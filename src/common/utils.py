import os
from src.common import config

def get_voyage_client():
    import voyageai
    return voyageai.Client(api_key=config.VOYAGE_API_KEY)

def get_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if not pc.has_index(config.INDEX_NAME):
        pc.create_index(
            name=config.INDEX_NAME,
            vector_type="dense",
            dimension=1024,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(config.INDEX_NAME)

def get_groq_client():
    from groq import Groq
    return Groq(api_key=config.GROQ_API_KEY)

def get_gemini_llm(model_name=config.LLM_MODEL):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_name)
