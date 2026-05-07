import os
import requests
from src.ingestion.ingest import load_bm25
from src.common import config, utils
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
from langchain_core.prompts import ChatPromptTemplate

# --- Global Client Placeholders ---
PC_CLIENT = None
INDEX = None
VO_CLIENT = None
BM25_MODEL = None
RETRIEVER_AGENT = None

def get_retrieval_clients():
    global PC_CLIENT, INDEX, VO_CLIENT, BM25_MODEL
    if INDEX is None:
        INDEX = utils.get_pinecone_index()
    if VO_CLIENT is None:
        VO_CLIENT = utils.get_voyage_client()
    if BM25_MODEL is None:
        BM25_MODEL = load_bm25()
    return INDEX, VO_CLIENT, BM25_MODEL

def get_retriever_agent():
    global RETRIEVER_AGENT
    if RETRIEVER_AGENT is None:
        llm = utils.get_gemini_llm(config.LLM_MODEL)
        structured_retriever_response = llm.with_structured_output(RetrievalParameters)
        RETRIEVER_AGENT = prompt | structured_retriever_response
    return RETRIEVER_AGENT

def hybrid_score_norm(dense, sparse, alpha: float):
    # ... (existing code remains same)
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs

def rerank(query, matches):
    if not matches:
        return []
        
    docs = [match["metadata"]["text"] for match in matches]
    url = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B"
    token = os.getenv("DEEPINFRA_TOKEN")
    
    if not token:
        print("⚠️ Warning: DEEPINFRA_TOKEN not found. Skipping rerank.")
        return matches[0:5]

    headers = {
        "Authorization": f"bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "queries": [query],
        "documents": docs
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if "scores" not in result:
            print(f"⚠️ Error: 'scores' key missing in DeepInfra response: {result}")
            return matches[0:5]
            
        scores = result["scores"]

        for m, s in zip(matches, scores):
            m["score"] = s

        sorted_docs = sorted(matches, key=lambda x: x["score"], reverse=True)
        return sorted_docs[0:5]

    except Exception as e:
        print(f"⚠️ Rerank failed: {e}. Falling back to original order.")
        return matches[0:5]

def search_vdb(query: str, alpha: float, metadata_filters: dict):
    # Ensure clients are initialized
    idx, vo_client, bm25_model = get_retrieval_clients()

    # 1. Get the single dense embedding
    dense_embeddings = vo_client.embed(
        [query],
        model="voyage-finance-2",
        input_type="query"
    ).embeddings
    
    dense_vector_query = dense_embeddings[0]

    # Normalize the 1D dense vector
    norm = np.linalg.norm(dense_vector_query)
    if norm > 0:
        dense_vector_query = (dense_vector_query / norm).tolist()

    # 2. Get the single sparse vector (returns a dict)
    sparse_vector_query = bm25_model.encode_queries(query)

    # 3. Apply hybrid weighting
    if alpha is not None:
        dense_vector_query, sparse_vector_query = hybrid_score_norm(
            dense_vector_query, 
            sparse_vector_query, 
            alpha
        )

    # 4. Query Pinecone directly without a loop
    response = idx.query(
        namespace="ns1",
        top_k=50,
        vector=dense_vector_query,
        sparse_vector=sparse_vector_query,
        filter=metadata_filters,
        include_values=False,
        include_metadata=True
    )

    top_5_chunks = rerank(query, response["matches"])
    
    return top_5_chunks

class MetadataFilters(BaseModel):
    """Explicit metadata filters extracted from the user query."""
    
    company_name: Optional[str] = Field(
        default=None, 
        description="The full name of the company if mentioned (e.g., 'TD Bank Group')."
    )
    ticker: Optional[str] = Field(
        default=None, 
        description="The stock ticker if mentioned (e.g., 'TD', 'AAPL')."
    )
    calendar_year: Optional[str] = Field(
        default=None, 
        description="The 4-digit year (e.g., '2025')."
    )
    fiscal_quarter: Optional[str] = Field(
        default=None, 
        description="The specific quarter (e.g., 'Q1', 'Q2', 'Q3', 'Q4')."
    )
    document_type: Optional[str] = Field(
        default=None, 
        description="The type of financial filing (e.g., '10-Q', '10-K', 'Earnings TransScript')."
    )
    '''
    financial_tags: Optional[List[str]] = Field(
        default=None, 
        description="Any specific topical tags mentioned, such as ['Reporting', 'Financials', 'Risk']."
    )'''
    is_audited: Optional[bool] = Field(
        default = None,
        description = "Whether the financial statement has been audited by an external firm. true for annual reports (10-K), false for quarterly reports (10-Q)"
    )
    is_table: Optional[bool] = Field(
        default = None,
        description = "true if the user wants hard data/numbers, false for narrative text"
    )

class RetrievalParameters(BaseModel):
    """Instructions extracted from the user query to execute a hybrid search."""
    
    optimized_query: str = Field(
        description="The core search query optimized for vector matching."
    )
    alpha: float = Field(
        description="Weight for dense vs. sparse search (0.0 to 1.0).",
        ge=0.0, 
        le=1.0
    )
    metadata_filters: MetadataFilters = Field(
        description="Specific constraints to filter the vector database."
    )

"""Search the financial document store for relevant information."""

RETRIEVER_PROMPT = """You are an expert Search & Retrieval Analyst for TD Bank Group (TD). 
Your job is to analyze a user's query and extract parameters for our vector database of TD financial reports.

METADATA DEFAULTS:
- company_name: "TD Bank Group"
- ticker: "TD"

EXTRACTION RULES:
1. optimized_query: Focus on financial metrics, segment names (e.g., 'U.S. Retail', 'Wholesale Banking'), or specific events mentioned.
2. alpha: 
    - 0.0-0.3 for hard numbers, specific line items, or regulatory codes (e.g., "CET1 ratio", "PCL").
    - 0.7-1.0 for strategy, qualitative commentary, or summaries.
    - 0.5 for general performance queries.
3. metadata_filters: Extract year, quarter, and document type (10-Q for quarterly, 10-K for annual). 
    - If the user says "this year", use 2025. 
    - If they say "last year", use 2024.
    - If they ask about "tables" or "data", set is_table to true.

Do not answer the question. Only provide the JSON search parameters."""


prompt = ChatPromptTemplate.from_messages([
    ("system", RETRIEVER_PROMPT),
    ("human", "{user_input}")
])

@tool
def retrieve_info(user_query):
    """Search the financial document store for relevant information."""
    agent = get_retriever_agent()
    search_params = agent.invoke({"user_input":user_query})
    filters_dict = search_params.metadata_filters.model_dump(exclude_none=True)

    print(f"Refined user query : {search_params.optimized_query}")  # "revenue drop project Apollo"
    print(f"Vibes weight : {search_params.alpha}")            # ~0.4 (mix of specific project and semantic concept)
    print(f"Filters : {filters_dict}") # {'quarter': 'Q3', 'year': 2023, 'report_type': 'financial'}

    retrieved_context = search_vdb(
        search_params.optimized_query,
        search_params.alpha,
        filters_dict
        )
    return retrieved_context




