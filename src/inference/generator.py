from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from src.inference.retriever import retrieve_info
from src.common import config, utils
import os

# 2. The Financial System Prompt - Specialized for TD Bank Group
system_prompt = """You are a Senior Equity Research Analyst specializing in TD Bank Group (TD). 
Your goal is to provide precise, data-driven answers using ONLY the provided context from TD's official financial reports (10-Qs, 10-Ks, and Earnings Reports).

TD BUSINESS SEGMENTS:
- Canadian Personal and Commercial Banking
- U.S. Retail (including TD Auto Finance and investment in Charles Schwab)
- Wealth Management and Insurance
- Wholesale Banking
- Corporate segment

CRITICAL RULES:
1. Strict Grounding: If the answer is not in the context, state: "I cannot find this information in the provided TD financial reports." Do not use outside knowledge.
2. Tables as Truth: Treat Markdown tables as the absolute source of truth for financial figures.
3. Math & Analysis: For growth rates or variances, always show your calculation: (New - Old) / Old.
4. Mandatory Citations: Cite the specific header and page number for every fact.
   Format: "[Data Point] (Source: [Header_2], Page [Page Number])"
5. Currency: All figures are in Canadian Dollars (CAD) unless otherwise specified (e.g., U.S. Retail results are often discussed in USD and CAD).

=== CONTEXT ===
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_input}")
])

# 3. The Metadata Injector
# Adjusted to handle the dictionary matches returned by search_vdb/retrieve_info
def format_docs(matches):
    formatted_chunks = []
    for i, m in enumerate(matches):
        metadata = m.get('metadata', {})
        page = metadata.get('page(s)', 'N/A')
        report = metadata.get('ticker', 'N/A') + " " +  metadata.get('calendar_year', 'N/A') + " " +  metadata.get('fiscal_quarter', 'N/A')
        header = metadata.get('Header_2', 'Unknown Section')
        content = m.get('metadata', {}).get('text', 'No content available')
        
        chunk_str = f"--- CHUNK {i+1} ---\nSOURCE: {report} : {header}, Page {page}\nTEXT/DATA:\n{content}\n"
        formatted_chunks.append(chunk_str)
        
    return "\n".join(formatted_chunks)

# 4. Build the LCEL Chain
def get_rag_chain():
    # 1. Initialize your final LLM
    llm = utils.get_gemini_llm()

    # We use a RunnableLambda to call our retrieve_info tool
    def run_retrieval(query):
        # Since retrieve_info is a LangChain tool, we must use .invoke()
        return retrieve_info.invoke(query)

    # Refactored chain to return both context and answer
    chain = (
        RunnableParallel({
            "context": RunnableLambda(run_retrieval) | format_docs, 
            "user_input": RunnablePassthrough()
        })
        | {
            "answer": prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"]
        }
    )
    return chain

# For backward compatibility and CLI use
rag_chain = None

# --- EXECUTION ---
if __name__ == "__main__":
    rag_chain = get_rag_chain()
    print("=" * 50)
    print("TD BANK EQUITY RESEARCH ANALYST AI")
    print("=" * 50)
    print("Ask questions about TD's financial performance, segments, or risk factors.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_question = input("QUESTION: ")

        if user_question.lower() in ["exit", "quit"]:
            print("Closing Analyst session. Goodbye.")
            break

        if not user_question.strip():
            continue

        print(f"\nAnalyzing reports for: '{user_question}'...")

        try:
            # Run the chain!
            result = rag_chain.invoke(user_question)

            print("-" * 30)
            print("ANALYSIS RESPONSE:")
            print("-" * 30)
            print(result["answer"])
            print("-" * 30)
            print("\n" + "=" * 50 + "\n")

        except Exception as e:
            print(f"\nAn error occurred during analysis: {e}")
            print("Please try rephrasing your question.\n")