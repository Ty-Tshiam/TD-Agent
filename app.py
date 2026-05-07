import streamlit as st
from src.inference.generator import get_rag_chain
from src.common import config
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TD Analyst AI",
    page_icon="🏦",
    layout="wide"
)

# --- CACHED COMPONENTS ---
@st.cache_resource
def load_analyst_chain():
    return get_rag_chain()

rag_chain = load_analyst_chain()

# --- TD BRANDING (CSS) ---
td_green = "#008A00"
td_light_green = "#00B32C"
td_dark_green = "#005D00"

st.markdown(f"""
    <style>
    /* Main App Background */
    .stApp {{
        background-color: #f8f9fa;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {td_dark_green} !important;
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        color: white !important;
    }}
    
    /* Header Styling */
    h1, h2, h3 {{
        color: {td_green};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Chat Input Styling */
    .stChatInputContainer {{
        border-top: 2px solid {td_green};
    }}
    
    /* Button Styling */
    div.stButton > button {{
        background-color: {td_green};
        color: white;
        border-radius: 5px;
        border: none;
    }}
    div.stButton > button:hover {{
        background-color: {td_light_green};
        color: white;
    }}
    
    /* Chat Bubbles (Customization via classes if needed, 
       but Streamlit's native ones are clean) */
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Document Explorer) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a4/Toronto-Dominion_Bank_logo.svg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original", width=120)
    st.title("TD Analyst AI")
    st.subheader("Document Repository")
    
    st.success("✅ Hybrid Index: **Active**")
    
    with st.expander("📊 Available Reports", expanded=True):
        st.markdown("**TD Bank Group (TD)**")
        reports = list(config.FINAL_DATA_DIR.glob("*.json"))
        if reports:
            for report in sorted(reports, reverse=True):
                # Format "TD_2025_Q4.json" to "2025 Q4"
                name_parts = report.stem.split("_")
                if len(name_parts) >= 3:
                    st.markdown(f"- {name_parts[1]} {name_parts[2]}")
                else:
                    st.markdown(f"- {report.stem}")
        else:
            st.warning("No reports found in data/final/")
    
    st.divider()
    st.markdown("### Search Settings")
    st.info("The analyst uses a **Hybrid Retrieval** system (Dense + Sparse) with **Qwen-8B Reranking** for maximum accuracy.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN INTERFACE ---
st.title("🏦 TD Bank Equity Research Analyst")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("View Source Citations"):
                st.text(message["context"])

# React to user input
if prompt := st.chat_input("Ask about TD's Q1 performance, risk factors, or segment growth..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing TD Financial Reports..."):
            try:
                # Invoke the RAG chain
                result = rag_chain.invoke(prompt)
                answer = result["answer"]
                context = result["context"]

                # Display the answer
                answer = answer.replace("$", r"\$")
                st.markdown(answer)
                
                # Display citations in an expander
                with st.expander("View Source Citations"):
                    st.text(context)

                # Add assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": context
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Sorry, I encountered an error during analysis: {e}"
                })
