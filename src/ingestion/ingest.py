import requests
from azure.core.credentials import AzureKeyCredential
import os
import json
import re
import time
from src.ingestion import process
from datetime import datetime
from pathlib import Path
from src.common import config, utils
import pickle

#Configuration & Global Variables
DOWNLOAD_DIR = config.RAW_DATA_DIR
AZURE_MARKDOWNS = config.PROCESSED_DATA_DIR
OUTPUT_DIR = config.FINAL_DATA_DIR

def get_groq_client():
    return utils.get_groq_client()

def get_pinecone_index(index_name=config.INDEX_NAME):
    return utils.get_pinecone_index()

def get_voyage_client():
    return utils.get_voyage_client()

def get_jsons():
    directory = config.FINAL_DATA_DIR
    files = list(directory.glob("*.json"))
    return [file.name for file in files]

def train_bm25():
    from pinecone_text.sparse import BM25Encoder
    files = get_jsons()
    
    all_chunk_contents = []

    # 1. Collect everything first
    for file in files:
        with open(config.FINAL_DATA_DIR / file, "r", encoding="utf-8") as f:
            chunks = json.loads(f.read())
            
            for cid, data in chunks.items():
                all_chunk_contents.append(data["content"])

    # 2. Train ONE model on everything
    bm25 = BM25Encoder()
    bm25.fit(all_chunk_contents)
    
    # 3. Save for future use
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 model trained and saved to {config.BM25_PATH}")
    
    return bm25

def load_bm25():
    # We need the class in the namespace for pickle to work correctly if it was pickled as a BM25Encoder object
    from pinecone_text.sparse import BM25Encoder
    with open(config.BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return bm25

# Only train if we run this script directly, otherwise let the caller decide
if __name__ == "__main__":
    BM25 = train_bm25()
else:
    # We'll let retrieval.py call load_bm25()
    pass

def find_reports(START_YEAR : int, END_YEAR : int) -> dict:
    # --- Configuration ---
    COMPANY_TICKER = "TD"
    # The base URL is now just up to the common PDF folder.
    BASE_URL = "https://www.td.com/content/dam/tdcom/canada/about-td/pdf/"

    # Define the range of years and quarters to target
    QUARTERS = ["q1", "q2", "q3"] 
    Q4_QUARTER = "q4"

    # --- Function to Find a Valid Report URL ---

    def find_report_url(full_url):
        """
        Checks if a URL is valid and accessible.
        
        Returns: The URL if it's valid (HTTP 200), or None on failure (404, etc.).
        """
        # Add a User-Agent header so the site does not block the request
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            # Use HEAD request for a quick check before downloading content
            head_response = requests.head(full_url, headers=headers, allow_redirects=True, timeout=5)
            if head_response.status_code != 200:
                return None # File doesn't exist at this URL
            return full_url

        except requests.exceptions.RequestException:
            # Gracefully handle download errors (e.g., connection timeout, 403 Forbidden)
            return None

    # ----------------------------------------------------
    # 2. Generate URLs and Find Reports
    # ----------------------------------------------------

    print(f"--- Starting Report Search for {COMPANY_TICKER} ({START_YEAR}-{END_YEAR}) ---")
    found_reports = {}
    found_quarters = set()
    attempted_urls = set() # To track and skip redundant checks

    for year in range(START_YEAR, END_YEAR + 1):
        for q_tag in QUARTERS + [Q4_QUARTER]:
            
            quarter_name = q_tag.upper() # Q1, Q2, Q3, Q4

            # Standardized final filename tag for the RAG pipeline (e.g., TD_2025_Q3.pdf)
            new_filename_tag = f"{COMPANY_TICKER}_{year}_{quarter_name}.pdf"
            
            # List of potential file names to check for this specific year/quarter
            possible_urls = []
            
            # --- Pattern 1: Nested (Most common for recent quarters) ---
            # Example: .../quarterly-results/2025/q3/q3-2025-report-to-shareholders-en.pdf
            if q_tag != Q4_QUARTER:
                # Q1, Q2, Q3 have messy names in the nested folder
                nested_path = f"quarterly-results/{year}/{q_tag}/"
                possible_urls.extend([
                    # Primary QX-YYYY name (like Q3 2025 example)
                    BASE_URL + nested_path + f"{q_tag}-{year}-report-to-shareholders-en.pdf",
                    # Secondary YYYY-QX name (like Q1 2025 example)
                    BASE_URL + nested_path + f"{year}-{q_tag}-reports-shareholders-en.pdf"
                ])
            
            else: # Q4 is the Annual Report (and uses a slightly simpler nested path)
                nested_path_q4 = f"quarterly-results/{year}/{q_tag}/"
                # The Q4 Report to Shareholders is the Annual Report PDF
                possible_urls.extend([
                    # Standard nested path
                    BASE_URL + nested_path_q4 + f"{year}-annual-report-en.pdf",
                    # Variation with just year folder and "-e.pdf" suffix
                    BASE_URL + f"quarterly-results/{year}/" + f"{year}-annual-report-e.pdf",
                    # 2022 Variation: Different path and name
                    f"https://www.td.com/content/dam/tdcom/canada/about-td/for-investors/investor-relations/ar{year}-Complete-Report.pdf"
                ])

            # --- Patterns for Q1, Q2, Q3 ---
            if q_tag != Q4_QUARTER:
                quarter_number = q_tag[1] # "1", "2", or "3"
                possible_urls.extend([
                    # Pattern: Flat structure (e.g., .../pdf/2023-q1-report-to-shareholders-en.pdf)
                    BASE_URL + f"{year}-{q_tag}-report-to-shareholders-en.pdf",
                    # Pattern: Semi-Nested (e.g., .../quarterly-results/2023/2023-q2-report-to-shareholders-en.pdf)
                    BASE_URL + f"quarterly-results/{year}/" + f"{year}-{q_tag}-report-to-shareholders-en.pdf",
                    # 2022 Q1 Variation: Uppercase Q, underscores, _F_EN suffix
                    BASE_URL + f"quarterly-results/{year}/" + f"{year}-Q{quarter_number}_Report_to_Shareholders_F_EN.pdf",
                    # 2022 Q3 Variation: lowercase q, dashes, -f-en suffix
                    BASE_URL + f"quarterly-results/{year}/" + f"{year}-{q_tag}-report-to-shareholders-f-en.pdf"
                ])

            
            # --- EXECUTION: Find the first valid URL for the quarter ---
            is_found = False
            for url in possible_urls:
                if url in attempted_urls:
                    continue
                attempted_urls.add(url)
                
                found_url = find_report_url(url)
                
                if found_url:
                    print(f"✅ Found: {new_filename_tag} (at {found_url.split(BASE_URL)[-1]})")
                    found_reports[new_filename_tag] = found_url
                    found_quarters.add((year, quarter_name))
                    is_found = True
                    break # Move to the next quarter once a successful download is made
            
            if not is_found:
                print(f"--- FAILED to find: {new_filename_tag} ---")

    print("\n----------------------------------------------------------------------")
    print(f"**Search Complete.** Found {len(found_reports)} reports.")
    print("Found Report URLs:")
    for url in found_reports:
        print(f"- {url}")

    return found_reports

def download_reports(reports : dict):

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for filename, url in reports.items():
        local_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(local_path):
            print(f"⏭️  Skipping existing file: {filename}")
            continue
        
        try:
            print(f"⬇️  Downloading: {filename} from {url}")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded: {filename}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {filename} from {url}. Error: {e}")

def analyze_documents(reports : dict):
    # [START analyze_documents_output_in_markdown]
    endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
    key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
    json_folder = "json_reports"
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(AZURE_MARKDOWNS, exist_ok=True)
    
    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    for filename, url in reports.items():
        title = filename.split(".")[0]
        json_destination = os.path.join(json_folder, title)
        markdown_destination = os.path.join(AZURE_MARKDOWNS, title)

        print(f"⬇️  Transforming: {title}")

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(url_source=url),
            output_content_format=DocumentContentFormat.MARKDOWN
        )
        result: AnalyzeResult = poller.result()

        with open(f"{json_destination}.json", "w", encoding = "utf-8") as file:
            json.dump(result.as_dict(), file, indent=4)
            
        with open(f"{markdown_destination}.md", "w", encoding = "utf-8") as file:
            file.write(result.content)
        print(f"✅ Transformed: {title}")
    # [END analyze_documents_output_in_markdown]

def chunk_w_metadata(file_path, title):
    with open(file_path, "r", encoding = 'utf-8') as f:
        report_text = f.read()
        report_text = re.sub(r'<!-- PageHeader="(.*?)" -->', r'## \1', report_text)
    
    title_data = title.split("_")
    ticker = title_data[0]
    calendar_year = title_data[1]
    fiscal_quarter = title_data[2]

    if fiscal_quarter in {"Q1", "Q2", "Q3"}:
        document_type = "10-Q"
        is_audited = False

        match = re.search(r"months ended ([A-Za-z]+ \d+, \d{4})", report_text)
        filing_date = match.group(1).strip() if match else None

        if filing_date:
            filing_date = datetime.strptime(filing_date, "%B %d, %Y")
            filing_date = filing_date.strftime("%Y-%m-%d")
    elif fiscal_quarter == "Q4":
        document_type = "10-K"
        is_audited = True

        match = re.search(r"year ended ([A-Za-z]+ \d+, \d{4})", report_text)
        filing_date = match.group(1).strip() if match else None

        if filing_date:
            filing_date = datetime.strptime(filing_date, "%B %d, %Y")
            filing_date = filing_date.strftime("%Y-%m-%d") 
    else:
        document_type = None
        is_audited = None
        filing_date = None

    company_names = {"TD": "TD Bank Group", "CTC" : "Canadian Tire"}

    # 1. Split the document by Headers
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    raw_chunks = splitter.split_text(report_text)


    # 2. Stateful Page Tracking
    current_page = 1  # Default starting page
    page_marker_pattern = r"--- PAGE (\d+) ---"
    
    cleaned_chunks = {}
    chunk_num = 0

    for chunk in raw_chunks:
        # Find all page markers inside this specific chunk
        markers = re.findall(page_marker_pattern, chunk.page_content)
        
        # Determine which pages this chunk covers
        spanned_pages = [current_page]
        
        if markers:
            # Add all new pages found in this chunk to our spanned list
            for marker in markers:
                page_num = int(marker)
                if page_num not in spanned_pages:
                    spanned_pages.append(page_num)
            
            # Update the global state so the NEXT chunk knows where it's starting
            current_page = int(markers[-1])

        # 3. Inject the page metadata
        # We store both the starting page and a list of all pages it touches
        chunk.metadata.update({
            "chunk_id" : title.lower() + "_" + str(chunk_num),
            "ticker": ticker,
            "company_name": company_names[ticker],
            "document_type": document_type,
            "calendar_year": calendar_year,
            "fiscal_quarter": fiscal_quarter,
            "is_audited": is_audited,
            "filing_date": filing_date,
            "page(s)": spanned_pages
        })

        # 4. Clean the chunk (Remove the markers so they don't confuse the LLM)
        chunk.page_content = re.sub(page_marker_pattern, "", chunk.page_content).strip()
        chunk.page_content = re.sub(r'<!-- PageFooter=".*?" -->', '', chunk.page_content)
        chunk.page_content = re.sub(r'<!-- PageNumber=".*?" -->', '', chunk.page_content)
        
        
        # Only keep chunks that actually have text left after cleaning
        if chunk.page_content:
            cleaned_chunks[f"{title}_{chunk_num}"] = {
                "content" : chunk.page_content,
                "metadata": chunk.metadata
            }
            chunk_num += 1

    output = os.path.join(OUTPUT_DIR, title + ".json")
    with open(output, "w", encoding = "utf-8") as file:
        json.dump(cleaned_chunks, file, indent = 4)


def add_llm_metadata(file_path, title):
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = json.loads(file.read())

    for chunk_name, chunk_data in chunks.items():

        metadata = chunks.get(chunk_name, {}).get("metadata", {})
        summary = metadata.get("section_summary")
        if summary == "" or summary:
            print(f"Skipped {chunk_name}")
            continue

        # Extracted content 
        content = chunk_data.get("content", "")
        
        prompt = f"""
        Analyze the following text block. You must respond ONLY with a valid JSON object containing exactly these five keys:
        - "section_summary": A one-sentence summary of the text.
        - "section_title": A short, 3-5 word title for the text.
        - "questions_answered": A single string containing 3 questions this text answers, separated by commas.
        - "financial_tags": A list/array of short tags indicating financial topics (e.g., ["Revenue", "Risk"]), or ["None"] if not financial.
        - "is_table": A boolean value (true or false) indicating whether the primary content is a formatted table.
        
        Text block to analyze:
        {content}
        """

        max_retries = 5
        model_id = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        groq_client = get_groq_client()
        for attempt in range(max_retries):
            try:
                # API Call using Groq's JSON mode
                response = groq_client.chat.completions.create(
                    model=model_id[attempt % 2], 
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful data-extraction assistant. You always output strictly valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"}
                )

                # Parse the guaranteed JSON response
                answer = response.choices[0].message.content
                llm_data = json.loads(answer)

                # Update dictionary safely using .get() to prevent KeyErrors
                chunks[chunk_name]["metadata"].update({
                    "section_summary": llm_data.get("section_summary", "No summary provided."),
                    "section_title": llm_data.get("section_title", "Untitled"),
                    "questions_answered": llm_data.get("questions_answered", ""),
                    "financial_tags": llm_data.get("financial_tags", ["None"]),
                    "is_table": llm_data.get("is_table", False)
                })
                
                print(f"Successfully processed {chunk_name}")
                
                # Baseline pace to respect the 30 RPM limit (1 request every ~2 seconds)
                time.sleep(2.1)
                break # Break out of retry loop on success
                
            except Exception as e:
                error_msg = str(e).lower()
                # Catch Rate Limits specifically (429 Error)
                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = (attempt + 1) * 10 # Exponential backoff: 10s, 20s, 30s...
                    print(f"Rate limit hit (TPM/RPM maxed). Switching models and sleeping for {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Unexpected error on {chunk_name}: {e}")
                    break # Break on non-rate-limit errors so you don't infinitely loop on bad data

    # Save output
    output_path = os.path.join(OUTPUT_DIR, f"{title}.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4)
        
    print(f"Finished processing. Saved to {output_path}")

def embedding(docs: list[str], input_type: str, model_id : str): 
    vo_client = get_voyage_client()
    dense_vectors = vo_client.embed(
            docs,
            model=model_id,
            input_type=input_type
    ).embeddings
    return dense_vectors

def create_embeddings(file_path):
    
    # 1. Initialize Pinecone
    index = get_pinecone_index()

    # 2. Initialize Voyage
    embed_model = config.EMBED_MODEL

    # 3. Load Data
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.loads(f.read())

    # We need to extract the keys and contents into ordered lists so they match up later
    chunk_ids = list(chunks.keys())
    chunk_contents = [chunks[cid]["content"] for cid in chunk_ids]

    # 4. Batch the Voyage Embedding Requests
    dense_embeddings = []
    sparse_embeddings = []
    voyage_batch_size = 50  # Adjust this if you still hit the token limit
    
    print(f"Embedding {len(chunk_contents)} chunks in batches of {voyage_batch_size}...")
    
    for i in range(0, len(chunk_contents), voyage_batch_size):
        batch_docs = chunk_contents[i : i + voyage_batch_size]
        batch_embeddings = embedding(
            docs=batch_docs, 
            input_type="document", 
            model_id=embed_model
        )
        normalized_vectors = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        dense_embeddings.extend(normalized_vectors)
        sparse_embeddings.extend(BM25.encode_documents(batch_docs))

    # 5. Assemble the Vectors for Pinecone
    vectors = []
    for c, d, s in zip(chunks, dense_embeddings, sparse_embeddings):
        # 1. Grab JUST the metadata dictionary from your JSON
        valid_metadata = chunks[c]["metadata"].copy()
        
        # 2. Add the raw text content into this flat dictionary 
        # (You need this so your RAG can return the actual text to the user)
        valid_metadata["text"] = chunks[c]["content"]
        
        # 3. Fix the page(s) array by converting any numbers to strings
        if "page(s)" in valid_metadata:
            valid_metadata["page(s)"] = [str(p) for p in valid_metadata["page(s)"]]
            
        # 4. Append to Pinecone payload
        vectors.append({
            "id": chunks[c]["metadata"]["chunk_id"], # Usually better to use the chunk_id from metadata
            "values": d,
            "sparse_values": s,
            "metadata": valid_metadata
        })
    
    # 6. Batch the Pinecone Upserts (Pinecone limits payload sizes)
    pinecone_batch_size = 100
    print(f"Upserting vectors to Pinecone in batches of {pinecone_batch_size}...")
    
    for i in range(0, len(vectors), pinecone_batch_size):
        batch_vectors = vectors[i : i + pinecone_batch_size]
        index.upsert(
            vectors=batch_vectors,
            namespace="ns1"
        )
    
    print("Embedding and indexing complete!")
        

#if __name__ == "__main__":
    
    #reports = find_reports(2025, 2025)
    #download_reports(reports)
    #analyze_documents(reports)
    #process_module.main()

    #for file, _ in reports.items():
        #title = file.split(".")[0]
        #markdown_path = os.path.join(OUTPUT_DIR, title + ".md")
        #json_path = os.path.join(OUTPUT_DIR, title + ".json")
        #chunk_w_metadata(markdown_path, title)
        #add_llm_metadata(json_path, title)
        #create_embeddings(json_path) 
    

