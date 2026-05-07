import os
import re
import json
from pathlib import Path
from src.common import config

# Configuration
INPUT_DIR = config.PROCESSED_DATA_DIR
OUTPUT_DIR = config.FINAL_DATA_DIR
BOILERPLATE_PATTERNS = [
    r"The financial information in this document is reported in Canadian dollars.*?(?=##|$)",
    r"Reported results conform with generally accepted accounting principles.*?(?=##|$)",
    r"Caution Regarding Forward-Looking Statements.*?(?=##|$)",
]

def clean_html_tables(content):
    """Removes empty table rows and cells that add noise."""
    # Remove empty cells: <td></td> or <td>  </td>
    content = re.sub(r'<td>\s*</td>', '', content)
    # Remove empty rows: <tr>\s*</tr>
    content = re.sub(r'<tr>\s*</tr>', '', content)
    # Clean up multiple newlines inside tables
    content = re.sub(r'(<table>.*?</table>)', lambda m: re.sub(r'\n\s*\n', '\n', m.group(1), flags=re.DOTALL), content, flags=re.DOTALL)
    return content

def normalize_numbers(content):
    """Ensures consistent currency and number formatting."""
    # Remove spaces between currency symbols and numbers: "$ 51,730" -> "$51,730"
    content = re.sub(r'(\$|€|£)\s*(\d)', r'\1\2', content)
    return content

def fix_bullet_points(content):
    """Standardizes inconsistent bullet points (·, •, .) to Markdown '-'."""
    # Replace middle dots, bullet symbols, and leading dots followed by a space
    # at the start of a line with a standard markdown bullet.
    # We use multiline mode to catch the start of every line.
    content = re.sub(r'^[ \t]*[·•●]\s*', '- ', content, flags=re.MULTILINE)
    # Be careful with periods - only replace if it's a period at the start of a line followed by a space
    content = re.sub(r'^[ \t]*\.\s+', '- ', content, flags=re.MULTILINE)
    return content

def resolve_footnotes(content):
    """Identifies and inlines footnotes (e.g., word1 or (Note 2)) for better RAG context."""
    lines = content.split('\n')
    footnote_defs = {}
    clean_lines = []
    
    # Scan from bottom to top for footnote definitions
    # Usually start with a number + space at beginning of line
    # but we skip lines that are page numbers or headers
    for line in lines:
        stripped = line.strip()
        # Look for "1 Footnote text" or "2 Footnote text"
        # Skip items that look like lists or table indices
        match = re.match(r'^(\d+)\s+([A-Z].*)', stripped)
        if match and not re.search(r'Page \d+', stripped):
            num = match.group(1)
            text = match.group(2)
            footnote_defs[num] = text
        else:
            clean_lines.append(line)

    if not footnote_defs:
        return content

    new_content = "\n".join(clean_lines)
    
    # Replace markers in text
    for num, text in footnote_defs.items():
        # Handle word1 or text1 (trailing digits)
        # Using [a-zA-Z] to avoid matching years or actual numbers
        pattern = rf'([a-z]|\n)({num})(\b|[^\d])'
        replacement = rf'\1 [Footnote {num}: {text}]\3'
        new_content = re.sub(pattern, replacement, new_content)
        
        # Handle (Note 1) or (Notes 1, 2)
        new_content = new_content.replace(f'(Note {num})', f'(Note {num}: {text})')
        new_content = new_content.replace(f'<td>{num}</td>', f'<td>[Footnote {num}: {text}]</td>')

    return new_content

def extract_metadata_from_filename(filename):
    """Extracts Bank, Year, and Quarter from filename (e.g., TD_2025_Q1.md)."""
    parts = filename.replace('.md', '').split('_')
    return {
        "bank": parts[0] if len(parts) > 0 else "Unknown",
        "year": parts[1] if len(parts) > 1 else "Unknown",
        "quarter": parts[2] if len(parts) > 2 else "Unknown"
    }

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    file_metadata = extract_metadata_from_filename(file_path.name)
    
    # Split by PageBreak to process metadata page-by-page
    pages = raw_content.split('<!-- PageBreak -->')
    processed_pages = []

    for i, page_content in enumerate(pages):
        page_num = i + 1
        
        # Extract page-specific metadata from comments
        page_header_match = re.search(r'<!-- PageHeader="(.*?)" -->', page_content)
        page_num_match = re.search(r'<!-- PageNumber="Page (\d+)" -->', page_content)
        
        page_header = page_header_match.group(1) if page_header_match else ""
        if page_num_match:
            page_num = int(page_num_match.group(1))

        # 1. Clean Comments (already extracted)
        #clean_page = re.sub(r'<!--.*?-->', '', page_content).strip()
        
        # 2. Remove Boilerplate (optional - uncomment if you want to strip it)
        # for pattern in BOILERPLATE_PATTERNS:
        #    clean_page = re.sub(pattern, '', clean_page, flags=re.DOTALL | re.IGNORECASE)

        # 3. Clean Tables
        clean_page = clean_html_tables(page_content)
        
        # 4. Normalize Numbers
        clean_page = normalize_numbers(clean_page)
        
        # 5. Fix Bullet Points
        clean_page = fix_bullet_points(clean_page)
        
        # 6. Resolve Footnotes
        clean_page = resolve_footnotes(clean_page)

        if clean_page:
            processed_pages.append({
                "text": clean_page,
                "metadata": {
                    **file_metadata,
                    "page": page_num,
                    "header": page_header,
                    "source_file": file_path.name
                }
            })

    return processed_pages

def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    all_processed_data = []

    for md_file in input_path.glob("*.md"):
        print(f"Processing {md_file.name}...")
        page_data = process_file(md_file)
        all_processed_data.extend(page_data)
        
        # Also save individual cleaned files for inspection
        with open(output_path / f"{md_file.name}", 'w', encoding='utf-8') as f:
            for page in page_data:
                f.write(f"--- PAGE {page['metadata']['page']} ---\n")
                f.write(page['text'] + "\n\n")

    # Save as a single JSON for easy RAG loading
    #with open(output_path / "rag_ready_data.json", 'w', encoding='utf-8') as f:
        #json.dump(all_processed_data, f, indent=2)

    print(f"\nSuccess! Processed data saved to {OUTPUT_DIR}")
    print(f"Total chunks/pages processed: {len(all_processed_data)}")

if __name__ == "__main__":
    main()
