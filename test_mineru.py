
from pathlib import Path  
from dotenv import load_dotenv 
# Load environment variables from .env file  
load_dotenv()  

from mineru.cli.common import do_parse, read_fn  
from mineru.utils.enum_class import MakeMode   
import os  
  

# Example for French PDF  
pdf_path_fr = Path("document.pdf")  
pdf_bytes_fr = read_fn(pdf_path_fr)  
  
do_parse(  
    output_dir=os.getenv("OUTPUT_DIR", "./output"),  # Output directory path (string)  
    pdf_file_names=[pdf_path_fr.stem],  # List of PDF file names without extension (list[str])  
    pdf_bytes_list=[pdf_bytes_fr],  # List of PDF bytes data (list[bytes])  
    p_lang_list=["en"],  # Language codes: 'ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'th', 'el', 'latin', 'arabic', 'east_slavic', 'cyrillic', 'devanagari' [1](#3-0)   
    backend="pipeline",  # Backend options: 'pipeline', 'vlm-http-client', 'hybrid-http-client', 'vlm-auto-engine', 'hybrid-auto-engine' [2](#3-1)   
    parse_method="text",  # Parse method: 'auto', 'txt', 'ocr' (pipeline and hybrid* backend only) [3](#3-2)   
    formula_enable=False,  # Enable formula parsing (boolean: True/False)  
    table_enable=False,  # Enable table parsing (boolean: True/False)  
    server_url=None,  # Server URL for http-client backends (string or None)  
    f_draw_layout_bbox=True,  # Draw layout bounding boxes (boolean: True/False)  
    f_draw_span_bbox=True,  # Draw span bounding boxes (boolean: True/False)  
    f_dump_md=True,  # Generate markdown files (boolean: True/False)  
    f_dump_middle_json=False,  # Generate intermediate JSON files (boolean: True/False)  
    f_dump_model_output=False,  # Generate raw model output files (boolean: True/False)  
    f_dump_orig_pdf=True,  # Copy original PDF files (boolean: True/False)  
    f_dump_content_list=True,  # Generate content list files (boolean: True/False)  
    f_make_md_mode=MakeMode.MM_MD,  # Markdown mode: 'mm_markdown', 'nlp_markdown', 'content_list', 'content_list_v2' [4](#3-3)   
    start_page_id=0,  # Starting page ID (integer, 0-based)  
    end_page_id=None,  # Ending page ID (integer or None for all pages)  
)

"""
Mode Differences
MM_MD (mm_markdown)
    Purpose: Standard markdown format with full multimedia support
    Output: Markdown string with images, tables, formulas, and text
    Use Case: General document processing, human-readable output
    Features: Includes image references, HTML tables, LaTeX formulas
NLP_MD (nlp_markdown)
    Purpose: Text-only markdown optimized for NLP processing
    Output: Plain text markdown without images or tables
    Use Case: Machine learning, text analysis, search indexing
    Features: Excludes images (continue in image processing) pipeline_middle_json_mkcontent.py:31-33
CONTENT_LIST (content_list)
    Purpose: Structured JSON with content blocks and metadata
    Output: List of content objects with type, text, and bounding boxes
    Use Case: Programmatic processing, layout analysis
    Features: Includes bbox coordinates, content types, page info pipeline_middle_json_mkcontent.py:182-261
CONTENT_LIST_V2 (content_list_v2)
    Purpose: Enhanced structured JSON with detailed content hierarchy
    Output: Nested content structure with span-level details
    Use Case: Advanced document understanding, fine-grained analysis
    Features: Span-level content types, enhanced metadata vlm_middle_json_mkcontent.py:285-370
"""