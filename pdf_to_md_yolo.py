import os
import sys
import glob
import cv2
import argparse
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from sklearn.cluster import KMeans

try:
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download
except ImportError:
    print("doclayout_yolo or huggingface_hub package is not installed. Please install them.")
    sys.exit(1)

try:
    import camelot
    from html_to_markdown import convert as html_to_md_convert
    from bs4 import BeautifulSoup
except ImportError:
    print("camelot-py, html-to-markdown, or beautifulsoup4 package is not installed. Please install them.")
    sys.exit(1)

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0 # Enforce consistent language detection
except ImportError:
    print("langdetect package is not installed. Please install it using: pip install langdetect")
    sys.exit(1)

# Mapping ISO 639-1 (langdetect) to ISO 639-2 (Tesseract)
LANG_MAP = {
    'en': 'eng', 'fr': 'fra', 'es': 'spa', 'de': 'deu',
    'it': 'ita', 'pt': 'por', 'nl': 'nld', 'ru': 'rus',
    'zh-cn': 'chi_sim', 'zh-tw': 'chi_tra', 'ja': 'jpn',
    'ko': 'kor', 'ar': 'ara'
}

VIS_COLORS = {
    'title': (0, 0, 255),               # Red (BGR)
    'plain text': (0, 255, 0),          # Green
    'abandon': (128, 128, 128),         # Gray
    'figure': (255, 0, 0),              # Blue
    'figure caption': (255, 100, 100),  # Light Blue
    'table': (0, 165, 255),             # Orange
    'table caption': (0, 200, 255),     # Light Orange
    'isolate formula': (255, 0, 255),   # Magenta
    'formula caption': (255, 100, 255)  # Light Magenta
}

import re

PUNCT_SPACE = re.compile(r'\s+([.,»!?;])')
PUNCT_SPACE2 = re.compile(r'([«])\s+')
PUNCT_PARAS = re.compile(r'[•}^$■]')
PUNCT_POINT = re.compile(r'\.{4,}')
PUNCT_DASH = re.compile(r'(-—)|(—-)')
PUNCT_DASH2 = re.compile(r'([^ ])—')
PUNCT_DASH3 = re.compile(r'—([^ ])')

def clean_extracted_text(text):
    """
    Cleans up punctuation and unnecessary line breaks natively returned by PDF parsers.
    It merges hyphenated word wraps seamlessly and combines paragraph lines with spaces.
    """
    if not text: return ""
    
    lines = text.split('\n')
    processed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        if len(line) < 3:
            processed_lines.append(line)
            i += 1
            continue
            
        # Detect end-of-line hyphens for word continuation
        merge_next = False
        if line.endswith('\xad') or line.endswith('-'):
            line = line[:-1] # strip the hyphen
            merge_next = True
            
        # Clean up punctuation spacing
        line = PUNCT_SPACE.sub(r'\1', line)
        line = PUNCT_SPACE2.sub(r'\1', line)
        line = PUNCT_PARAS.sub('', line)
        line = PUNCT_POINT.sub('...', line)
        line = PUNCT_DASH.sub('—', line)
        line = PUNCT_DASH2.sub(r'\1 —', line)
        line = PUNCT_DASH3.sub(r'— \1', line)
        
        if merge_next and i + 1 < len(lines):
            # Prep the next line to receive the prefix without any space
            lines[i+1] = line + lines[i+1].strip()
        else:
            processed_lines.append(line)
            
        i += 1
        
    # Join with a single space to construct unbroken markdown paragraphs
    result = " ".join(processed_lines)
    result = re.sub(r'\s+', ' ', result)
    return result.strip()

def sanitize_filename(name):
    """Keep only alphanumeric characters, dashes, and underscores for safe directory creation"""
    # Replace spaces with underscores and strip all other special symbols
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w\-]', '', name)
    return name.lower()

def get_center(xyxy):
    # Retrieve center coordinates of bounding box
    x_min, y_min, x_max, y_max = map(int, xyxy)
    return (x_min + x_max) / 2, (y_min + y_max) / 2

def get_distance(p1, p2):
    # Calculate Euclidean distance between two points
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

def is_contained(box_inner, box_outer, tolerance=0.9):
    """
    Returns True if box_inner is almost entirely (e.g. 90%+) inside box_outer.
    """
    x_left = max(box_inner[0], box_outer[0])
    y_top = max(box_inner[1], box_outer[1])
    x_right = min(box_inner[2], box_outer[2])
    y_bottom = min(box_inner[3], box_outer[3])
    
    if x_right < x_left or y_bottom < y_top:
        return False
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1])
    
    if inner_area == 0:
        return False
        
    return (intersection_area / inner_area) >= tolerance

def filter_duplicate_elements(elements, iou_threshold=0.8):
    """
    Remove heavily overlapping elements of the EXACT SAME type (like 2 layered titles),
    OR swallow smaller elements strictly contained within larger elements.
    Elements is list: [ [[x1,y1,x2,y2], name], ... ]
    """
    if not elements: return []
    
    # We will mutate the elements in place for unions, so work on a copy of the list structure
    working_elements = [[e[0].copy(), e[1]] for e in elements]
    skip_indices = set()
    
    for i in range(len(working_elements)):
        if i in skip_indices: continue
        
        for j in range(i + 1, len(working_elements)):
            if j in skip_indices: continue
            
            box_i = working_elements[i][0]
            box_j = working_elements[j][0]
            class_i = working_elements[i][1].lower().replace("_", " ")
            class_j = working_elements[j][1].lower().replace("_", " ")
            
            textual_classes = {"title", "plain text", "text"}
            is_same_class = class_i == class_j
            is_overlap_textual = (class_i in textual_classes) and (class_j in textual_classes)
            
            should_check = is_same_class or is_overlap_textual
            
            if should_check:
                # 1. Check for Total Containment (Swallowing)
                i_contains_j = is_contained(box_j, box_i)
                j_contains_i = is_contained(box_i, box_j)
                
                if i_contains_j or j_contains_i:
                    # Expand the survivor to the absolute Union of both boxes
                    new_x1 = min(box_i[0], box_j[0])
                    new_y1 = min(box_i[1], box_j[1])
                    new_x2 = max(box_i[2], box_j[2])
                    new_y2 = max(box_i[3], box_j[3])
                    union_box = np.array([new_x1, new_y1, new_x2, new_y2])
                    
                    if i_contains_j:
                        working_elements[i][0] = union_box
                        skip_indices.add(j)
                    else:
                        working_elements[j][0] = union_box
                        skip_indices.add(i)
                        break # outer element i is dead, stop evaluating against it
                else:
                    # 2. Check for Standard Heavy Intersection
                    iou = calculate_iou(box_i, box_j)
                    if iou > iou_threshold:
                        skip_indices.add(j)
                        
    filtered_elements = [working_elements[i] for i in range(len(working_elements)) if i not in skip_indices]
    return filtered_elements

def detect_ocr_language(sample_text):
    if not sample_text or len(sample_text.strip()) < 10:
        return 'eng' # Default if not enough text to detect confidently
    try:
        iso_code = detect(sample_text)
        return LANG_MAP.get(iso_code, 'eng') # Fallback to English if unknown mapping
    except:
        return 'eng'

def extract_title_font_size(doc, page_num, fitz_rect):
    """
    Extracts the median font_size of the text enclosed in the rectangle.
    Returns: Float font_size or None.
    """
    if not doc: return None
    try:
        page = doc[page_num]
        words = page.get_text("dict", clip=fitz_rect)
        sizes = []
        for block in words.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span.get("size", 0))
        if sizes:
            sizes.sort()
            return sizes[len(sizes)//2] # Return median to ignore weird stray characters
    except Exception as e:
        pass
    return None

def check_bold_title_promotion(doc, page_num, fitz_rect):
    """
    Evaluates if ALL text within a bounding box is bold.
    Returns True if entire box is bold text, False otherwise.
    """
    if not doc: return False
    try:
        page = doc[page_num]
        words = page.get_text("dict", clip=fitz_rect)
        
        has_text = False
        for block in words.get("blocks", []):
            if block.get("type", 0) != 0: # Only analyze text blocks
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text: continue # Ignore empty whitespace spans
                    
                    has_text = True
                    font_flags = span.get("flags", 0)
                    font_name = span.get("font", "").lower()
                    
                    # 16 is the bitmask for bold in PyMuPDF's span flags
                    is_bold_flag = bool(font_flags & 16)
                    is_bold_name = "bold" in font_name or "black" in font_name or "heavy" in font_name
                    
                    # If ANY valid textual span inside this bounding box is NOT bold, the promotion fails!
                    if not (is_bold_flag or is_bold_name):
                        return False
                        
        # If we found valid text and never triggered the non-bold failure condition, it is a pure bold box!
        return has_text 
    except Exception as e:
        pass
        
    return False

def infer_heading_level(text, font_size, seen_fonts):
    """
    1) Try Numbering (e.g., '1.2.1 Details')
    2) Try Font Size ranking (Largest = 1, Second = 2)
    3) Fallback based on Word Count length
    """
    text_clean = text.strip()
    
    # 1. Numbering Check matches 1, 1.1, 1.2.1, etc.
    m = re.match(r'^(\d+(\.\d+)*)', text_clean)
    if m:
        return m.group(1).count('.') + 1

    # 2. Font Size Check
    if font_size and seen_fonts:
        # Create a ranking: largest font -> level 1, second largest -> level 2, etc.
        sizes = sorted(list(seen_fonts), reverse=True)
        # Assign numeric level based on rank (1-indexed)
        font_map = {size: idx + 1 for idx, size in enumerate(sizes)}
        
        # Round font size slightly to collapse minute float discrepancies in extraction
        rounded_size = round(font_size, 1)
        # Find closest matching font size in map
        closest_size = min(font_map.keys(), key=lambda k: abs(k - rounded_size))
        
        if abs(closest_size - rounded_size) < 1.0: # Only trust it if we have a near-exact match
             return font_map[closest_size]

    # 3. Fallback check
    words = len(text_clean.split())
    if words <= 4:
        return 1
    elif words <= 8:
        return 2
    else:
        return 3

def clean_camelot_html(html_string):
    """Clean up formatting/padding artifacts dynamically via BeautifulSoup"""
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        
        # Target all rows in the table (bypassing thead/tbody strictness)
        rows = soup.find_all("tr")
        if not rows:
            return html_string
            
        # 1. Top padding removal
        while rows and all(not cell.get_text(strip=True) for cell in rows[0].find_all(["td", "th"])):
            rows[0].decompose()
            rows.pop(0)
            
        # 2. Bottom padding removal
        while rows and all(not cell.get_text(strip=True) for cell in rows[-1].find_all(["td", "th"])):
            rows[-1].decompose()
            rows.pop(-1)
            
        # 3. Empty columns removal
        if rows:
            # Find the max cols to avoid zip truncation if rows are uneven
            max_cols = max(len(tr.find_all(["td", "th"])) for tr in rows)
            
            # Identify columns that are empty across ALL surviving rows
            empty_cols_indices = []
            for col_idx in range(max_cols):
                is_empty = True
                for tr in rows:
                    cells = tr.find_all(["td", "th"])
                    if col_idx < len(cells):
                        if cells[col_idx].get_text(strip=True):
                            is_empty = False
                            break
                if is_empty:
                    empty_cols_indices.append(col_idx)
                    
            # Remove from right-to-left to avoid index shifting breaking destruction
            for tr in rows:
                cells = tr.find_all(["td", "th"])
                for idx in reversed(empty_cols_indices):
                    if idx < len(cells):
                        cells[idx].decompose()
                        
        return str(soup)
    except Exception as parse_e:
        print(f"  [WARN] BeautifulSoup cleaning failed, falling back to raw HTML. ({parse_e})")
        return html_string

def extract_table_camelot(input_path, page_num_str, x_min, y_min, x_max, y_max, pdf_h, pdf_w):
    """
    Extracts a table using Camelot logic mapped from YOLO coordinates.
    Returns: A clean Markdown string of the table, or None if extraction failed.
    """
    try:
        # Map 300 DPI YOLO coordinates to 72 DPI PDF coordinates with PADDING
        scale = 72 / 300
        padding = 3 # Give Camelot a slightly larger viewport to "see" the table borders
        
        x1 = max(0, (x_min * scale) - padding)
        y1_orig = pdf_h - (y_min * scale)
        y1 = min(pdf_h, y1_orig + padding) # Top-Left Y (PDF origin is bottom-left, so + pushes it "up")
        
        x2 = min(pdf_w, (x_max * scale) + padding)
        y2_orig = pdf_h - (y_max * scale)
        y2 = max(0, y2_orig - padding) # Bottom-Right Y (- pushes it "down")
        
        # Camelot expects string format "x1,y1,x2,y2"
        table_area = f"{x1},{y1},{x2},{y2}"
        
        print(f"  [GEOMETRY] Mapping table crop {table_area} to Camelot Lattice engine...")
        tables = camelot.read_pdf(
            input_path, 
            pages=page_num_str, 
            flavor='lattice',
            table_areas=[table_area]
        )
        
        if len(tables) > 0:
            # When given a region containing multiple tables, Camelot might return all of them.
            # Find the table whose center is closest to our YOLO bounding box center
            target_table = tables[0]
            if len(tables) > 1:
                yolo_center_x = (x1 + x2) / 2
                yolo_center_y = (y1 + y2) / 2
                min_dist = float('inf')
                for t in tables:
                    tx1, ty1, tx2, ty2 = t._bbox
                    t_center_x = (tx1 + tx2) / 2
                    t_center_y = (ty1 + ty2) / 2
                    dist = ((t_center_x - yolo_center_x)**2 + (t_center_y - yolo_center_y)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        target_table = t
            
            import tempfile
            import os
            temp_html = tempfile.mktemp(suffix='.html')
            # Bypass Camelot's default HTML export which writes pandas DataFrame indices to the table headers
            target_table.df.to_html(temp_html, index=False, header=False)
            
            with open(temp_html, 'r', encoding='utf-8') as html_file:
                html_string = html_file.read()
            os.remove(temp_html)
            
            clean_html = clean_camelot_html(html_string)
            md_table = html_to_md_convert(clean_html)
            
            # Explicitly delete table objects to release file locks on Windows
            del tables
            
            if md_table and len(md_table.strip()) > 5:
                return md_table
                
    except Exception as e:
        print(f"  [WARN] Camelot extraction failed: {e}")
    finally:
        # Force Garbage Collection to purge dangling pypdf/ghostscript handles inside Camelot
        import gc
        gc.collect()
        
    return None

def sort_elements_multicolumn(elements, page_width):
    if not elements:
        return []
        
    # Pre-sort everything top-to-bottom so our iteration through the page is strictly chronological
    elements.sort(key=lambda x: x[0][1])
        
    # 1. Identify wide elements (>60% page width) to serve as column breakers (e.g. Banners/Large Titles)
    wide_elements_idx = []
    for idx, (xyxy, name) in enumerate(elements):
        w = xyxy[2] - xyxy[0]
        if w > 0.6 * page_width:
            wide_elements_idx.append(idx)
            
    # 2. Slice page horizontally by the breakers
    bands = []
    current_y = 0
    wide_bounds = []
    
    for idx in wide_elements_idx:
        xyxy = elements[idx][0]
        y_min, y_max = xyxy[1], xyxy[3]
        if y_min > current_y:
            bands.append((current_y, y_min))
        wide_bounds.append((idx, y_min, y_max))
        current_y = y_max
        
    bands.append((current_y, float('inf')))
    print(f"  [KMEANS] Page separated into {len(bands)} vertical bands broken by {len(wide_elements_idx)} wide-spanning titles/images.")
    
    placed_indices = set(wide_elements_idx)
    sorted_output = []
    column_separators = []
    
    # 3. Process the bands using KMeans horizontally, injecting the wide banners between them!
    for b_idx in range(len(bands)):
        b_top, b_bottom = bands[b_idx]
        band_elements = []
        
        for idx in range(len(elements)):
            if idx in placed_indices: continue
            xyxy = elements[idx][0]
            # Use the TOP of the element to determine which band it belongs in
            y_top = xyxy[1]
            
            # Handle float('inf') math gracefully
            bottom_val = b_bottom if b_bottom != float('inf') else float('inf')
            
            if b_top <= y_top < bottom_val:
                band_elements.append(elements[idx])
                placed_indices.add(idx)
                
        if band_elements:
            # Group by Left-most text alignment rather than center, since ragged-right documents warp center calculations
            x_lefts = np.array([e[0][0] for e in band_elements]).reshape(-1, 1)
            
            n_cols = 1
            if len(band_elements) >= 2:
                # Perform an Inertia-based (Variance Drop) evaluation to choose 1 vs 2 columns dynamically
                km1 = KMeans(n_clusters=1, random_state=0, n_init='auto').fit(x_lefts)
                
                if km1.inertia_ > 0:
                    km2 = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(x_lefts)
                    
                    # If grouping into 2 columns destroys >60% of the variance (meaning the layout is distinctly bi-modal)...
                    if km2.inertia_ < (km1.inertia_ * 0.4):
                        # Ensure the columns are ACTUALLY physically separated (e.g. > 10% of page width)
                        # This prevents severe paragraph indents from creating microscopic fake columns!
                        centers = sorted(km2.cluster_centers_.flatten())
                        if (centers[1] - centers[0]) > (0.1 * page_width):
                            n_cols = 2
                        
            print(f"  [KMEANS] Scanning Band {b_idx + 1} ({len(band_elements)} elements) -> Grouping into {n_cols} columns (Inertia Evaluated).")
            
            try:
                # Use left-point sorting to determine columns dynamically
                kmeans = KMeans(n_clusters=n_cols, random_state=0, n_init='auto').fit(x_lefts)
                col_assignment = kmeans.labels_
                col_order = np.argsort(kmeans.cluster_centers_.flatten())
                
                col_groups = {i: [] for i in range(n_cols)}
                for i, col_idx in enumerate(col_assignment):
                    # Append elements to their assigned KMeans column
                    col_groups[col_idx].append(band_elements[i])
                    
                # Sort each column structurally block by block from top-to-bottom
                for col_idx in col_groups:
                    # Sort primarily by Top Y-coordinate (y_min) to ensure true vertical reading order
                    col_groups[col_idx].sort(key=lambda x: x[0][1]) 
                    
                for col_idx in col_order:
                    sorted_output.extend(col_groups[col_idx])
                    
                # Collect actual column separator bounds to visualize them!
                if n_cols == 2:
                    centers = sorted(kmeans.cluster_centers_.flatten())
                    mid_x = (centers[0] + centers[1]) / 2
                    column_separators.append((mid_x, b_top, b_bottom))
                    
            except Exception as e:
                print(f"  [WARN] KMeans failed on this band ({e}). Defaulting to standard Y sort.")
                band_elements.sort(key=lambda x: x[0][1])
                sorted_output.extend(band_elements)

        # Append the specific wide element that borders the BOTTOM of this band
        if b_idx < len(wide_bounds):
            wide_idx = wide_bounds[b_idx][0]
            sorted_output.append(elements[wide_idx])
            
    # Sweep stragglers strictly by reading order
    stragglers = [elements[i] for i in range(len(elements)) if i not in placed_indices]
    if stragglers:
        print(f"  [KMEANS] Sweeping up {len(stragglers)} stray blocks via standard layout fallback.")
        stragglers.sort(key=lambda x: (x[0][1] // 20, x[0][0]))
        sorted_output.extend(stragglers)
        
    return sorted_output, bands, column_separators

# ... Inside process_pdf_to_markdown ...
def process_pdf_to_markdown(input_path, output_md_path, image_output_dir, model, args):
    print(f"\n==================================================")
    print(f"Starting Document: {input_path}")
    print(f"  > Target Markdown: {output_md_path}")
    print(f"  > Target Image Directory: {image_output_dir}")
    print(f"==================================================")
    
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        return
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    
    # Empty out or create the output file so we can stream (append) to it piece by piece
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Extracted Content from {os.path.basename(input_path)}\n\n")

    is_pdf = input_path.lower().endswith('.pdf')
    doc = None
    
    if is_pdf:
        doc = fitz.open(input_path)
        total_pages = len(doc)
        print(f"[INFO] Successfully loaded PDF document with {total_pages} pages.")
    else:
        total_pages = 1
        print(f"[INFO] Loaded static image.")

    detected_lang = None # Will figure this out on the first chunk of text
    annotated_pages = [] # Will store visual debugging representations
    raw_annotated_pages = [] # Will store raw YOLO visualizations before any heuristics
    caption_padding = 20
    
    # Master collection of observed Title Font Sizes (so we can determine what H1/H2 sizes are)
    seen_title_font_sizes = set()
    # Track titles to defer heading level resolution until we've parsed the full document
    deferred_titles = []

    for i in range(total_pages):
        
        md_content = []
        
        print(f"\n======================================")
        print(f"Starting to process Page {i+1}/{total_pages}...")
        print(f"======================================")
        
        # Load exactly one page at a time to prevent memory bloat on large files
        if is_pdf:
            page = doc[i]
            pix = page.get_pixmap(dpi=300, alpha=False)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # OpenCV BGR format for cropping
            page_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            page_img = cv2.imread(input_path)
            if page_img is None:
                print("[ERROR] Failed to read image into memory.")
                continue
        
        # doclayout_yolo expects image in RGB format
        page_img_rgb = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
        pil_img_rgb = Image.fromarray(page_img_rgb)
        vis_img = page_img.copy() # Clone image for visualization
        raw_vis_img = page_img.copy() # Clone image specifically for raw YOLO un-altered visualization
        
        # YOLO layout prediction
        
        # We need the real page width to orchestrate columns
        page_width = page_img.shape[1]
        
        det_res = model.predict(
            pil_img_rgb,
            imgsz=1024,
            conf=0.2,
            device="cpu",
            verbose=False # Turn off default YOLO logging since we handle it
        )
        
        if len(det_res) > 0:
            boxes = det_res[0].boxes
            names = det_res[0].names
            
            elements = []
            for j in range(len(boxes)):
                xyxy = boxes.xyxy[j].cpu().numpy()
                cls_id = int(boxes.cls[j].item())
                cat_name = names[cls_id]
                elements.append([xyxy, cat_name]) # Convert from tuple to list exactly so we can append matched captions inside later
                
            # Filter Overlapping/Duplicate Detections (e.g. YOLO double-detecting a Title exactly on top of itself)
            if args.iou_filter > 0:
                original_count = len(elements)
                elements = filter_duplicate_elements(elements, iou_threshold=args.iou_filter)
                if len(elements) < original_count:
                    print(f"  [CLEANUP] Removed {original_count - len(elements)} duplicate overlapping elements detected by YOLO.")
                
            print(f"[LAYOUT] Detected {len(elements)} unique structural elements on Page {i+1}:")
            for _, cat_name in elements:
                print(f"  - {cat_name}")
                
            # --- Draw Raw YOLO Predictions before ANY layout/caption heuristics ---
            if args.draw_pdf:
                for xyxy, cat_name in elements:
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    h, w = page_img.shape[:2]
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    cat_norm = cat_name.lower().replace("_", " ")
                    color = VIS_COLORS.get(cat_norm, (0, 255, 255))
                    
                    # Draw thick bounding box
                    cv2.rectangle(raw_vis_img, (x_min, y_min), (x_max, y_max), color, 4)
                    
                    # Draw enlarged, highly visible class label slightly above the top border
                    # Ensures it does not clip off the top of the page if the box is at y=0
                    label_y = max(35, y_min - 10) 
                    
                    # Draw thick black outline for extreme contrast, then color fill
                    label_text = cat_name.upper()
                    cv2.putText(raw_vis_img, label_text, (x_min, label_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 8)
                    cv2.putText(raw_vis_img, label_text, (x_min, label_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                
            # Phase 1: STRICT Caption to Image Mapping Logic
            caption_mapping = {}  
            
            search_rules = {
                "figure caption": ["figure"],
                "table caption": ["table"],
                "formula caption": ["isolate formula"]
            }
            
            caption_elements = [(j, e) for j, e in enumerate(elements) if e[1].lower().replace("_", " ") in search_rules]
            image_elements = [(j, e) for j, e in enumerate(elements) if e[1].lower().replace("_", " ") in ["figure", "table", "isolate formula"]]
            
            for cap_idx, cap_elem in caption_elements:
                cap_xyxy, cap_name = cap_elem
                cap_name_lower = cap_name.lower().replace("_", " ")
                valid_target_types = search_rules.get(cap_name_lower, [])
                
                cap_center = get_center(cap_xyxy)
                
                closest_img_idx = None
                shortest_dist = float('inf')
                
                for img_idx, img_elem in enumerate(elements):
                    img_xyxy, img_name = img_elem
                    img_name_lower = img_name.lower().replace("_", " ")
                    
                    if img_name_lower in valid_target_types:
                        img_center = get_center(img_xyxy)
                        dist = get_distance(cap_center, img_center)
                        
                        if dist < shortest_dist:
                            shortest_dist = dist
                            closest_img_idx = img_idx
                
                if closest_img_idx is not None:
                    # Physically extract the text from the caption BEFORE standard pipeline so we can hold the text string
                    extracted_text = ""
                    x_min, y_min, x_max, y_max = map(int, cap_xyxy)
                    h, w = page_img.shape[:2]
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    if is_pdf and doc:
                        scale = 72 / 300
                        fitz_rect = fitz.Rect(x_min * scale, y_min * scale, x_max * scale, y_max * scale)
                        try:
                            pdf_text = doc[i].get_textbox(fitz_rect).strip()
                            if pdf_text: extracted_text = pdf_text
                        except Exception as e: pass
                    
                    if not extracted_text:
                        crop_img = page_img[y_min:y_max, x_min:x_max]
                        if crop_img.size > 0:
                            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            extracted_text = pytesseract.image_to_string(gray, lang=detected_lang or 'eng').strip()
                    
                    if extracted_text:
                        print(f"  [GEOMETRY] Strictly Paired '{cap_name}' ({extracted_text[:20]}...) to '{elements[closest_img_idx][1]}' #{closest_img_idx}")
                        # Attach text mapping
                        if closest_img_idx in caption_mapping:
                            caption_mapping[closest_img_idx] += " " + extracted_text
                        else:
                            caption_mapping[closest_img_idx] = extracted_text
                            
                    # Remove the caption layout node entirely from the flow so we don't accidentally treat it like a normal text block
                    elements[cap_idx][1] = "abandon" 
            
            # Phase 2: Processing Pipeline
            bands = []
            column_separators = []
            if args.enable_kmeans:
                print("  [ANALYSIS] Calculating document reading flow (KMeans Multi-Column)...")
                elements, bands, column_separators = sort_elements_multicolumn(elements, page_width)
            else:
                print("  [ANALYSIS] Skpping KMeans, using standard Vertical Reading flow...")
                elements.sort(key=lambda x: (x[0][1] // 20, x[0][0]))
            
            if args.draw_pdf:
                h, w = page_img.shape[:2]
                
                # Draw Band Separators (Horizontal)
                if bands:
                    for (b_top, b_bottom) in bands:
                        y_start = max(0, int(b_top))
                        y_end = min(h, int(b_bottom)) if b_bottom != float('inf') else h
                        
                        # Draw bright purple horizontal lines to indicate KMeans Vertical Slice Regions!
                        # Don't draw if it's strictly on the edge of the screen!
                        if y_start > 0:
                            cv2.line(vis_img, (0, y_start), (w, y_start), (255, 0, 255), 5) 
                        if y_end < h:
                            cv2.line(vis_img, (0, y_end), (w, y_end), (255, 0, 255), 5)
                            
                # Draw Column Separators (Vertical)
                if column_separators:
                    for (mid_x, b_top, b_bottom) in column_separators:
                        y_start = max(0, int(b_top))
                        y_end = min(h, int(b_bottom)) if b_bottom != float('inf') else h
                        x_pos = int(mid_x)
                        
                        # Draw cyan vertical lines separating the KMeans columns
                        cv2.line(vis_img, (x_pos, y_start), (x_pos, y_end), (255, 255, 0), 5) 
            
            for j, (xyxy, cat_name) in enumerate(elements):
                x_min, y_min, x_max, y_max = map(int, xyxy)
                
                # Ensure the crop coordinates are within image boundaries (300 DPI image scale)
                h, w = page_img.shape[:2]
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)
                
                cat_norm = cat_name.lower().replace("_", " ")
                
                if args.draw_pdf:
                    # Draw Visual Debugging Box and KMeans Reading Order Index
                    color = VIS_COLORS.get(cat_norm, (0, 255, 255)) # Default yellow
                    cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color, 3)
                    # Position number near the top-left of the box (Red with thick white outline)
                    cv2.putText(vis_img, str(j + 1), (x_min + 5, y_min + 65), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 10) # Thick White Outline
                    cv2.putText(vis_img, str(j + 1), (x_min + 5, y_min + 65), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 4) # Bright Red Inside
                
                if cat_norm == "abandon":
                    continue  # Will skip regions naturally flagged as junk, AND the captions we just processed above!
                                    
                # Text Extraction Logic
                if cat_norm in ["title", "plain text", "text"]:
                    extracted_text = ""
                    used_ocr = False
                    
                    # 1. Direct PDF Extraction using fitz
                    if is_pdf and doc:
                        # fitz points are typically 72 DPI, YOLO coordinates are over our 300 DPI image
                        scale = 72 / 300
                        fitz_rect = fitz.Rect(x_min * scale, y_min * scale, x_max * scale, y_max * scale)
                        try:
                            # Extract bounded text from the PDF directly
                            pdf_text = doc[i].get_textbox(fitz_rect).strip()
                            if pdf_text:
                                extracted_text = clean_extracted_text(pdf_text)
                                
                                # Dynamic Bold-to-Title Promotion
                                if cat_norm != "title" and args.enable_bold_title:
                                    is_fully_bold = check_bold_title_promotion(doc, i, fitz_rect)
                                    if is_fully_bold:
                                        print(f"  [HEURISTIC] Auto-promoting Bold Text box to 'Title': '{extracted_text[:30]}...'")
                                        cat_norm = "title"
                                
                            # Always attempt font extraction for titles to build our H1/H2 map!
                            if cat_norm == "title":
                                fs = extract_title_font_size(doc, i, fitz_rect)
                                if fs is not None:
                                    seen_title_font_sizes.add(round(fs, 1))
                                    
                        except Exception as e:
                            pass
                    
                    # Do Language Detection on the first discovered text block in the document!
                    if not detected_lang and extracted_text:
                        print(f"  [ANALYSIS] Detecting language for entire document from first text payload...")
                        detected_lang = detect_ocr_language(extracted_text)
                        print(f"  [AUTO-CONFIG] Language selected for PyTesseract OCR: '{detected_lang}'")

                    # 2. OCR Fallback if PDF text was empty or we're processing a raw image
                    if not extracted_text:
                        used_ocr = True
                        print(f"  [WARN] Native text missing for {cat_name}, falling back to OCR engine...")
                        crop_img = page_img[y_min:y_max, x_min:x_max]
                        if crop_img.size > 0:
                            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            extracted_text = pytesseract.image_to_string(gray, lang=detected_lang or 'eng').strip()
                            
                            # Same detection logic catch for files that had NO direct text (Image files)
                            if not detected_lang and extracted_text:
                                print(f"  [ANALYSIS] Detecting language for entire document from first OCR payload...")
                                detected_lang = detect_ocr_language(extracted_text)
                                print(f"  [AUTO-CONFIG] Language selected for PyTesseract OCR: '{detected_lang}'")
                    
                    if extracted_text:
                        extraction_method = "PyTesseract OCR" if used_ocr else "PyMuPDF Direct Native Extraction"
                        print(f"  [EXTRACT] ({extraction_method}): {extracted_text[:40].replace(chr(10), ' ')}...")
                        
                        if cat_norm == "title":
                            # Pull active font size from bounding rect if possible to assist heuristic mapping
                            active_font_size = None
                            if is_pdf and doc:
                                scale = 72 / 300
                                fitz_rect = fitz.Rect(x_min * scale, y_min * scale, x_max * scale, y_max * scale)
                                active_font_size = extract_title_font_size(doc, i, fitz_rect)
                                
                            title_id = len(deferred_titles)
                            deferred_titles.append({
                                "id": title_id,
                                "text": extracted_text,
                                "font_size": active_font_size
                            })
                            
                            # Inject placeholder
                            md_content.append(f"{{{{TITLE_PLACEHOLDER_{title_id}}}}} {extracted_text}\n")
                        else:
                            md_content.append(f"{extracted_text}\n")
                    else:
                        print(f"  [FAILED] Could not derive any usable text from '{cat_name}' element.")
                                
                else:
                    # Treat Tables, Figures, Formulas as images
                    crop_img = page_img[y_min:y_max, x_min:x_max]
                    if crop_img.size == 0:
                        continue
                        
                    # Check if this element was geographically paired with a caption text!
                    # Note: we need to find its pre-sorted index to pull from our mapping
                    original_j = None
                    for orig_idx, orig_elem in image_elements:
                        if (orig_elem[0] == xyxy).all():
                            original_j = orig_idx
                            break
                            
                    paired_caption_text = caption_mapping.get(original_j, None)

                    # Camelot Table Extraction Override
                    table_extracted = False
                    if "table" in cat_norm and is_pdf and doc:
                        pdf_h = doc[i].rect.height
                        pdf_w = doc[i].rect.width
                        md_table = extract_table_camelot(input_path, str(i + 1), x_min, y_min, x_max, y_max, pdf_h, pdf_w)
                        
                        if md_table:
                            print(f"  [EXTRACT] (Camelot Table Parse): Successfully converted HTML table to Native Markdown.")
                            # Append the parsed Markdown table
                            md_content.append(f"{md_table}\n")
                            if paired_caption_text:
                                md_content.append(f"*{paired_caption_text}*\n")
                            table_extracted = True
                            
                    # Fallback to pure Image logic if Camelot failed or it's a Figure/Formula
                    if not table_extracted:
                        # Enforce generic naming scheme logic for tables & figures
                        cat_safe_name = cat_norm.replace(' ', '_')
                        if "table" in cat_norm:
                            cat_safe_name = "table"
                        
                        img_filename = f"page_{i+1}_{cat_safe_name}_{j}.jpg"
                        img_filepath = os.path.join(image_output_dir, img_filename)
                        cv2.imwrite(img_filepath, crop_img)
                        
                        print(f"  [IMAGE DEPLOYED] '{cat_name}' successfully cropped and saved to -> {img_filepath}")
                        
                        # Store rel_path for pristine Markdown links
                        rel_img_filepath = os.path.relpath(img_filepath, os.path.dirname(output_md_path))
                        
                        if paired_caption_text:
                             print(f"  [ATTACHING] Injecting paired caption into Markdown below image.")
                             md_content.append(f"![Image associated with caption: {paired_caption_text}]({rel_img_filepath})")
                             md_content.append(f"\n*{paired_caption_text}*\n")
                        else:
                             md_content.append(f"![{cat_name}]({rel_img_filepath})\n")
                         
        
        # End of Page separator
        md_content.append(f"\n--- End of Page {i+1} ---\n\n")
        print(f"[SUCCESS] Finished writing Page {i+1} layout to Markdown.")
        
        # Stream content to file and clear memory
        with open(output_md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(md_content))
            
        # Register Annotated Page if visualizing
        if args.draw_pdf:
            vis_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            annotated_pages.append(vis_pil)
            
            raw_vis_pil = Image.fromarray(cv2.cvtColor(raw_vis_img, cv2.COLOR_BGR2RGB))
            raw_annotated_pages.append(raw_vis_pil)
        
        del page_img, page_img_rgb, pil_img_rgb, vis_img, raw_vis_img  # Force free memory references to large arrays
        
    if doc:
        doc.close()
        del doc
        import gc
        gc.collect()
        
    # Phase 3: Deferred Two-Pass Title Hierarchy Resolution
    if deferred_titles:
        if args.enable_heading_hierarchy:
            print(f"  [ANALYSIS] Re-evaluating {len(deferred_titles)} titles against global document font scales...")
            with open(output_md_path, "r", encoding="utf-8") as f:
                full_md_text = f.read()
                
            for t in deferred_titles:
                heading_level = infer_heading_level(t["text"], t["font_size"], seen_title_font_sizes)
                heading_level = min(6, heading_level) # Standardize markdown max-depth ceiling
                hashes = "#" * heading_level
                
                # Find and replace the specific placeholder for this title in the markdown stream
                placeholder = f"{{{{TITLE_PLACEHOLDER_{t['id']}}}}}"
                full_md_text = full_md_text.replace(placeholder, hashes)
                
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(full_md_text)
        else:
            print(f"  [ANALYSIS] Heuristic Heading algorithm disabled by config. Defaulting all titles to H2...")
            with open(output_md_path, "r", encoding="utf-8") as f:
                full_md_text = f.read()
            for t in deferred_titles:
                placeholder = f"{{{{TITLE_PLACEHOLDER_{t['id']}}}}}"
                full_md_text = full_md_text.replace(placeholder, "##")
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(full_md_text)
            
    # Compile the Visual Layout Debug PDF
    if args.draw_pdf:
        if annotated_pages:
            vis_pdf_path = output_md_path.replace('.md', '_visual_layout.pdf')
            annotated_pages[0].save(vis_pdf_path, save_all=True, append_images=annotated_pages[1:], resolution=300.0)
            print(f"  [DEBUG] Complete Layout Visualization PDF successfully saved to -> {vis_pdf_path}")
            
        if raw_annotated_pages:
            raw_pdf_path = output_md_path.replace('.md', '_raw_yolo_predictions.pdf')
            raw_annotated_pages[0].save(raw_pdf_path, save_all=True, append_images=raw_annotated_pages[1:], resolution=300.0)
            print(f"  [DEBUG] Raw YOLO Predictions PDF successfully saved to -> {raw_pdf_path}")

    print(f"\n[DONE] Layout parsing complete. Markdown successfully saved to {output_md_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse PDFs/Images into structurally organized Markdown with Layout YOLO.')
    parser.add_argument('target_path', nargs='?', default="pdf_dataset", help='System path to a PDF, Image, or Directory. (Default: "pdf_dataset")')
    parser.add_argument('--no-kmeans', dest='enable_kmeans', action='store_false', default=True, help='Disable the multi-column KMeans algorithm entirely (falls back to vertical sweep). (Default: KMeans Enabled)')
    parser.add_argument('--no-pdf', dest='draw_pdf', action='store_false', default=True, help='Disable the generation of the companion visual layout debug PDF. (Default: PDF Generation Enabled)')
    parser.add_argument('--iou-filter', type=float, default=0.85, help='Threshold [0.0 - 1.0]. Deletes perfectly overlapping bounding boxes of the same type. (Default: 0.85)')
    parser.add_argument('--enable-heading-hierarchy', dest='enable_heading_hierarchy', action='store_true', default=False, help='Toggles the experimental dynamic H1/H2 font-scaling heuristic algorithm. (Default: Disabled)')
    parser.add_argument('--enable-bold-title', dest='enable_bold_title', action='store_true', default=False, help='Toggles the promotion of exclusively bold plain text boxes to semantic titles. (Default: Disabled)')
    
    args = parser.parse_args()
    target_path = args.target_path
    
    # Establish root Output Directory wherever the script is run
    root_output_dir = "output"

    # Define the local model path we expect to find
    local_model_path = "doc_layout_yolo.pt"

    # Pre-Load YOLO model (so it is only loaded once across batches)
    try:
        if os.path.exists(local_model_path):
            print(f"Loading local YOLO model from {local_model_path}...")
            model = YOLOv10(local_model_path)
        else:
            print(f"Local model {local_model_path} not found.")
            print("Downloading/Loading YOLO model from Hugging Face...")
            filepath = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench", 
                filename="doclayout_yolo_docstructbench_imgsz1024.pt"
            )
            model = YOLOv10(filepath)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    if os.path.isdir(target_path):
        print(f"Directory detected. Batch processing all images and PDFs in {target_path}...")
        
        # Collect all valid image/document formats (case-insensitive deduplication)
        valid_extensions = ('.pdf', '.jpg', '.jpeg', '.png')
        all_files_set = set()
        
        for ext in valid_extensions:
            # Add exact matches
            for f in glob.glob(os.path.join(target_path, f"*{ext}")):
                all_files_set.add(os.path.abspath(f))
            # Add uppercase matches (like .PDF)
            for f in glob.glob(os.path.join(target_path, f"*{ext.upper()}")):
                all_files_set.add(os.path.abspath(f))
        
        all_files = sorted(list(all_files_set))
        
        if not all_files:
            print("No valid PDF or Image files found in the directory.")
        else:
            for file_path in all_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                safe_name = sanitize_filename(base_name)
                
                # Each file in the directory gets its own markdown file + own image folder inside the targeted directory
                output_md = os.path.join(root_output_dir, "mds", f"{safe_name}.md")
                output_images = os.path.join(root_output_dir, "images", safe_name)
                
                process_pdf_to_markdown(file_path, output_md, output_images, model, args)
    else:
        # Standard Single File execution
        base_name = os.path.splitext(os.path.basename(target_path))[0]
        safe_name = sanitize_filename(base_name)
        
        output_md = os.path.join(root_output_dir, "mds", f"{safe_name}.md")
        output_images = os.path.join(root_output_dir, "images", safe_name)
        
        process_pdf_to_markdown(target_path, output_md, output_images, model, args)
