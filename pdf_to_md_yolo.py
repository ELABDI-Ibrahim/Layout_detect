import os
import sys
import glob
import cv2
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

import re

# ... Keep existing imports ...

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

def detect_ocr_language(sample_text):
    if not sample_text or len(sample_text.strip()) < 10:
        return 'eng' # Default if not enough text to detect confidently
    try:
        iso_code = detect(sample_text)
        return LANG_MAP.get(iso_code, 'eng') # Fallback to English if unknown mapping
    except:
        return 'eng'

def sort_elements_multicolumn(elements, page_width):
    if not elements:
        return []
        
    # 1. Identify wide elements (>60% page width) to serve as column breakers
    wide_elements_idx = []
    for idx, (xyxy, name) in enumerate(elements):
        w = xyxy[2] - xyxy[0]
        if w > 0.6 * page_width:
            wide_elements_idx.append(idx)
            
    wide_elements_idx.sort(key=lambda i: elements[i][0][1])

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
    
    band_queue_idx = 0
    wide_queue_idx = 0
    
    # 3. Process the bands using KMeans horizontally
    while band_queue_idx < len(bands):
        b_top, b_bottom = bands[band_queue_idx]
        band_elements = []
        band_original_indices = []
        
        for idx, (xyxy, _) in enumerate(elements):
            if idx in placed_indices: continue
            y_center = (xyxy[1] + xyxy[3]) / 2
            # Handle float('inf') math gracefully
            bottom_val = b_bottom if b_bottom != float('inf') else float('inf')
            
            if b_top <= y_center < bottom_val:
                band_elements.append(elements[idx])
                band_original_indices.append(idx)
                placed_indices.add(idx)
                
        if band_elements:
            x_centers = np.array([get_center(e[0])[0] for e in band_elements]).reshape(-1, 1)
            n_cols = 2 if len(band_elements) >= 2 else 1
            print(f"  [KMEANS] Scanning Band {band_queue_idx + 1} ({len(band_elements)} elements) -> Grouping into {n_cols} columns.")
            
            try:
                # Use cluster center sorting to determine left-vs-right columns dynamically
                kmeans = KMeans(n_clusters=n_cols, random_state=0, n_init='auto').fit(x_centers)
                col_assignment = kmeans.labels_
                col_order = np.argsort(kmeans.cluster_centers_.flatten())
                
                col_groups = {i: [] for i in range(n_cols)}
                for i, col_idx in enumerate(col_assignment):
                    col_groups[col_idx].append(band_elements[i])
                    
                for col_idx in col_groups:
                    col_groups[col_idx].sort(key=lambda x: x[0][1]) 
                    
                for col_idx in col_order:
                    sorted_output.extend(col_groups[col_idx])
                    
            except Exception as e:
                print(f"  [WARN] KMeans failed on this band ({e}). Defaulting to standard Y sort.")
                band_elements.sort(key=lambda x: (x[0][1] // 20, x[0][0]))
                sorted_output.extend(band_elements)

        band_queue_idx += 1
        
        # Append the wide element separator back into the flow
        if wide_queue_idx < len(wide_bounds):
            wide_idx = wide_bounds[wide_queue_idx][0]
            sorted_output.append(elements[wide_idx])
            wide_queue_idx += 1
            
    # Sweep stragglers strictly by reading order
    stragglers = [elements[i] for i in range(len(elements)) if i not in placed_indices]
    if stragglers:
        print(f"  [KMEANS] Sweeping up {len(stragglers)} stray blocks via standard layout fallback.")
        stragglers.sort(key=lambda x: (x[0][1] // 20, x[0][0]))
        sorted_output.extend(stragglers)
        
    return sorted_output

# ... Inside process_pdf_to_markdown ...
def process_pdf_to_markdown(input_path, output_md_path, image_output_dir, model):
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
                
            print(f"[LAYOUT] Detected {len(elements)} structural elements on Page {i+1}:")
            for _, cat_name in elements:
                print(f"  - {cat_name}")
                
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
            
            # Phase 2: KMeans Multi-Column Processing Pipeline
            print("  [ANALYSIS] Calculating document reading flow (KMeans Multi-Column)...")
            elements = sort_elements_multicolumn(elements, page_width)
            
            for j, (xyxy, cat_name) in enumerate(elements):
                x_min, y_min, x_max, y_max = map(int, xyxy)
                
                # Ensure the crop coordinates are within image boundaries (300 DPI image scale)
                h, w = page_img.shape[:2]
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)
                
                cat_norm = cat_name.lower().replace("_", " ")
                
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
                                extracted_text = pdf_text
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
                            md_content.append(f"## {extracted_text}\n")
                        else:
                            md_content.append(f"{extracted_text}\n")
                    else:
                        print(f"  [FAILED] Could not derive any usable text from '{cat_name}' element.")
                                
                else:
                    # Treat Tables, Figures, Formulas as images
                    crop_img = page_img[y_min:y_max, x_min:x_max]
                    if crop_img.size == 0:
                        continue
                        
                    # Enforce generic naming scheme logic for tables & figures
                    cat_safe_name = cat_norm.replace(' ', '_')
                    if "table" in cat_norm:
                        cat_safe_name = "table"
                    
                    img_filename = f"page_{i+1}_{cat_safe_name}_{j}.jpg"
                    img_filepath = os.path.join(image_output_dir, img_filename)
                    cv2.imwrite(img_filepath, crop_img)
                    
                    print(f"  [IMAGE DEPLOYED] '{cat_name}' successfully cropped and saved to -> {img_filepath}")
                    
                    # Check if this image was geographically paired with a caption text!
                    # Note: we need to find its pre-sorted index to pull from our mapping
                    original_j = None
                    for orig_idx, orig_elem in image_elements:
                        if (orig_elem[0] == xyxy).all():
                            original_j = orig_idx
                            break
                            
                    paired_caption_text = caption_mapping.get(original_j, None)
                    
                    # Store rel_path for pristine Markdown links, not the absolute system path.
                    rel_img_filepath = os.path.relpath(img_filepath, os.path.dirname(output_md_path))
                    
                    if paired_caption_text:
                         print(f"  [ATTACHING] Injecting paired caption into Markdown below image.")
                         md_content.append(f"![Image associated with caption: {paired_caption_text}]({rel_img_filepath})")
                         # Append the caption text openly in Markdown directly underneath!
                         md_content.append(f"\n*{paired_caption_text}*\n")
                    else:
                         md_content.append(f"![{cat_name}]({rel_img_filepath})\n")
                         
        
        # End of Page separator
        md_content.append(f"\n--- End of Page {i+1} ---\n\n")
        print(f"[SUCCESS] Finished writing Page {i+1} layout to Markdown.")
        
        # Stream content to file and clear memory
        with open(output_md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(md_content))
            
        del page_img, page_img_rgb, pil_img_rgb  # Force free memory references to large arrays
        
    if doc:
        doc.close()

    print(f"\n[DONE] Layout parsing complete. Markdown successfully saved to {output_md_path}\n")

if __name__ == "__main__":
    target_path = "pdf_dataset"
    
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
                
                process_pdf_to_markdown(file_path, output_md, output_images, model)
    else:
        # Standard Single File execution
        base_name = os.path.splitext(os.path.basename(target_path))[0]
        safe_name = sanitize_filename(base_name)
        
        output_md = os.path.join(root_output_dir, "mds", f"{safe_name}.md")
        output_images = os.path.join(root_output_dir, "images", safe_name)
        
        process_pdf_to_markdown(target_path, output_md, output_images, model)
