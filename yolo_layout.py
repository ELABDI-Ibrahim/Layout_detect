import os
import sys
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

# Load the pre-trained model from Hugging Face
# filepath = hf_hub_download(
#     repo_id="juliozhao/DocLayout-YOLO-DocStructBench", 
#     filename="doclayout_yolo_docstructbench_imgsz1024.pt"
# )

filepath = "doc_layout_yolo.pt"
model = YOLOv10(filepath)

# Allow passing the file path (PDF or Image) as a command line argument
input_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"
pages = []

if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    sys.exit(1)

if input_path.lower().endswith('.pdf'):
    print(f"Loading PDF: {input_path}")
    # Convert PDF to Images using PyMuPDF (fitz)
    doc = fitz.open(input_path)
    for page in doc:
        # Get a pixmap from the page at 300 DPI
        pix = page.get_pixmap(dpi=300)
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
else:
    print(f"Loading Image: {input_path}")
    img = Image.open(input_path).convert("RGB")
    pages.append(img)

# ----------------------------
# Process each page
# ----------------------------
os.makedirs("output_yolo", exist_ok=True)

for i, page in enumerate(pages):
    print(f"\nProcessing Page {i+1}...")
    
    # YOLO prediction expects an image format (Path, PIL Image, or numpy array)
    det_res = model.predict(
        page,         # Image to predict
        imgsz=1024,   # Prediction image size
        conf=0.2,     # Confidence threshold
        device="cpu"  # Device to use (e.g., 'cuda:0' or 'cpu')
    )
    
    # Annotate and save the result
    if len(det_res) > 0:
        # Get bounding boxes and class names
        boxes = det_res[0].boxes
        names = det_res[0].names
        print(f"Detected {len(boxes)} elements.")
        
        # YOLO's built-in plotting function adds boxes & labels automatically
        annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
        
        output_path = f"output_yolo/page_{i+1}_layout.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved annotated image to {output_path}")
    else:
        print(f"No results returned for Page {i+1}.")

print("\nLayout parsing complete.")