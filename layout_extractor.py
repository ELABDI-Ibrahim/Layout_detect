import cv2
import fitz 
from PIL import Image, ImageFont
import numpy as np
import os

# Monkey-patch PIL for layoutparser (which uses deprecated getsize)
if not hasattr(ImageFont.FreeTypeFont, 'getsize'):
    def getsize(self, text, *args, **kwargs):
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)
        return right - left, bottom - top
    ImageFont.FreeTypeFont.getsize = getsize

import layoutparser as lp

# ----------------------------
# Load EfficientDet Layout Model
# ----------------------------

model = lp.models.effdet.layoutmodel.EfficientDetLayoutModel(
    "tf_efficientdet_d0",   # model architecture name
    model_path="publaynet-tf_efficientdet_d0.pth.tar",
    label_map={1: "Text", 2: "Title", 3: "List", 4: "Table", 5: "Figure"},
    device="cpu",  # or "cuda" if you have a GPU
    extra_config={"output_confidence_threshold": 0.1}
)

# ----------------------------
# Convert PDF to Images using PyMuPDF (fitz)
# ----------------------------
pdf_path = "document.pdf"
pages = []

if os.path.exists(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        # Get a pixmap from the page at 300 DPI
        pix = page.get_pixmap(dpi=300)
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
else:
    print(f"File not found: {pdf_path}")

# ----------------------------
# Process each page
# ----------------------------
for i, page in enumerate(pages):

    # Convert PIL image → OpenCV
    image = np.array(page)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect layout
    layout = model.detect(image)

    print(f"\nPage {i+1} Layout:")
    for block in layout:
        print(block)

    # Draw boxes
    vis_image = lp.draw_box(
        image,
        layout,
        box_width=3,
        show_element_type=True
    )

    # Save visualization
    os.makedirs("output", exist_ok=True)
    output_path = f"output/page_{i+1}_layout.jpg"
    vis_image.save(output_path)

print("Layout parsing complete.")