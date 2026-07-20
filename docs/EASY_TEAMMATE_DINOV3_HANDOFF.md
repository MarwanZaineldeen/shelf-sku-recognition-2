# Ultra-Simple Teammate Guide: DINOv3 Hand-Off Instructions

## Executive Summary

To make integration as effortless as possible, **your teammate does NOT need to learn SQLite, database schemas, or FAISS indexing**. 

We (the AI Lead Team) will handle **100% of the SQLite database construction, vector blob conversion, and index building automatically**.

---

## What We Need from Your Teammate (Only 2 Simple Things!)

Your teammate only needs to provide **2 simple items**:

### Item #1: The DINOv3 Model Loader / Weights
Provide either:
- **Option A (PyTorch Weight File)**: A `.pt` or `.safetensors` model weight file placed in `configs/weights/dinov3_vitb16_exemplar.pt`.
- **Option B (Python Snippet or HuggingFace ID)**: A 5-line python code snippet showing how to load the DINOv3 model and extract $L_2$-normalized 768-D features for PIL images:

```python
import torch
from PIL import Image

# 1. How to load your DINOv3 model
model = load_dinov3_model("path/to/weights.pt")
model.eval()

# 2. How to extract 768-D L2-normalized features for a list of PIL images
def extract_dinov3_features(pil_images: list[Image.Image]) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(images=pil_images, return_tensors="pt")
        features = model(**inputs)
        # Ensure L2 normalization
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features  # Shape: (N, 768)
```

---

### Item #2: The Training Embeddings or FAISS Index File ⭐
Provide any ONE of the following (whichever is easiest for him!):

- **Option A (FAISS Index File - SUPER EASY ⭐)**: If he already built a FAISS index, he can simply share:
  1. His FAISS index file (e.g. `dinov3_exemplar.index` or `index.faiss`).
  2. His crop ID list file (e.g. `crop_ids.json` or `image_list.txt`) mapping FAISS row numbers to crop filenames.
  - *We will automatically read his FAISS file and link it to our SQLite database in 5 seconds!*

- **Option B (Raw `.npy` or `.pkl` File)**: A single NumPy array or pickle dictionary file containing the 768-D training vectors:
  ```python
  # Simple dictionary format:
  {
      "crop_00001.jpg": array_768d,
      "crop_00002.jpg": array_768d,
      ...
  }
  ```

- **Option C (Let Us Extract Features Automatically)**: If he only shares Item #1 (DINOv3 model weights), **we will run an automated script (`scripts/build_dinov3_sqlite_registry.py`) that extracts 768-D features for all 31,656 training crops automatically**!

---

## What We (AI Lead Team) Will Do Automatically

Once your teammate gives us Item #1 and/or Item #2:

1. **Automated SQLite Building**: We will automatically generate `retail_sku_registry_dinov3.db` with all 768-D vector blobs and 67 commercial product metadata records.
2. **Automated Model Wrapper**: We will write `ml/embeddings/dinov3.py` implementing our `BaseEmbedder` interface.
3. **Automated Search Index**: We will configure `NumpyCosineIndex` to load the 768-D DINOv3 SQLite database at server startup.

---

## Summary Checklist for Teammate

- [ ] Share PyTorch model weights or HuggingFace model ID for DINOv3 (ViT-B/16 Exemplar).
- [ ] Share a 5-line Python snippet showing PIL image pre-processing & forward pass.
- [ ] *(Optional)* Share a `.npy` or `.pkl` file of training embeddings, OR let us run feature extraction automatically.
