# inference/

## Contents

| File | Description |
|------|-------------|
| `engine.py` | `InferenceEngine` — production-ready inference wrapper |

## Usage

```python
from inference.engine import InferenceEngine

# Load from checkpoint
engine = InferenceEngine.from_checkpoint("./checkpoints/classifier/best_model.pth")

# Single image
label, conf = engine.predict("path/to/image.jpg")
print(f"{label} (Confidence: {conf:.2f})")

# PIL Image
from PIL import Image
img = Image.open("image.jpg")
label, conf = engine.predict(img)

# NumPy array
import numpy as np
arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
label, conf = engine.predict(arr)

# Batch of paths
results = engine.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# Full directory
results = engine.predict_directory("./test_folder/")
for r in results:
    print(f"{r['label']:4s} ({r['confidence']:.2f})  {r['path']}")
```

## InferenceEngine API

```
from_checkpoint(path, vit_model, image_size, threshold, device) → InferenceEngine
predict(img_input, return_features)    → (label, confidence[, features])
predict_batch(img_inputs, batch_size)  → [(label, confidence), ...]
predict_directory(dir_path, ...)       → [{"path", "label", "confidence"}, ...]
```
