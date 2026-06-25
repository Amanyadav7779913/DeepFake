# visualization/

## gradcam.py

Visual explainability tools:

| Class | Description |
|-------|-------------|
| `ViTGradCAM` | Gradient-weighted class activation maps for ViT |
| `AttentionRollout` | DINO-style cumulative self-attention maps |
| `VisualisationPipeline` | 4-panel combined figure |

## Usage

```bash
# Via predict.py
python predict.py --img test.jpg --visualize

# Direct
from visualization.gradcam import VisualisationPipeline
pipeline = VisualisationPipeline(model)
fig = pipeline.visualise("image.jpg", rgb, fft, label="Fake", confidence=0.92)
fig.savefig("output.png")
```
