"""
Processor — Simple model template for reference.

This is the original processor.py kept as a template showing the
simplest way to wrap a PyTorch model. For the experiment framework,
use the adapter pattern in src/models/ instead.

Example of the pattern this has been replaced by:
    src/core/interfaces.py     → BaseModelAdapter (ABC)
    src/models/gemma_vlm.py    → GemmaVLMAdapter (concrete)
    src/models/base.py         → detect_device(), build_quantization_config()
    tests/conftest.py          → MockModelAdapter (for testing)

Usage (legacy):
    model = VideoModel("tiny_stub")
    output = model.predict(torch.randn(1, 3, 64, 64))
    # output.shape == (1, 10)
"""

import torch
import torch.nn as nn


class VideoModel:
    """Minimal model template — replaced by BaseModelAdapter in the framework.

    This class is kept as a reference showing the simplest possible
    model wrapper. To add a new model to the experiment framework:

    1. Subclass BaseModelAdapter
    2. Implement capabilities, load(), predict()
    3. Register with @MODEL_REGISTRY.register("your_model")
    4. Add to configs/
    """

    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 10)
        ).to(self.device)

    def predict(self, x):
        with torch.no_grad():
            return self.model(x.to(self.device))