## Overview

We implement the  **SSR** (Single-view Neural Implicit Shape and Radiance Fields), a novel framework for the simultaneous high-fidelity recovery of both object shapes and textures from single-view images. By utilizing neural implicit representations, SSR combines **shape and radiance fields** to generate realistic 3D models from partial, single-view observations. This approach incorporates **explicit 3D shape supervision** and integrates **volume rendering** techniques for color, depth, and surface normal images to enhance both shape and texture quality.

Key features include:

- **High-fidelity Shape and Texture Recovery**: This framework recovers detailed 3D shapes and realistic textures, even from a single-view image.

- **Shape-Appearance Ambiguity Resolution**: This addresses the challenges of shape-appearance ambiguity typically encountered with partial observations using a two-stage learning curriculum that leverages both 3D and 2D supervision.

- **Fine-Grained Textured Meshes**: This approach produces **fine-grained, textured 3D meshes** that preserve intricate surface details.

## Usage
### Training

We can train the model using 
```
python train.py --config config.yaml
```

Any changes to the architectural design can be either addressed using the specification in the `config.yaml`

More low level changes to the model can be done inside the `model` module


### Inference
We can perform the inference using the `inference.py` giving the input to the model and then running the inference on that.