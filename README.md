## CBCT Reconstruction in MATLAB

Overview
Welcome to the CBCT Reconstruction in MATLAB® repository! This project showcases three complementary cone‑beam CT (CBCT) reconstruction approaches using MATLAB:

- FDK (Filtered Backprojection) with automatic GPU utilization
- Hybrid Iterative Refinement (IR) that enforces data‑consistency with light regularization
- Deep Learning (2D U‑Net) slice‑wise enhancement from FDK to target quality


All examples are written to be GPU‑aware for rapid prototyping and can integrate with CI/CD pipelines for automated testing and verification*. Code is structured for easy extension and production‑grade deployment.

*Additional products/tooling may be required for regulated environments.

# Features
1. **FDK‑Based Reconstruction**

Description: Classical cone‑beam FDK reconstruction with robust input handling. Accepts projection stacks in [nu nv nViews] or [nViews nv nu] form and auto‑permutes. Supports auto GPU/CPU selection and batching to fit device memory.

2. **Hybrid Iterative Refinement (IR)**

Description: Starts from an FDK volume and iteratively reduces the forward‑projection residual using a backprojected gradient and a light Gaussian/Tikhonov‑like prior. Tunable step size, iterations, forward projector sampling, and batching for stability and speed.

3. **Deep Learning‑Based Enhancement (2D U‑Net)**

Description: Trains a 2D U‑Net per slice to map FDK → target (e.g., higher‑quality IR or ground truth). Handles train/val selection, TIFF export, patch or full‑slice training, tiled inference with Hann blending, and PSNR/SSIM reporting.
## Highlights
**Rapid Prototyping and Deployment**

GPU‑Enabled: Auto‑detects CUDA‑capable GPUs via gpuDeviceCount. Key parameters (BatchZ, PadFactor, BatchPixels) allow scaling to your hardware.
Ease of Use: Minimal function signatures with sensible defaults; robust variable auto‑detection from .mat files.

**Automated Testing and Verification**

CI/CD Ready: Includes an optional GitLab CI snippet for GPU smoke tests*. Add small sample inputs to validate pipeline integrity on every commit.
Reproducibility: Deterministic seeds for DL, structured output folders, and explicit option logging recommended.

* Requires a GPU‑enabled runner and MATLAB availability on the runner (or Docker with MATLAB).

# Getting Started
**Prerequisites**
You will need MATLAB release R2026a or newer and the following MathWorks Products to run these examples.


• [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)&trade;

• [MATLAB Coder](https://www.mathworks.com/products/matlab-coder.html)&trade;

• [GPU Coder](https://www.mathworks.com/products/gpu-coder.html)&trade;

• [Medical Imaging Toolbox](https://www.mathworks.com/products/medical-imaging.html)&trade;

• [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)&trade;

• [Image Processing Toolbox](https://www.mathworks.com/products/image-processing.html)&trade;


• [Computer Vision Toolbox](https://www.mathworks.com/products/computer-vision.html)&trade;

### Reference

[1] Wu, M., FitzGerald, P., Zhang, J., Segars, W.P., Yu, H., Xu, Y. and De Man, B., 2022. XCIST—an open access x-ray/CT simulation toolkit. Physics in Medicine & Biology, 67(19), p.194002.

### License
The license is available in license.txt file in this GitHub repository.

Copyright 2026 The MathWorks, Inc.

