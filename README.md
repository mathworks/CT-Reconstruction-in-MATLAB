<p align="center">
  <img src="animation.gif" alt="MRI Volume" width="400" height="300">
</p>


# CBCT Reconstruction in MATLAB
[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=mathworks/CT-Reconstruction-in-MATLAB&file=https://github.com/mathworks/CT-Reconstruction-in-MATLAB/blob/main/CT_Project/startup.m)

## Overview
Welcome to the **CBCT Reconstruction in MATLAB®** repository. This project demonstrates three complementary cone‑beam CT (CBCT) reconstruction approaches implemented in MATLAB:

- FDK (Filtered Backprojection) with automatic GPU utilization  
- Hybrid Iterative Refinement (HIR) for improved data consistency  
- Deep Learning (2D U‑Net) for slice‑wise enhancement  

The workflows are GPU‑aware for fast prototyping and are structured to integrate with CI/CD pipelines for automated validation. The code organization supports both research exploration and production‑oriented deployment.

---

## Reconstruction Methods

### FDK‑Based Reconstruction  
This implementation provides a classical cone‑beam FDK pipeline with robust input handling. Projection stacks are automatically interpreted whether provided as `[nu nv nViews]` or `[nViews nv nu]`. The pipeline dynamically selects GPU or CPU execution and supports batching to fit available memory.

### Hybrid Iterative Refinement (HIR)  
The IR approach starts from an FDK reconstruction and iteratively reduces the forward‑projection residual. A backprojected gradient is combined with a light regularization term, such as Gaussian smoothing or Tikhonov‑like priors. Parameters such as step size, iteration count, and batching can be tuned to balance convergence, stability, and runtime.

### Deep Learning‑Based Enhancement (2D U‑Net)  
A slice‑wise 2D U‑Net is used to enhance FDK reconstructions toward a target quality (e.g., IR or ground truth). The workflow includes data preparation, training/validation splitting, TIFF export, and inference with tiled prediction and blending. Quantitative metrics such as PSNR and SSIM are supported for evaluation.

---

## Highlights

### Rapid Prototyping and Deployment  
The implementation automatically detects CUDA‑capable GPUs and adapts execution accordingly. Key parameters such as batching depth and padding factors allow the workflows to scale across different hardware configurations. Function interfaces are intentionally minimal, enabling easy integration into custom pipelines.

### Automated Testing and Verification  
The repository is designed to be CI/CD‑ready. A lightweight pipeline can be configured to run GPU smoke tests using small datasets, ensuring reproducibility and stability across updates. Deterministic seeds, structured outputs, and logging recommendations support consistent experimentation.

---

## Getting Started

### Prerequisites  
The examples require MATLAB R2026a or newer along with the following products:

- [MATLAB](https://www.mathworks.com/products/matlab.html)&trade;  
- [MATLAB Coder](https://www.mathworks.com/products/matlab-coder.html)&trade;  
- [GPU Coder](https://www.mathworks.com/products/gpu-coder.html)&trade;  
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)&trade;  
- [Fuzzy Logic Toolbox](https://www.mathworks.com/products/fuzzy-logic.html)&trade;  
- [Image Processing Toolbox](https://www.mathworks.com/products/image-processing.html)&trade;  
- [Computer Vision Toolbox](https://www.mathworks.com/products/computer-vision.html)&trade;  
- [Medical Imaging Toolbox](https://www.mathworks.com/products/medical-imaging.html)&trade;  
- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)&trade;  
- [Signal Processing Toolbox](https://www.mathworks.com/products/signal.html)&trade;  
- [System Identification Toolbox](https://www.mathworks.com/products/system-identification.html)&trade;  

### Optional (for IEC 62304 workflows)

- [MATLAB Test](https://www.mathworks.com/products/matlab-test.html)&trade;  
- [Requirements Toolbox](https://www.mathworks.com/products/requirements.html)&trade;  
- [Embedded Coder](https://www.mathworks.com/products/embedded-coder.html)&trade;  
- [IEC Certification Kit](https://www.mathworks.com/products/iec-certification-kit.html)&trade;  

---

## Reference
Wu, M., FitzGerald, P., Zhang, J., Segars, W.P., Yu, H., Xu, Y., and De Man, B., 2022.  
*XCIST—an open access x-ray/CT simulation toolkit*.  
Physics in Medicine & Biology, 67(19), p.194002.

---

## License
The license is available in the `license.txt` file in this repository.  

© 2026 The MathWorks, Inc.
