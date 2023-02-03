# What is LaMa?
- [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/saic-mdal/lama)

# Directory Structure

# Model Architecture
- Summary
  ```
  (0): ReflectionPad2d
  (1 ~ 4): FFC_BN_ACT: Downsample
    - 4: 1 + 3 (= n_downsampling)
  (5 ~ 22): FFCResnetBlock (Fast Fourier Conv Residual Block)
    - 18 (= n_blocks)
  (23): ConcatTupleLayer : Upsample
  (24 ~ 26): ConvTranspose2d -> BatchNorm2d -> ReLU
  (27 ~ 29): ConvTranspose2d -> BatchNorm2d -> ReLU
  (30 ~ 32): ConvTranspose2d -> BatchNorm2d -> ReLU
  (33): ReflectionPad2d
  (34): Conv2d
  (35): Sigmoid
  ```
  - `FourierUnit`
  ```
  (conv_layer): Conv2d
  (bn): BatchNorm2d
  (relu): ReLU
  ```
  - `SpectralTransform`
  ```
  (downsample): Identity
  (conv1): Sequential(
    (0): Conv2d
    (1): BatchNorm2d
    (2): ReLU
  )
  (fu): FourierUnit
  (conv2): Conv2d
  ```
  - `FFC_BN_ACT`
  ```
  (ffc): FFC(
    (convl2l): Conv2d
    (convl2g): Conv2d
    (convg2l): Conv2d
    (convg2g): SpectralTransform
    (gate): Identity
    (bn_l): BatchNorm2d
    (bn_g): BatchNorm2d
  (act_l): ReLU
  (act_g): ReLU
  ```
  - `FFCResnetBLock`
    ```
    (conv1): FFC_BN_ACT
    (conv2): FFC_BN_ACT
    ```

# Paper Summary
- Paper: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/pdf/2109.07161.pdf)
## Image Inpainting
- The inpainting problem is inherently ambiguous. There could be many plausible fillings for the same missing areas,
especially when the “holes” become wider.
## Train
- The usual practice is to train inpainting systems on a large automatically generated dataset, created by randomly
masking real images.
- The training is performed on a dataset of (image, mask) pairs obtained from real images and synthetically generated masks.
- Big LaMa uses a larger batch size of 120 (instead of 30 for our other models). Although we consider this model relatively large, it is still smaller than some of the baselines. It was trained on eight NVidia V100 GPUs for approximately 240 hours.
## Dataset
- The training dataset; The model was trained on a subset of 4.5M images from Places-Challenge dataset. Just as our standard base model, the Big LaMa was trained only on low-resolution 256 × 256 crops of approximately 512 × 512 images.
## Characteristics
- It’s common to use complicated two-stage models with intermediate predictions, such as smoothed images, edges, and segmentation maps. In this work, we achieve state-of-the-art results with a simple single-stage network.
- Generalizes surprisingly well to resolutions that are higher than those seen at train time.
- Can generalize to high-resolution images after training only on low-resolution data.
- Can capture and generate complex periodic structures, and is robust to large masks.
- Significantly less trainable parameters and inference time costs.
### Large Effective Receptive Field
- Essential for understanding the global structure of an image.
- In the case of a large mask, an even large yet limited receptive field may not be enough to access information necessary for generating a quality inpainting.
- Popular convolutional architectures might lack a sufficiently large effective receptive field.
### large training masks

## Method
- Our goal is to inpaint a color image 'x' masked by a binary mask of unknown pixels 'm'. The mask 'm' is stacked with the masked image, resulting in a four-channel input tensor 'x'.
  - 'lama'>'saicinpainting'>'evaluation'>'refinement.py'>`_infer` (121 ~ 122):
    ```python
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)
    ```
  - 'lama'>'saicinpainting'>'training'>'trainers'>'default.py'>`DefaultInpaintingTrainingModule`>`forward`:
    ```python
    masked_img = img * (1 - mask)
    ...
    if self.concat_mask:
        masked_img = torch.cat([masked_img, mask], dim=1)
    ```
## Architecture
- Processes the input in a fully-convolutional manner.
- the generation of proper inpainting requires to consider global context. Thus, we argue that a good architecture should have units with as wide-as-possible receptive field as early as possible in the pipeline.
### ResNet
- The conventional fully convolutional models, e.g. ResNet, suffer from slow growth of effective receptive field.
- Receptive field might be insufficient, especially in the early layers of the network, due to the typically small (e.g. 3 × 3) convolutional kernels. Thus, many layers in the network will be lacking global context and will waste computations and parameters to create one.
- For wide masks, the whole receptive field of a generator at the specific position may be inside the mask, thus observing only missing pixels. The issue becomes especially pronounced for high-resolution images.
### Fast Fourier Convolutions (FCCs)
- Have image-wide receptive field
- Allow for a receptive field that covers an entire image even in the early layers of the network.
- The inductive bias of FFC allows the network to generalize to high resolutions that are never seen during training.
- Fast Fourier convolution (FFC) is the recently proposed operator that allows to use global context in early layers.
- FFC is based on a channel-wise fast Fourier transform (FFT) and has a receptive field that covers the entire image.
- FFC splits channels into two parallel branches: i) local branch uses conventional convolutions, and ii) global branch uses real FFT to account for global context. Real FFT can be applied only to real valued signals, and inverse real FFT ensures that the output is real valued. Real FFT uses only half of the spectrum compared to the FFT.
- the outputs of the local (i) and global (ii) branches are fused together
- FFCs are fully differentiable and easy-to-use drop-in replacement for conventional convolutions.
- Due to the image-wide receptive field, FFCs allow the generator to account for the global context starting from
the early layers, which is crucial for high-resolution image inpainting.
## Loss
- a multi-component loss that combines adversarial loss and a high receptive field perceptual loss
```python
"""l1 loss on src pixels, and downscaled predictions if on_pred=True"""
loss = torch.mean(torch.abs(pred[mask<1e-8] - image[mask<1e-8]))
if on_pred: 
    loss += torch.mean(torch.abs(pred_downscaled[mask_downscaled>=1e-8] - ref[mask_downscaled>=1e-8]))  
```
### High Receptive Field Perceptual Loss (HRF PL)
- Naive supervised losses require the generator to reconstruct the ground truth precisely. However, the visible parts of the image often do not contain enough information for the exact reconstruction of the masked part. Therefore, using naive supervision leads to blurry results due to the averaging of multiple plausible modes of the inpainted content.
## Big LaMa-Fourier
- Big LaMa-Fourier differs from LaMa-Fourier in three aspects:
  - The depth of the generator; It has 18 residual blocks, all based on FFC, resulting in 51M parameters.


# Fast Fourier Convolution
- Reference: https://medium.com/mlearning-ai/fast-fourier-convolution-a-detailed-view-a5149aae36c4
- The idea is to replace the convolution layer (Conv2D) with the FFC block.
- *FFC block consists of 2 paths — local and global. The local path uses ordinary convolution operators on the input feature maps and the global path operates in the spectral domain.*
- FFC
  - <img src="https://miro.medium.com/v2/resize:fit:1208/format:webp/1*yb9nibDAneeAjVTEOAH8qw.png" width="400">
