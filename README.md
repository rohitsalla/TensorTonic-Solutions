# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# rohit salla's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/sallarohit1.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Adaptive Average Pool 2D | Implement adaptive two-dimensional average pooling with variable, possibly overlapping regions for arbitrary output sizes. | https://www.tensortonic.com/study-plans/cracking-cv/cv-adaptive-avg-pool-2d |
| Anchor Box Generation | Generate axis-aligned anchor boxes for a feature map of shape (feature h, feature w) at stride s. | https://www.tensortonic.com/study-plans/cracking-cv/cv-anchor-generation |
| Average Pool 2D | Apply two-dimensional average pooling to an image with the requested kernel and stride, without padding. | https://www.tensortonic.com/study-plans/cracking-cv/cv-avg-pool-2d |
| Bilinear Resize | Resize a grayscale image with corner-aligned bilinear interpolation matching PyTorch interpolation semantics. | https://www.tensortonic.com/study-plans/cracking-cv/cv-bilinear-resize |
| Box Encode and Decode | Implement Faster-RCNN style box regression target encoding and decoding, using two arrays of axis-aligned boxes in xyxy format. | https://www.tensortonic.com/study-plans/cracking-cv/cv-box-encode-decode |
| Per-Channel Mean and Std | Compute population means and standard deviations for every channel across a batch of images in NHWC layout. | https://www.tensortonic.com/study-plans/cracking-cv/cv-channel-statistics |
| Multichannel 2D Convolution | Implement multichannel two-dimensional cross-correlation with output filters, optional bias, stride, and padding. | https://www.tensortonic.com/study-plans/cracking-cv/cv-conv2d-multichannel |
| 2D Transposed Convolution | Implement two-dimensional transposed convolution matching PyTorch output layout, stride, padding, and optional bias. | https://www.tensortonic.com/study-plans/cracking-cv/cv-conv2d-transpose |
| Depthwise Separable Convolution | Factor image convolution into per-channel spatial filters and a pointwise convolution that mixes output channels. | https://www.tensortonic.com/study-plans/cracking-cv/cv-depthwise-separable-conv |
| Dilated 2D Convolution | Implement a 2D dilated cross-correlation (no kernel flip) on a single multi-channel image. | https://www.tensortonic.com/study-plans/cracking-cv/cv-dilated-conv2d |
| Gaussian Blur 2D | Build a normalized separable Gaussian kernel and apply it to a grayscale image with the specified padding behavior. | https://www.tensortonic.com/study-plans/cracking-cv/cv-gaussian-blur |
| Image Normalize | Normalize each image channel by its supplied mean and standard deviation to produce standardized vision-model inputs. | https://www.tensortonic.com/study-plans/cracking-cv/cv-image-normalize |
| Max Pool 2D | Apply two-dimensional max pooling to an image with a configurable window and stride, without padding. | https://www.tensortonic.com/study-plans/cracking-cv/cv-max-pool-2d |
| Non-Maximum Suppression | Implement greedy non-maximum suppression matching torchvision selection order and strict IoU threshold behavior. | https://www.tensortonic.com/study-plans/cracking-cv/cv-nms |
| RGB to Grayscale | Convert an RGB image to grayscale with luminance-weighted color channels for classical computer vision preprocessing. | https://www.tensortonic.com/study-plans/cracking-cv/cv-rgb-to-grayscale |
| Sobel Edge Detection | Apply horizontal and vertical Sobel filters to a grayscale image with one-pixel zero padding and return edge gradients. | https://www.tensortonic.com/study-plans/cracking-cv/cv-sobel-edges |
| Soft Non-Maximum Suppression | Apply Gaussian Soft-NMS by repeatedly selecting the strongest box and decaying overlapping box scores by IoU. | https://www.tensortonic.com/study-plans/cracking-cv/cv-soft-nms |
| Zero Pad and Center Crop | Zero-pad a two-dimensional grayscale image on every side, then extract a centered crop with the requested dimensions. | https://www.tensortonic.com/study-plans/cracking-cv/cv-zero-pad-and-center-crop |
| Activation Functions | Implement ReLU, sigmoid, tanh, Leaky ReLU, GELU, and Swish with their analytical derivatives. | https://www.tensortonic.com/study-plans/cracking-dl/dl-activation-functions |
| Autoencoder Forward Pass | Implement a NumPy autoencoder forward pass through an encoder bottleneck and decoder reconstruction network. | https://www.tensortonic.com/study-plans/cracking-dl/dl-autoencoder |
| Computational Graph & Autograd | Build a minimal autograd engine that performs forward and backward passes on a computational graph. | https://www.tensortonic.com/study-plans/cracking-dl/dl-autograd |
| Backpropagation (Single Hidden Layer) | Backpropagate MSE loss through a one-hidden-layer NumPy network to compute gradients for every weight and bias. | https://www.tensortonic.com/study-plans/cracking-dl/dl-backpropagation |
| Batch Normalization | Implement batch normalization for training and inference, including batch statistics and running-statistic updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-batch-normalization |
| 2D Convolution from Scratch | Implement multichannel two-dimensional cross-correlation from scratch with multiple filters, stride, padding, and bias. | https://www.tensortonic.com/study-plans/cracking-dl/dl-convolution-operation |
| Transformer Decoder Block | Implement a NumPy Transformer decoder block with masked self-attention, cross-attention, and feed-forward layers. | https://www.tensortonic.com/study-plans/cracking-dl/dl-decoder-block |
| Depthwise Separable Convolution | Implement NumPy depthwise spatial convolution followed by pointwise channel mixing and optional bias. | https://www.tensortonic.com/study-plans/cracking-dl/dl-depthwise-separable-convolution |
| Dropout | Apply inverted dropout from a supplied binary mask during training and preserve the input unchanged during evaluation. | https://www.tensortonic.com/study-plans/cracking-dl/dl-dropout |
| Transformer Encoder Block | Implement a NumPy Transformer encoder block with self-attention, residual connections, normalization, and feed-forward layers. | https://www.tensortonic.com/study-plans/cracking-dl/dl-encoder-block |
| Multi-Layer Perceptron (Forward Pass) | Implement the forward pass of a multi-layer perceptron (MLP) with arbitrary depth and width. | https://www.tensortonic.com/study-plans/cracking-dl/dl-forward-pass |
| GAN Training Step | Implement a single forward-pass training step for a Generative Adversarial Network (GAN). | https://www.tensortonic.com/study-plans/cracking-dl/dl-gan-training |
| GRU Cell | Implement a Gated Recurrent Unit (GRU) cell that processes a sequence using two gates: an update gate and a reset gate. | https://www.tensortonic.com/study-plans/cracking-dl/dl-gru-cell |
| Inception Module | Implement a simplified Inception module with two parallel convolution branches whose outputs are concatenated along the channel dimension. | https://www.tensortonic.com/study-plans/cracking-dl/dl-inception-module |
| Layer Normalization | Implement Layer Normalization (Ba et al, 2016), the standard normalization technique in Transformers. | https://www.tensortonic.com/study-plans/cracking-dl/dl-layer-normalization |
| LeNet Forward Pass | Implement the forward pass of a simplified LeNet-style convolutional neural network using only NumPy. | https://www.tensortonic.com/study-plans/cracking-dl/dl-lenet-5 |
| Loss Functions | Implement MSE, binary cross-entropy, categorical cross-entropy, and Huber losses from supplied predictions and targets. | https://www.tensortonic.com/study-plans/cracking-dl/dl-loss-functions |
| LSTM Cell | Implement a Long Short-Term Memory (LSTM) cell that processes a sequence using four gates: forget, input, output, and a candidate cell state. | https://www.tensortonic.com/study-plans/cracking-dl/dl-lstm-cell |
| Mini-Batch Training Loop | Train a NumPy multilayer perceptron over ordered mini-batches with forward passes, backpropagation, and SGD updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-mini-batch-training |
| Multi-Head Attention | Implement NumPy multi-head attention with QKV projections, optional masking, stable softmax, and output projection. | https://www.tensortonic.com/study-plans/cracking-dl/dl-multi-head-attention |
| Non-Maximum Suppression | Implement greedy non-maximum suppression for scored object-detection boxes using an IoU threshold and stable selection order. | https://www.tensortonic.com/study-plans/cracking-dl/dl-nms |
| Perceptron | Train a binary perceptron from zero-initialized weights using ordered samples, step predictions, and error-correction updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-perceptron |
| Pooling Layers | Implement channelwise two-dimensional max and average pooling with configurable kernel and stride. | https://www.tensortonic.com/study-plans/cracking-dl/dl-pooling-layers |
| Rotary Position Embedding (RoPE) | Implement Rotary Position Embedding (RoPE), which encodes position by rotating pairs of dimensions in the embedding vector. | https://www.tensortonic.com/study-plans/cracking-dl/dl-positional-encoding |
| ResNet Residual Block | Build a NumPy ResNet block with padded convolutions, a skip connection, and ReLU after the residual addition. | https://www.tensortonic.com/study-plans/cracking-dl/dl-resnet-block |
| Scaled Dot-Product Attention | Implement scaled dot-product attention, the fundamental building block of all Transformer architectures. | https://www.tensortonic.com/study-plans/cracking-dl/dl-scaled-dot-product-attention |
| Sequence-to-Sequence Forward Pass | Implement the forward pass of a sequence-to-sequence (seq2seq) model using a simple RNN for both the encoder and decoder. | https://www.tensortonic.com/study-plans/cracking-dl/dl-sequence-to-sequence |
| Squeeze-and-Excitation Block | Implement a Squeeze-and-Excitation (SE) block that adaptively recalibrates channel-wise feature responses. | https://www.tensortonic.com/study-plans/cracking-dl/dl-squeeze-excitation |
| Transposed Convolution | Implement transposed convolution from scratch as the learned spatial upsampling operation used by decoder networks. | https://www.tensortonic.com/study-plans/cracking-dl/dl-transposed-convolution |
| U-Net Skip Connection | Implement a minimal U-Net with one skip connection using fully connected layers (no convolutions). | https://www.tensortonic.com/study-plans/cracking-dl/dl-unet |
| Variational Autoencoder Forward Pass | Implement a NumPy variational autoencoder forward pass with mean, log variance, reparameterization, and reconstruction. | https://www.tensortonic.com/study-plans/cracking-dl/dl-vae |
| Vanilla RNN Cell | Implement a vanilla recurrent cell with sequential hidden states and optional backpropagation through time gradients. | https://www.tensortonic.com/study-plans/cracking-dl/dl-vanilla-rnn-cell |
| VGG Convolution Block | Build a NumPy VGG-style block with two valid three-by-three convolutions, ReLU activations, and max pooling. | https://www.tensortonic.com/study-plans/cracking-dl/dl-vgg-block |
| Vision Transformer Patch Embedding | Convert an image into non-overlapping Vision Transformer patch embeddings with a learned linear projection. | https://www.tensortonic.com/study-plans/cracking-dl/dl-vision-transformer |
| Weight Initialization | Compute per-layer NumPy weight initialization parameters from network dimensions and the selected initialization method. | https://www.tensortonic.com/study-plans/cracking-dl/dl-weight-initialization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/sallarohit1)
<!-- tensortonic:end -->
