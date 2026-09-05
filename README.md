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
| 2D Sinusoidal Positional Embedding | Build the 2D sin-cos positional embedding used by ViT-MAE, DINOv2, and similar models. | https://www.tensortonic.com/study-plans/cracking-cv/cv-2d-sincos-pos-embed |
| Adaptive Average Pool 2D | Implement adaptive two-dimensional average pooling with variable, possibly overlapping regions for arbitrary output sizes. | https://www.tensortonic.com/study-plans/cracking-cv/cv-adaptive-avg-pool-2d |
| Anchor Box Generation | Generate axis-aligned anchor boxes for a feature map of shape (feature h, feature w) at stride s. | https://www.tensortonic.com/study-plans/cracking-cv/cv-anchor-generation |
| Average Pool 2D | Apply two-dimensional average pooling to an image with the requested kernel and stride, without padding. | https://www.tensortonic.com/study-plans/cracking-cv/cv-avg-pool-2d |
| Bilinear Resize | Resize a grayscale image with corner-aligned bilinear interpolation matching PyTorch interpolation semantics. | https://www.tensortonic.com/study-plans/cracking-cv/cv-bilinear-resize |
| Box Encode and Decode | Implement Faster-RCNN style box regression target encoding and decoding, using two arrays of axis-aligned boxes in xyxy format. | https://www.tensortonic.com/study-plans/cracking-cv/cv-box-encode-decode |
| Per-Channel Mean and Std | Compute population means and standard deviations for every channel across a batch of images in NHWC layout. | https://www.tensortonic.com/study-plans/cracking-cv/cv-channel-statistics |
| CLIP Cosine Retrieval | Rank corpus embeddings for each CLIP-style query by stable cosine similarity, including safe handling of zero vectors. | https://www.tensortonic.com/study-plans/cracking-cv/cv-clip-cosine-retrieval |
| Multichannel 2D Convolution | Implement multichannel two-dimensional cross-correlation with output filters, optional bias, stride, and padding. | https://www.tensortonic.com/study-plans/cracking-cv/cv-conv2d-multichannel |
| 2D Transposed Convolution | Implement two-dimensional transposed convolution matching PyTorch output layout, stride, padding, and optional bias. | https://www.tensortonic.com/study-plans/cracking-cv/cv-conv2d-transpose |
| CutMix Augmentation | Apply deterministic CutMix by pasting a supplied image region and mixing labels according to the retained pixel area. | https://www.tensortonic.com/study-plans/cracking-cv/cv-cutmix |
| Depthwise Separable Convolution | Factor image convolution into per-channel spatial filters and a pointwise convolution that mixes output channels. | https://www.tensortonic.com/study-plans/cracking-cv/cv-depthwise-separable-conv |
| Dice Loss | Compute binary segmentation Dice loss from probability and target masks with the specified smoothing term. | https://www.tensortonic.com/study-plans/cracking-cv/cv-dice-loss |
| Dilated 2D Convolution | Implement a 2D dilated cross-correlation (no kernel flip) on a single multi-channel image. | https://www.tensortonic.com/study-plans/cracking-cv/cv-dilated-conv2d |
| FPN Top-Down Fusion | Fuse multiscale feature maps through top-down nearest-neighbor upsampling and aligned lateral addition. | https://www.tensortonic.com/study-plans/cracking-cv/cv-fpn-fusion |
| Gaussian Blur 2D | Build a normalized separable Gaussian kernel and apply it to a grayscale image with the specified padding behavior. | https://www.tensortonic.com/study-plans/cracking-cv/cv-gaussian-blur |
| Image Normalize | Normalize each image channel by its supplied mean and standard deviation to produce standardized vision-model inputs. | https://www.tensortonic.com/study-plans/cracking-cv/cv-image-normalize |
| Max Pool 2D | Apply two-dimensional max pooling to an image with a configurable window and stride, without padding. | https://www.tensortonic.com/study-plans/cracking-cv/cv-max-pool-2d |
| Mean IoU for Segmentation | Compute the mean Intersection-over-Union (mIoU) for a semantic segmentation prediction. | https://www.tensortonic.com/study-plans/cracking-cv/cv-miou |
| MixUp Augmentation | MixUp augmentation forms a new training sample by taking a convex combination of two images and their labels using a mixing weight [0, 1]. | https://www.tensortonic.com/study-plans/cracking-cv/cv-mixup |
| Non-Maximum Suppression | Implement greedy non-maximum suppression matching torchvision selection order and strict IoU threshold behavior. | https://www.tensortonic.com/study-plans/cracking-cv/cv-nms |
| ViT Patchify | Split image batches into row-major non-overlapping patches and flatten each patch in spatial-then-channel order. | https://www.tensortonic.com/study-plans/cracking-cv/cv-patchify |
| RGB to Grayscale | Convert an RGB image to grayscale with luminance-weighted color channels for classical computer vision preprocessing. | https://www.tensortonic.com/study-plans/cracking-cv/cv-rgb-to-grayscale |
| Sobel Edge Detection | Apply horizontal and vertical Sobel filters to a grayscale image with one-pixel zero padding and return edge gradients. | https://www.tensortonic.com/study-plans/cracking-cv/cv-sobel-edges |
| Soft Non-Maximum Suppression | Apply Gaussian Soft-NMS by repeatedly selecting the strongest box and decaying overlapping box scores by IoU. | https://www.tensortonic.com/study-plans/cracking-cv/cv-soft-nms |
| Top-K Classification Accuracy | Given logits of shape (N, K) and integer targets of shape (N,) with each target in [0, K), compute the top-k classification accuracy. | https://www.tensortonic.com/study-plans/cracking-cv/cv-topk-accuracy |
| ViT Multi-Head Self-Attention | Implement Vision Transformer multi-head self-attention with combined QKV projection, scaled softmax, and output projection. | https://www.tensortonic.com/study-plans/cracking-cv/cv-vit-attention-block |
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
| AdaBoost from Scratch | Implement AdaBoost binary classification using decision stumps, weighted errors, adaptive sample weights, and weighted voting. | https://www.tensortonic.com/study-plans/cracking-ml/ml-adaboost |
| Agglomerative Clustering | Implement agglomerative hierarchical clustering with single, complete, and average linkage and deterministic cluster labels. | https://www.tensortonic.com/study-plans/cracking-ml/ml-agglomerative |
| AUC-ROC from Scratch | Build an AUC-ROC evaluator by ranking prediction scores, computing TPR and FPR, and integrating with the trapezoidal rule. | https://www.tensortonic.com/study-plans/cracking-ml/ml-auc-roc |
| Averaged Perceptron | Implement an averaged perceptron for binary classification with online mistake updates and mean weights across training steps. | https://www.tensortonic.com/study-plans/cracking-ml/ml-averaged-perceptron |
| Bagging Classifier | Build a bagging classifier from scratch using bootstrap-sampled CART trees and majority-vote predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-bagging-classifier |
| Decision Tree Classifier (CART) | Implement a CART decision tree classifier with Gini impurity splits, depth limits, and majority-class leaf predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-cart-classifier |
| Decision Tree Regressor | Implement a CART regression tree with MSE reduction splits, stopping criteria, and mean-value leaf predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-cart-regressor |
| Categorical Encoding | Encode categorical strings with deterministic label encoding or one-hot vectors ordered by sorted category values. | https://www.tensortonic.com/study-plans/cracking-ml/ml-categorical-encoding |
| DBSCAN | Implement DBSCAN clustering with epsilon neighborhoods, minimum-point density checks, cluster expansion, and noise labels. | https://www.tensortonic.com/study-plans/cracking-ml/ml-dbscan |
| Distance Metrics | Compute Euclidean, Manhattan, cosine, Chebyshev, and Minkowski distances between numeric vectors. | https://www.tensortonic.com/study-plans/cracking-ml/ml-distance-metrics |
| Feature Scaling | Implement column-wise min-max scaling and z-score standardization while handling constant features safely. | https://www.tensortonic.com/study-plans/cracking-ml/ml-feature-scaling |
| Gaussian Naive Bayes | Implement Gaussian Naive Bayes with class priors, per-feature Gaussian likelihoods, and log-probability predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-gaussian-naive-bayes |
| Gradient Boosted Regressor | Build gradient boosted regression trees that fit sequential residuals and combine learners with a configurable learning rate. | https://www.tensortonic.com/study-plans/cracking-ml/ml-gbr |
| Missing Value Imputation | Impute missing numeric values with column-wise mean, median, or most-frequent statistics while preserving observed values. | https://www.tensortonic.com/study-plans/cracking-ml/ml-imputation |
| Isolation Forest | Implement Isolation Forest anomaly detection with random partition trees and path-length based anomaly scores. | https://www.tensortonic.com/study-plans/cracking-ml/ml-isolation-forest |
| K-Fold Cross-Validation | Implement deterministic K-fold cross-validation with shuffled splits, held-out evaluation, and aggregated model scores. | https://www.tensortonic.com/study-plans/cracking-ml/ml-kfold-cv |
| K-Means Clustering | Implement K-means clustering with nearest-centroid assignments, centroid updates, convergence checks, and stable labels. | https://www.tensortonic.com/study-plans/cracking-ml/ml-kmeans |
| K-Means++ Initialization | Implement K-means++ initialization by sampling centroids according to squared distance from the nearest chosen center. | https://www.tensortonic.com/study-plans/cracking-ml/ml-kmeans-plusplus |
| KNN Classifier | Implement K-nearest neighbors classification using Euclidean distance, majority voting, and deterministic tie-breaking. | https://www.tensortonic.com/study-plans/cracking-ml/ml-knn-classifier |
| Lasso Regression | Implement Lasso regression with gradient descent, an L1 subgradient penalty on weights, and an unregularized bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-lasso-regression |
| Linear Discriminant Analysis | Implement Linear Discriminant Analysis for classification using class means, priors, and a shared covariance matrix. | https://www.tensortonic.com/study-plans/cracking-ml/ml-lda-classify |
| Linear Regression from Scratch | Train linear regression from scratch with mean squared error gradients for weights and bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-linear-regression-from-scratch |
| Log Loss (Binary Cross-Entropy) | Compute numerically stable binary log loss by clipping predicted probabilities before cross-entropy evaluation. | https://www.tensortonic.com/study-plans/cracking-ml/ml-log-loss |
| Logistic Regression from Scratch | Train binary logistic regression from scratch using sigmoid probabilities, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-logistic-regression |
| PCA from Scratch | Implement PCA by centering data, eigendecomposing the covariance matrix, and projecting onto the leading components. | https://www.tensortonic.com/study-plans/cracking-ml/ml-pca |
| Permutation Feature Importance | Measure permutation feature importance by shuffling each feature and comparing the accuracy drop against a baseline. | https://www.tensortonic.com/study-plans/cracking-ml/ml-perm-importance |
| Precision-Recall Curve & Average Precision | Build a precision-recall curve and compute average precision from ranked binary-classification scores. | https://www.tensortonic.com/study-plans/cracking-ml/ml-precision-recall-ap |
| Random Forest from Scratch | Implement a random forest classifier with bootstrap sampling, random feature subsets at each CART split, and majority voting. | https://www.tensortonic.com/study-plans/cracking-ml/ml-random-forest |
| Regression Metrics | Compute MSE, MAE, and R-squared from scratch, including constant-target handling and rounded metric output. | https://www.tensortonic.com/study-plans/cracking-ml/ml-regression-metrics |
| Ridge Regression | Train Ridge regression with gradient descent, L2-regularized weights, and an unregularized bias term. | https://www.tensortonic.com/study-plans/cracking-ml/ml-ridge-regression |
| Stacking Ensemble | Build a stacking classifier using cross-validated predictions from decision-stump and KNN base models to train a logistic meta-learner. | https://www.tensortonic.com/study-plans/cracking-ml/ml-stacking |
| SVM with Hinge Loss (SGD) | Train a linear SVM with sequential SGD updates on hinge loss, L2 weight regularization, and signed class predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-svm-hinge-sgd |
| Apply Ranked BPE Merges | Apply learned byte-pair merge rules to UTF-8 byte IDs in their supplied priority order, then reconstruct text through the supplied vocabulary. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l01-apply-bpe-merge-ranks |
| Train a Deterministic BPE Vocabulary | Choose the highest count with a lexicographic byte-string tie break, assign the next token ID, and replace non-overlapping matches from left to right. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l01-train-byte-pair-encoding |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-einsum-attention-scores |
| Gradient Accumulation Equivalence | Combine mean-loss gradients from unequal microbatches into one full-batch mean gradient, then apply a single SGD update. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-gradient-accumulation-step |
| Transformer Training FLOP Estimator | Estimate one training step from forward matrix multiplications and a supplied forward attention cost. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-training-flop-estimator |
| Mixed-Precision Training Memory Accountant | Compute exact storage for parameters, gradients, saved activations, and optimizer state from tensor shapes and byte widths. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-training-memory-accountant |
| Causal Grouped-Query Attention | Compute causal scaled dot-product attention in which each contiguous group of query heads shares one key/value head. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-causal-grouped-query-attention |
| Parameter-Matched SwiGLU Block | Choose a parameter-matched SwiGLU hidden width under an available-width limit, then evaluate the bias-free block. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-parameter-matched-swiglu |
| RMSNorm Forward Pass | Normalize each final-dimension vector by its root mean square and apply the learned scale without mean subtraction. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-rmsnorm-forward |
| Rotary Query and Key Embeddings | Rotate each adjacent coordinate pair of query and key vectors by a position-dependent angle. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-rotary-query-key-embeddings |
| Gated DeltaNet State Update | Decay the recurrent state, erase its component along a unit key, write the new value, and read the just-updated state. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-gated-deltanet-scan |
| Parallel and Recurrent Linear Attention | Compute causal softmax-free linear attention through both a parallel formulation and a recurrent state scan. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-linear-attention-duality |
| Mamba 2 Gated State Scan | Apply a gated recurrent state update and read each output from the just-updated state. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-mamba2-gated-state-scan |
| Top-k MoE Router with Load Statistics | Route each token to its highest-scoring experts, combine selected outputs, add the shared expert, and report load statistics. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-topk-moe-router |
| Global-Memory Coalescing Counter | Count the aligned cache lines touched by a warp's fixed-width global-memory accesses and measure useful transferred bytes. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l05-global-memory-coalescing |
| GPU Occupancy Calculator | Calculate resident blocks, resident warps, and occupancy from one block's resource use and one SM's limits. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l05-gpu-occupancy-calculator |
| Shared-Memory Bank Conflict Analyzer | Analyze GPU shared-memory addresses by warp, reporting bank indices and the conflict degree for each access step. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l05-shared-memory-bank-conflicts |
| Blockwise Online Softmax | Implement stable blockwise online softmax in CUDA for contiguous or strided rows across float32, float16, and bfloat16 inputs. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cuda/cs336-l05-blockwise-online-softmax |
| Masked Triton GELU Kernel | Implement masked tanh-approximate GELU in Triton for contiguous CUDA tensors, partial final blocks, and multiple dtypes. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/triton/cs336-l06-triton-masked-gelu |
| Fused Tiled Matmul and ReLU | Fuse tiled matrix multiplication and ReLU in Triton for strided float16 or bfloat16 CUDA inputs and preallocated outputs. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/triton/cs336-l06-triton-matmul-relu |
| Triton Row-Wise Softmax | Implement stable row-wise softmax in Triton for padded CUDA rows, partial tiles, and float32, float16, or bfloat16 data. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/triton/cs336-l06-triton-row-softmax |
| Tiled Triton Row Sum | Implement tiled row reduction in Triton for strided CUDA tensors, fixed-width tiles, and float32 output sums. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/triton/cs336-l06-triton-tiled-row-sum |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/sallarohit1)
<!-- tensortonic:end -->
