# Reimagining Monochrome: A Deep Learning Approach to Image Colorization 

Final project for the course 02456 - Depp Learning - DTU

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── dlcolorization  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model

```


 This project explores the automatic colorization of grayscale images using convolutional autoencoders trained on RGB and LAB color spaces.

---

## Project Overview

The goal of this project was to design, train, and evaluate deep learning models to automatically add color to grayscale images using different datasets and architectures. We compared performance across RGB and LAB color spaces and explored how dataset domain and size affect the output quality.

---

## Model Architectures

### 1. **Baseline Autoencoder**
- Simple encoder-decoder with Conv2D and Conv2DTranspose layers.
- Used ReLU activations.
- Performed poorly—unable to reproduce realistic colors.

### 2. **UNet-like Autoencoder**
- Deeper architecture with 5 downsampling/upsampling layers.
- LeakyReLU, BatchNorm, Dropout, and skip connections.
- Significantly improved image quality and detail retention.

### 3. **LAB-Space UNet-like Autoencoder**
- Predicts LAB channels instead of RGB.
- Uses tanh activation to match LAB output range.
- Focuses learning on chrominance (color) while preserving luminance.

---

## Datasets Used

| Dataset      | Purpose           | Size     |
|--------------|-------------------|----------|
| COCO         | General-purpose   | 5k / 40k |
| Dogs         | Domain-specific   | 5k       |
| Landscapes   | Domain-specific   | 5k       |

- All images resized to 160x160.
- Training/validation/test split: 80% / 10% / 10%.
- Colorization performed in both RGB and LAB color spaces.

---

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Primary loss metric. Measures pixel-wise error.
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality in dB.
- **SSIM (Structural Similarity Index)**: Measures preservation of structure and texture.

Tracked using [Weights & Biases](https://wandb.ai).

---

## Results Summary

| Model         | Dataset     | MAE   | PSNR  | SSIM  |
|---------------|-------------|-------|-------|-------|
| RGB-Unet      | COCO 5k     | 0.053 | 23.22 | 0.89  |
| RGB-Unet      | COCO 40k    | 0.054 | 23.02 | 0.90  |
| RGB-Unet      | Dogs 5k     | 0.053 | 23.07 | 0.90  |
| RGB-Unet      | Landscapes  | 0.049 | 24.01 | 0.93  |
| LAB-Unet      | COCO 40k    | 0.052 | 23.18 | 0.68  |

- RGB-based models outperformed LAB-based ones.
- COCO dataset proved too diverse; better results were obtained with domain-specific datasets.
- LAB models underperformed possibly due to lack of dedicated hyperparameter tuning.

---

## Optimized Hyperparameters

| Parameter     | Value     |
|---------------|-----------|
| Learning Rate | 0.001     |
| Batch Size    | 32        |
| Optimizer     | Adam      |
| Dropout Rate  | 0.2       |

---


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
