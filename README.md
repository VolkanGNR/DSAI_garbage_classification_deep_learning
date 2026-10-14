# Garbage Classification Using Deep Learning

**Computer Vision Capstone Project | Data Science and Artificial Intelligence Minor | MEF University**  
**Author:** Volkan Güner | **Project date:** February 2026

## Overview

This project explores whether transfer learning can be used to classify images of waste into six categories. It compares three pretrained convolutional neural network (CNN) architectures—**MobileNetV2, EfficientNetB0, and ResNet50**—using a small, imbalanced image dataset.

The analysis, preprocessing, model training, evaluation, and visualizations were carried out in Python. The accompanying Jupyter Notebook contains the original code and saved outputs.

## Dataset

The notebook loads **431 labeled images** across six classes:

| Class | Images |
|---|---:|
| Paper | 108 |
| Glass | 82 |
| Plastic | 74 |
| Cardboard | 70 |
| Metal | 68 |
| Trash | 29 |

The experiment uses a stratified split with **311 training images, 55 validation images, and 65 test images**. The uneven class distribution is addressed with class weights during training.

**Data availability:** The image dataset and label file are **not included** in this repository. Their original download source and redistribution terms should be checked before sharing the images publicly.

## Methodology

1. **Exploratory analysis:** Examine class counts and display sample waste images.
2. **Preprocessing:** Load and resize images to 224 × 224 pixels; construct a TensorFlow `tf.data` pipeline.
3. **Augmentation:** Apply horizontal flips, rotations, zoom, and contrast adjustments.
4. **Transfer learning:** Train MobileNetV2, EfficientNetB0, and ResNet50 with ImageNet-pretrained weights.
5. **Two-stage training:** First train with the pretrained backbone frozen, then fine-tune approximately the final 20% of backbone layers.
6. **Evaluation:** Compare classification reports, confusion matrices, learning curves, and test accuracy.

Training uses AdamW optimization, class weights, early stopping, and learning-rate reduction on a validation-loss plateau.

## Results

Results recorded in the notebook:

| Model | Test accuracy |
|---|---:|
| EfficientNetB0 | **78.46%** |
| ResNet50 | 76.92% |
| MobileNetV2 | 72.31% |

**EfficientNetB0** achieved the highest accuracy on the 65-image test split. The notebook also contains an inference function that produces a predicted class and class probabilities for a supplied image.

## Technologies

Python · TensorFlow / Keras · scikit-learn · NumPy · pandas · Matplotlib · Jupyter Notebook

## Notebook

- [`Capstone01.ipynb`](Capstone01.ipynb) — original exploratory analysis, model training, evaluation, and inference code, **unchanged**.

### Running the original notebook

Open `Capstone01.ipynb` in Jupyter Notebook or JupyterLab and provide the image dataset and metadata in the relative locations expected by the notebook. In particular, the existing code refers to:

- `grabage_txt/one-indexed-files-notrash_test.txt` — the label/filename list (**spelling preserved from the notebook**).
- `Garbage classification_in/Garbage classification_inner/` — the image root, with class-specific subfolders.

The notebook's first image-path check reports missing files; a later cell resolves paths through `IMAGES_ROOT`. Both cells have been retained. Reproduction also depends on compatible library versions and the external image data, which are not bundled here.

## Limitations

- The dataset is small, particularly the **trash** category; the 65-image test result should not be interpreted as a production-level benchmark.
- Results may vary with a different train/test split, environment, or dataset.
- The metadata filename contains `test`, although the notebook constructs its **own** train/validation/test split from those 431 image entries. The dataset's original partitioning should be verified before using these results as a comparison with other benchmarks.
- Images taken in different real-world environments may be harder to classify than the examples in this experiment.

## Project status

Completed academic capstone. This repository documents the original experiment and its observed results; it is not a deployed waste-sorting system.
