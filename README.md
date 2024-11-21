# Pixel Art Classifier: A FastAI and Hugging Face Deployment

Welcome to the **Pixel Art Classifier**! This project uses FastAI to build a machine learning model capable of classifying pixel art images into two categories: **Character** and **Scenery**. The trained model is deployed using Gradio on Hugging Face Spaces for interactive use.

---

## Features

### 🎨 Pixel Art Classification
- Classifies images into two categories:
  - **Character**: Pixel art of characters.
  - **Scenery**: Pixel art of landscapes or environments.

### ⚡ Deployment
- The model is deployed on **Hugging Face Spaces** using **Gradio**, allowing anyone to upload pixel art images and get instant predictions.

### 📖 Learn from the Notebook
- Includes a well-documented Jupyter notebook to guide you through:
  - Data collection and preprocessing.
  - Model training and evaluation.
  - Deployment on Hugging Face Spaces.

---

## Getting Started

### Repository Overview
- **`pixel.ipynb`**: Main notebook with step-by-step documentation.
- **`export.pkl`**: Trained model file ready for deployment.
- **`app.py`**: Python script for deployment using Gradio and Hugging Face.
- **`README.md`**: This file!

### How to Use the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/joao-l-rossi/PixelClassification.git
