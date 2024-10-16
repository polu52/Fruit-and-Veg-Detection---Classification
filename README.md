# Fruit and Vegetable Detection Project

This project aims to classify fruits and vegetables from images using machine learning techniques. With practical applications in agriculture and food sorting, automating this task can improve efficiency and accuracy. The project explored both a custom CNN model and transfer learning to enhance performance.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Comparison of Models](#comparison-of-models)
- [Reflection](#reflection)
- [Streamlit Web Application](#streamlit-web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
This project addresses the challenge of fruit and vegetable classification using image data. Accurate classification can streamline agricultural processes and food sorting systems. Two main approaches were used:
1. A **custom Convolutional Neural Network (CNN)**.
2. **Transfer learning** using a pretrained VGG16 model.

## Dataset
The dataset consists of labeled images of fruits and vegetables. Key preprocessing steps included:
- **Resizing** images to 128x128 pixels.
- **Normalization** of pixel values.
- **Data Augmentation** to increase variability and prevent overfitting.

## Model Architecture

### 1. Custom CNN:
- **Input**: 128x128 pixel images.
- **Layers**: Conv2D with ReLU activation, MaxPooling2D for downsampling, followed by dense layers.
- **Output**: A softmax layer for classification.
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam.

### 2. Transfer Learning:
- **Pretrained Model**: VGG16 finetuned on the fruit and vegetable dataset.
- **Modification**: The final fully connected layers were replaced to match the number of classes in the dataset.
- **Benefit**: Pre-learned feature extraction from large datasets reduces the need for extensive training.

## Results

- **Custom CNN**: Achieved over 90% training accuracy but struggled with new images, resulting in around 50% validation accuracy due to overfitting.
- **Transfer Learning**: The pretrained model significantly improved performance, reaching approximately 90% validation accuracy.

## Comparison of Models

- **Accuracy**: Transfer learning outperformed the custom CNN, achieving a 40% higher validation accuracy.
- **Training Time**: The transfer learning model required fewer epochs to converge, benefiting from prelearned features.
- **Generalization**: The transfer learning model demonstrated better generalization, particularly in distinguishing visually similar categories.

## Reflection
Using transfer learning dramatically improved both model performance and training efficiency. Future iterations of the project may include:
- Further finetuning of additional layers in the pretrained model.
- More extensive data augmentation to increase model robustness.

## Streamlit Web Application
A web application was created using Streamlit, allowing users to upload images of fruits and vegetables and receive predictions on their classification.

You can try the web application hosted on Hugging Face Spaces:
👉 [Fruit and Vegetable Detection Web App](https://huggingface.co/spaces/poluhamdi/Fruit-Veg_Detector)

Explore the project's code on Kaggle:
👉 [Kaggle Notebook](https://www.kaggle.com/code/hamdipolu/fruit-and-veg-detection)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fruit-veg-detection.git
    cd fruit-veg-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `data/` folder.

## Usage
To run the Streamlit web application locally, use the following command:
```bash
streamlit run app.py
```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
