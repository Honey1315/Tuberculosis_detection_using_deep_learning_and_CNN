# Tuberculosis Detection Using Deep Learning and CNN

This project applies deep learning techniques to detect the presence of Tuberculosis (TB) in Chest X-ray images. We used two models: a custom Convolutional Neural Network (CNN) and a pre-trained VGGNet, to classify chest X-rays as either "Normal" or "Tuberculosis". The dataset used for this project is the **Pulmonary Chest X-ray Abnormalities** dataset, sourced from Kaggle.

## Dataset

- **Name**: Pulmonary Chest X-ray Abnormalities
- **Source**: [Kaggle](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities)
- **Description**: This dataset contains chest X-ray images, with labels indicating if the image shows signs of Tuberculosis (TB) or not. It includes images from patients with TB as well as healthy individuals.
  
## Models Used

1. **Custom CNN Model**
    - We designed a custom Convolutional Neural Network from scratch to classify the chest X-ray images.
    - It was trained for a specified number of epochs and achieved decent performance metrics.

2. **VGGNet**
    - We also used a pre-trained VGGNet model, which is known for its performance in image classification tasks.
    - This model was fine-tuned for the specific task of Tuberculosis detection and also trained for a specified number of epochs.

## Results

### Model 1: VGGNet

- **Accuracy**: 86.93%
- **Precision**: 92.00%
- **Recall**: 77.53%
- **F1 Score**: 84.15%

### Model 2: Custom CNN

- **Accuracy**: 78.89%
- **Precision**: 77.78%
- **Recall**: 79.38%
- **F1 Score**: 78.57%

The VGGNet model outperformed the custom CNN model in most metrics, particularly in precision, indicating it is better at avoiding false positives. The custom CNN, however, performed reasonably well given it was built from scratch.

## Dependencies

To run this project, you will need the following libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Kaggle API (for downloading the dataset)

You can install the dependencies by running:
pip install tensorflow keras numpy pandas matplotlib scikit-learn kaggle


## Usage
- Dataset Preparation: First, download the dataset from Kaggle.
  kaggle datasets download -d kmader/pulmonary-chest-xray-abnormalities
  Extract the dataset and ensure the images are in the correct format for input to the models.
- Training the Models: The models can be trained using the provided code. Both the VGGNet and custom CNN models can be trained separately by running the appropriate scripts.
- Evaluation: After training, both models will output performance metrics including Accuracy, Precision, Recall, and F1 Score.
- Prediction: Use the trained models to make predictions on new chest X-ray images to detect the presence of Tuberculosis.

## Conclusion
This project demonstrates the application of both a custom CNN and a VGGNet model for the task of TB detection from chest X-rays. The pre-trained VGGNet model provided higher accuracy and precision, but the custom CNN still showed competitive results. This indicates the potential of deep learning models in the field of medical imaging and TB detection.
