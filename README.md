# Traffic Sign Recognizer

## Overview
The Traffic Sign Recognizer is a machine learning-based web application designed to classify traffic signs from uploaded images. The application uses a convolutional neural network (CNN) model to identify different traffic signs, providing accurate predictions on the uploaded images. It provides a user-friendly interface for uploading images, displaying predictions, and showing the result in real-time.

## Features
- **Upload Image**: Users can upload an image of a traffic sign to get its classification.
- **Model Prediction**: The model predicts the class of the uploaded traffic sign.
- **Web Interface**: Clean, intuitive, and easy-to-use interface with a simple image upload button and results display.
- **Supports Multiple Formats**: PNG, JPG, JPEG image formats are supported.
- **Real-time Prediction**: Once an image is uploaded, the system immediately classifies it and shows the result.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning**: TensorFlow / Keras for training and deploying the model
- **Model**: Convolutional Neural Network (CNN)
- **Libraries**:
  - Flask: Python web framework for the backend
  - TensorFlow: For machine learning and model deployment

## Installation

### Prerequisites
- Python (preferably version 3.7+)
- pip (Python package manager)

### Setup
1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/traffic-sign-recognizer.git
   cd traffic-sign-recognizer

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the application:

   ```bash
   python app.py

4. Open your browser and go to http://127.0.0.1:5000/ to access the web app.

## Usage
- On the homepage, click the "Choose Image" button to upload an image of a traffic sign.
- The image will automatically be processed by the model, and the predicted class of the traffic sign will be displayed on the screen.
- Supported image formats include PNG, JPG, and JPEG.


## Model Training
- Collected a dataset of labeled traffic signs (e.g., GTSRB dataset).
- Preprocessed the dataset (resize images, normalize pixel values, etc.).
- Used a CNN model to train the dataset using TensorFlow/Keras.
- Saved the trained model using pickle [model1.pkl](model1.pkl).
- The Jupyter Notebook for Model training is [TrafficSignRecognizer.ipynb](TrafficSignRecognizer.ipynb)

### Model Accuracy
The trained model achieves an impressive accuracy of **99.83%** on the validation dataset, ensuring highly reliable predictions for classifying traffic signs.

## Acknowledgments
- **TensorFlow**: For the deep learning framework used to build the CNN model.
- **Flask**: For providing the web framework to build the app.
- OpenCV: For image preprocessing and enhancement.

## Future Enhancements
- Support for live camera input to detect traffic signs in real-time.
- Addition of more traffic sign classes and model optimization.
- Support for handling larger file sizes for image uploads.

