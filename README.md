# Fashion MNIST Deep Learning Challenge

This project demonstrates an end-to-end deep learning pipeline for classifying fashion items from the Fashion MNIST dataset. The solution leverages data augmentation, a custom convolutional neural network with regularization, and advanced training techniques to achieve robust performance.

---

## 1. Environment Setup

- **Library Installation & Imports:**  
  The notebook installs and imports required libraries such as TensorFlow, Keras, NumPy, Matplotlib, and more.  
  It sets a random seed across Python, NumPy, and TensorFlow to ensure reproducibility.

- **Dataset Loading:**  
  The Fashion MNIST dataset is loaded directly using Keras' built-in methods.  
  Data shapes are confirmed to have 60,000 training and 10,000 testing images of 28×28 pixels.

---

## 2. Data Preprocessing

- **Normalization & Reshaping:**  
  Pixel values are normalized to the [0, 1] range and reshaped to add a channel dimension, preparing the images for CNN input.

- **Label Encoding:**  
  The target labels are converted into one-hot encoded vectors for multi-class classification.

---

## 3. Data Augmentation

- **Augmentation Techniques:**  
  The code utilizes Keras’ `ImageDataGenerator` to apply random rotations, shifts, shearing, zooming, and horizontal flips.  
  Augmented batches are generated and concatenated with the original dataset, effectively increasing the training and testing data.

---

## 4. Model Building

- **CNN Architecture with Regularization:**  
  The model is built using the Keras Sequential API and features:
  - **Block 1:** Two convolutional layers (32 filters) with ReLU activation, batch normalization, max pooling, and dropout.
  - **Block 2:** Two convolutional layers (64 filters) with similar settings to extract higher-level features.
  - **Fully Connected Layers:** A flattening layer followed by a dense layer with 256 neurons (with dropout) and a softmax output layer for classification.
  - **Regularization:** L2 regularization is applied to mitigate overfitting.

- **Compilation:**  
  The network is compiled with the Adam optimizer, using categorical cross-entropy as the loss function and accuracy as the metric.

---

## 5. Model Training and Evaluation

- **Callbacks:**  
  Early stopping and learning rate reduction callbacks are configured to dynamically adjust the learning process and prevent overfitting.

- **Training:**  
  The model is trained for 10 epochs with a batch size of 64, using both training and validation datasets.

- **Visualization:**  
  Training and validation accuracy and loss curves are plotted to monitor the model’s learning progress.

---

## Conclusion

This notebook outlines a comprehensive deep learning approach for fashion item classification using the Fashion MNIST dataset. By incorporating data augmentation and a well-regularized CNN architecture, the model achieves competitive performance. The modular design of the pipeline allows for further experimentation and adaptation to similar image classification challenges.


