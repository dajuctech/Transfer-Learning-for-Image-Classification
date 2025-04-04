# Transfer Learning for Dog vs. Cat Classification

This project demonstrates an end-to-end pipeline for image classification using transfer learning. The workflow involves preparing the environment, loading and augmenting a small dataset of dog and cat images, building a deep learning model based on MobileNetV2, and training and evaluating the model for binary classification.

---

## 1. Environment Setup

- **Library Installation and Imports:**  
  The notebook installs required packages including `tensorflow`, `keras`, `opencv-python`, `matplotlib`, and others. Key libraries are imported for image processing, data handling, and deep learning.
  
- **Google Drive Integration:**  
  The code mounts Google Drive to access a zipped dataset and then extracts it to a temporary directory for further processing.

---

## 2. Data Loading and Preparation

- **Dataset Extraction:**  
  A zip file containing a small "dogs vs. cats" dataset is copied from Google Drive and extracted.  
  The dataset is organized into training and validation directories with subfolders for cats and dogs.

- **Directory Inspection:**  
  The script lists sample file names and prints the total number of images in each class (cats and dogs) for both training and validation sets.

- **Image Display:**  
  A sample of images from the dataset is displayed in a grid to visually verify data quality and content.

---

## 3. Data Augmentation

- **Augmentation Strategy:**  
  Two separate `ImageDataGenerator` objects are created for training and validation data with rescaling.  
  Additional data augmentation (rotation, shifts, shear, zoom) is applied to further increase the variety of images.
  
- **Batch Concatenation:**  
  Augmented images are generated and then concatenated to the original datasets, enriching the training and validation sets for better model generalization.

---

## 4. Model Building with Transfer Learning

- **Base Model:**  
  MobileNetV2 pre-trained on ImageNet is loaded without its top layers. The base model is frozen to retain learned features.
  
- **Custom Layers:**  
  A Sequential model is built on top of the base:
  - A flatten layer converts feature maps into a one-dimensional vector.
  - A fully connected (Dense) layer with 512 neurons and ReLU activation (with L2 regularization) is added.
  - Batch normalization and dropout are used to improve training stability and reduce overfitting.
  - The output layer uses a sigmoid activation function for binary classification.

- **Compilation:**  
  The model is compiled with the Adam optimizer (learning rate set to 0.0001) and binary cross-entropy loss.

---

## 5. Callbacks and Training

- **Callbacks:**  
  Early stopping and a learning rate reduction callback (ReduceLROnPlateau) are configured to monitor the validation loss and prevent overfitting.

- **Model Training:**  
  The model is trained using the augmented image generators over 10 epochs. Training and validation metrics are logged for performance tracking.

- **Visualization:**  
  The training history is plotted to show accuracy and loss curves, providing insight into model convergence over epochs.

---

## Conclusion

This project showcases the use of transfer learning with MobileNetV2 for classifying images of dogs and cats. By combining robust data augmentation with a carefully tuned CNN architecture, the pipeline effectively learns discriminative features for binary classification. The modular design allows for further experimentation and can be adapted to similar image classification challenges.


