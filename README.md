# RESNet-Deep-Learning

Implementation of RESNet using CNN Architecture

Residual Network (ResNet) is a Convolutional Neural Network (CNN) architecture that overcomes the “vanishing gradient” problem, making it possible to construct networks with up to thousands of convolutional layers, which outperform shallower networks.

ResNet improves the efficiency of deep neural networks with more neural layers while minimizing the percentage of errors.

This code is a complete workflow for building and training a deep learning model (ResNet50) for image classification on the "Flower Photos" dataset. It uses TensorFlow and Keras to load the dataset, create the model, compile it, train it, evaluate its performance, and make predictions. Additionally, it includes code to visualize the training history and display the confusion matrix for the model's performance.


1. Importing the necessary libraries. The code begins by importing various libraries, including TensorFlow, NumPy, Matplotlib, OpenCV (cv2), Seaborn, and scikit-learn's confusion_matrix.

2. Downloading and preparing the "Flower Photos" dataset. The dataset is downloaded from the specified URL and saved locally using TensorFlow's get_file function. It is then loaded using pathlib.Path and stored in the data_dir variable. The dataset is split into training and validation sets using image_dataset_from_directory, and the class names are extracted using the class_names attribute of the training dataset.

3. Visualizing a few sample images from the dataset. Some sample images are displayed from the training dataset using Matplotlib. For each image, the corresponding class name is shown as the title.

4. Creating and compiling the ResNet50 model. The ResNet50 model is built using Sequential API from Keras. The pre-trained ResNet50 model from TensorFlow's applications is used as a base with its top layers removed. Only the last layer (pooling layer) is retained. The model is then flattened and followed by two dense layers, one with 512 neurons and ReLU activation, and the other with 5 neurons (equal to the number of classes) and softmax activation. The model's summary is printed to display its architecture.

5. Compiling the model. The model is compiled using the Adam optimizer with a learning rate of 0.001, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.

6. Training the model. The model is trained on the training dataset and validated on the validation dataset. The number of epochs is set to 5. The training history is stored in the history variable.

7. Visualizing the training history. Two plots are created to visualize the model's accuracy and loss on both the training and validation datasets over the epochs.

8. Making predictions. A sample image from the "roses" class is read using OpenCV, resized to the required input shape (180x180), and converted to a NumPy array. A prediction is made using the trained model, and the predicted class label is displayed.

9. Computing and displaying the confusion matrix. The code computes the confusion matrix for the model's performance on the training dataset. The true labels (y_true) are collected from the dataset, and predictions (y_pred) are made using the model. The confusion matrix is then calculated and displayed as a heatmap using Seaborn's heatmap function.

The code provides a comprehensive example of how to build, train, evaluate, and visualize a deep learning model for image classification using TensorFlow and Keras. It uses the popular ResNet50 architecture as the base model and demonstrates important steps in the machine learning workflow.
