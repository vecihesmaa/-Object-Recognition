# CIFAR-10 Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

### Project Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. 

The classes are:

- Airplane 
- Automobile 
- Bird 
- Cat 
- Deer 
- Dog 
- Frog 
- Horse 
- Ship 
- Truck 
This project uses a simple CNN model to classify the images into one of these 10 categories.

# Installation
To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required packages using pip:

bash
pip install tensorflow
pip install keras 


# Model Architecture
The model is a Sequential CNN model with the following layers:

- Conv2D: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Conv2D: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Flatten
- Dense: 64 units, ReLU activation
- Dense: 10 units, Softmax activation (for classification)

# Training
The model is compiled with the Adam optimizer and trained using the sparse categorical crossentropy loss function for 15 epochs.

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(xtrain, ytrain, epochs=15)

# Evaluation
The model is evaluated on the test dataset, achieving an accuracy of around 68.2%.
model.evaluate(xtest, ytest)

# Prediction
The model predicts the class labels for the test dataset. The predictions are displayed using sample images from the dataset.
ypred = model.predict(xtest)
ypred1 = [np.argmax(element) for element in ypred]

# Example
You can visualize the predictions on test images using the following helper function:
def example(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(siniflar[y[index]])
    
# License
This project is open source and available under the MIT License.
