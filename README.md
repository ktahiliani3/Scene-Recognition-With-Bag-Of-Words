This project involves the recognition and categorization of a particular scene from a given dataset of images. We start by taking tiny images as image features and then use the nearest neighbor classifier to classify the images. The tiny image features are obtained by down sampling the image to a fixed resolution which we have chosen as 16X16 for this project. In simple words, the nearest neighbor classifier classifies a test feature into a specific category by assigning it the label of the nearest training example. The second part of this project deals with taking SIFT features of the images and then classifying using a linear support vector machine. In short, the project can be divided into the following parts:


Using tiny image features and nearest neighbour classifier.
Using SIFT features and nearest neighbour classifier.
Using SIFT features and linear support vector machine classifier.
