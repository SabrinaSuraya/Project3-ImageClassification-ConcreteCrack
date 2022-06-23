# Project3-ImageClassification-ConcreteCrack
To classify the image of concrete wheteher it is crack or not
 link: https://data.mendeley.com/datasets/5y9wdsg2zt/2
 
 
 1. Summary
 - To make classify the image of concrete
 - to predict the new image of concrete
 - deal with a large dataset
 - the deep learning model is used and trained
 - The model use Transfer Learning from mobilenetV2

2. IDE and Framework
- The project built with Spyder as the main IDE
- use Tensorflow, Keras, Numpy, Mathplot

3. Methodology
- The folder contain 2 type of image which is positive(teh concrete is crack) and negative ( the concrete is not crack).
- There are 40 000 images, 20 000 in positive folder and 20 000 in negative folder.
- The we classify the images into training and validation dataset. 70% of total images used for training and 30% of images for validation test.
- 


- the model constist of 5 dense layers. 
- Model summary:
![image](https://user-images.githubusercontent.com/73817610/174966416-20240580-bc9e-4baf-b7b5-ded01dc06426.png)


-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= BinaryCrossentropy', metrics of accuracy, batch_size of 32 and epochs of 200
- The value is display by using TensorBoard:
