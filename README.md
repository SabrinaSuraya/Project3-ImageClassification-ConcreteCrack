# Project3-ImageClassification-ConcreteCrack
To classify the image of concrete wheteher it is crack or not
 link: https://data.mendeley.com/datasets/5y9wdsg2zt/2
 
 
# 1. Summary
 - To make classify the image of concrete
 - to predict the new image of concrete
 - deal with a large dataset
 - the deep learning model is used and trained
 - The model use Transfer Learning from mobilenetV2

# 2. IDE and Framework
- The project built with Spyder as the main IDE
- use Tensorflow, Keras, Numpy, Mathplot

# 3. Methodology
- The folder contain 2 type of image which is positive(the concrete is crack) and negative ( the concrete is not crack).
- There are 40 000 images, 20 000 in positive folder and 20 000 in negative folder.
- The we classify the images into training and validation dataset. 70% of total images used for training and 30% of images for validation test. 12 000 use for validation and 28 000 for training
- we create a pipeline for data augmentation including random flip and random rotation, to increate data's varities and prevent verfit in data training.


# Model
- use base model from mobilenetV2, uninclude the top layer ( generally for classification task) and freeze the trainable layer
- Set the base model layer below 100 layer into false, so that the model will not update its weight / paramter during training;

![image](https://user-images.githubusercontent.com/73817610/176494829-5ae4fc4e-20ee-4efb-aa84-ee04af04da8e.png)

- Create a classification layer as top layer of the base model. Use global Average Pooling 2D, activation function of 'softmax' and number of class is 2 because we want to classify the image of concrete into 2; positive or negative.
- Use Functional APi in creating Transfer Learning model:

![image](https://user-images.githubusercontent.com/73817610/176495848-929b26a3-213a-4836-9cda-c520d0cc6fea.png)

- Now, we need to train the model to update the trainable parameters.

# Model Evaluate
-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= Sparse Crossentropy', metrics of accuracy, batch_size of 32 and epochs of 200
- The value is display by using TensorBoard:

- ![image](https://user-images.githubusercontent.com/73817610/175440981-0a8b8a63-ebb4-4260-94d2-fb949a20e101.png)


![image](https://user-images.githubusercontent.com/73817610/175440728-ddd1fd46-2706-4a09-857e-b45ad99b5122.png)

# Model Prediction

Predicting a new image

![image](https://user-images.githubusercontent.com/73817610/175445492-b8bc30a0-276a-4f99-ab58-0808cd018dc3.png)

![image](https://user-images.githubusercontent.com/73817610/175445528-89291039-bd9d-4bd7-89a9-07072873e46c.png)


