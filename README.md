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
![image](https://user-images.githubusercontent.com/73817610/175441144-f6fafeba-43f6-4319-86ac-c5ca0099513d.png)



-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= BinaryCrossentropy', metrics of accuracy, batch_size of 32 and epochs of 200
- The value is display by using TensorBoard:
- ![image](https://user-images.githubusercontent.com/73817610/175440981-0a8b8a63-ebb4-4260-94d2-fb949a20e101.png)


![image](https://user-images.githubusercontent.com/73817610/175440728-ddd1fd46-2706-4a09-857e-b45ad99b5122.png)



Predicting a new image
![image](https://user-images.githubusercontent.com/73817610/175445492-b8bc30a0-276a-4f99-ab58-0808cd018dc3.png)

![image](https://user-images.githubusercontent.com/73817610/175445528-89291039-bd9d-4bd7-89a9-07072873e46c.png)


