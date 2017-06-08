# Behavioral Cloning using Udacity simulator

[//]: # (Image References)

[image1]: ./images/bright.png "Bright Image"
[image2]: ./images/clr.png "YUV image"
[image3]: ./images/crop.png "Crop Image"
[image4]: ./images/flip.png "flip Image"
[image5]: ./images/normal.png "normal Image"
[image6]: ./images/resized.png "resized Image"
[image7]: ./images/recovery.jpg "recovery Image"

The Behaviour cloning project involves in training a convolution neural network to learn a driver behaviour and reproduce the behaviour to autonomously navigate a trained path.

In this project the following tasks are accomplished.

- Generate training data set by driving in the unity simulator.
- Design and implement the convolution neural network architecture.
- Train the model using the training data set.
- Generate a sucessfull drive around the track in an autnomous mode using the trained model.

Files Uploaded

- model.py - This conatins the script to train and validate the model. The network architecure is implemented in the file.
- drive.py - The script to drive the car in autonomous mode using the trained model.
- imagemanager.py - The script that contains the utilities to read the image, peform necessary image processing and a generator to feed the model.
- model.h5 - The trained model weights.
- Video.mp4 - The video of the center camera during the autonomous drive.

## Model Architecture and Training Strategy

The implementation is based on the [NVIDIA model] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The architecture seems to be proven for this project. The architecture is based on several layers of convolution network followed by several fully connected layers.

The following additions are made to adapt the model to this project.
- A lambda layer is added to normalise the training data. ( Code line 82)
- A specific learning rate is used for the adam optimizer.
- In order to avoid overfitting a dropout layer has been added. .( Code line 100)

|layer				 | shape  				 |
|:------------------:|:---------------------:|
|Input Normlaisation | 66 x 200 x 3			 |
|Convolution 		 | Filter 24, Kernel (5 x 5), Stride (2 x 2) , "elu"|
|Convolution 		 | Filter 36, Kernel (5 x 5), Stride (2 x 2) , "elu"|
|Convolution 		 | Filter 48, Kernel (5 x 5), Stride (2 x 2) , "elu"|
|Convolution 		 | Filter 64, Kernel (3 x 3), Stride (2 x 2) , "elu"|
|Convolution 		 | Filter 64, Kernel (3 x 3), Stride (2 x 2) , "elu"|
|Dropout 		 	 | 0.5					 |	
|Flatten 		 	 | 1164, "relu"			 | 
|Flatten 		 	 | 100, "relu"			 | 
|Flatten 		 	 | 50, "relu"			 |
|Flatten 		 	 | 10, "relu"			 |
|Flatten 		 	 | 1, 			 		 |


The model uses RELU functions in the dense layer and ELU function in the convolution layer to introduce non-linearity in the layers. 

#### Model parameter tuning

After monitoring several training session a learning rate of 1e-4 was choosen for the adam optimiser.

#### Training data

The behaviour of the car in the simulator is very sensitive to the quality of the training data and driver behaviour. After serveral attempts a good sample of training data was generated to train the model. Methods employed to train the model is covered in the next section.


|										Autonomous Drive Video					       |
|:------------------------------------------------------------------------------------:|
|[![Test Track](./images/Track.png)](https://www.youtube.com/watch?v=uAmqHHTDNF8&t=28s)|



###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
