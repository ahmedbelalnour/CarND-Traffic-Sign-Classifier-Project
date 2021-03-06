# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the train, validate and test [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)
* Convert the images to greyscale
* Explore, summarize and visualize the data set
* Normalize data
* Increase data set for the output classes that have small number of dataset by shifting, rotating, and shifting rotating 
* Shuffle the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1a]: ./examples/first_visualization.PNG "First Visualization"
[image1b]: ./examples/second_visualization.PNG "Second Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3a]: ./examples/Rotated_image.png "Rotated Images"
[image3b]: ./examples/Shifted_image.png "Shifted Images"
[image4]: ./examples/German_Traffic_Sign.png "15 German Traffic Sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one, and here is a link to my [project code](https://github.com/ahmedbelalnour/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Following is a basic summary of the data set, and an exploratory visualization of the dataset. In the code, the analysis done using python, numpy methods rather than hardcoding results manually. 

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed

![alt text][image1a]

You may notice that the number of data set is not normally distributed for the output classes, so we increased the number of training data set for the classes that have small number of data set by shifting/ rotating or shifting and rotating, the new data set details the following:
* The size of training set is 62487
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed

![alt text][image1b]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale for simplicity because it has only one color layer instead of three layers for the color image

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Normalized the image data to avoid high and low frequancy noise and make the image data satisfied normal distribution which is mean =0

I decided to generate additional data because for some output classes there are small number of tain data set and for other output classes there are a large number of the train data set.

To add more data to the the data set, I used the shifting and rotating techniques for the existing dataset as shown in the following images

Here is an example of an original image and an augmented image:

![alt text][image3a]
![alt text][image3b]

The difference between the original data set and the augmented data set is the added shifted and rotated images


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   							| 
| Convolution 1     	| input = 32x32x1, valid padding, output = 28x28x16 	|
| Convolution 2     	| input = 28x28x16, same padding, output = 28x28x32 	|
| Max pooling	      	| input = 28x28x32, 2x2 stride, valid padding, outputs 14x14x32 				|
|	Dropout					|	Keep probability 0.9											|
| Convolution 3     	| input = 14x14x32, valid padding, output = 10x10x64 	|
| Convolution 4     	| input = 10x10x64, same padding, output = 10x10x128 	|
| Max pooling	      	| input = 10x10x128, 2x2 stride, valid padding, outputs 5x5x128 				|
|	Dropout					|	Keep probability 0.8											|
| Fully connected	1	| input = 3200, output = 2400       									|
| RELU					|												|
|	Dropout					|	Keep probability 0.7											|
| Fully connected	2	| input = 2400, output = 1600       									|
| RELU					|												|
|	Dropout					|	Keep probability 0.6											|
| Fully connected	3	| input = 1600, output = 800       									|
| RELU					|												|
|	Dropout					|	Keep probability 0.5											|
| Fully connected	3	| input = 800, output = 43       									|
| Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
* learning rate = 0.001
* Number of epochs = 20
* Batch size = 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96.2 %
* validation set accuracy of 94.5 %
* test set accuracy of 73.3 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
We tried the Lenet architecture used in the MNIST lab.

* What were some problems with the initial architecture?
The accuracy of the architecture did not meet the needed accuracy as it was about 87%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
We made many modifications to the model like:
 - Modifying the Lenet model architecture by increasing convolutional layers, fully connected layers, and increasing the number of neurons in each hidden layer as disccused below.
 - Augment the data set after finding that the distribution of the data set is unstable at all.
 - Convert the images to gray scale.
 - Shift, and rotate the training images with random values.
 - Get the histogram equalization for the input images to get rid of the low density images, and high density images in the training data set.

* Which parameters were tuned? How were they adjusted and why?
The hyperparameters tuned are : learning rate, number of epochs, batch size, and drop out.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? 
How might a dropout layer help with creating a successful model?
Convolutional layers works perfectly in the kind of problems as it deals with images, and as the number of convolutional layers get deep, they can detect more complex patterns in the images, and classify more accurately.

If a well known architecture was chosen:
* What architecture was chosen? LeNet architecture.
* Why did you believe it would be relevant to the traffic sign application? Because it deals with images that have multiple classes and many features to extract from each image.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final accuracy on the training is 96.2 %, the validation is 94.5 %, and the testing is 73.3 %.
 

### Test a Model on New Images

#### 1. Choose 15 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 15 German traffic signs that I found on the web:

![alt text][image4]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Keep right			| Keep right									|
| No entry      		| No entry   									| 
| Double curve     		| Double curve 									|
| Children crossing		| Children crossing				 				|
| Turn left ahead		| Turn left ahead      							|
| Keep left				| Keep left										|
| Roundabout mandatory	| Priority road									|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	| 
| Speed limit (20km/h)	| Speed limit (120km/h)							|
| Speed limit (30km/h)	| Ahead only				 					|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Speed limit (70km/h)	| Stop											|
| stop					| Stop						 					|
| No entry				| No entry										|


The model was able to correctly guess 11 of the 15 traffic signs, which gives an accuracy of 73.3 %.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Yield sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   										| 
| 0.0     				| Keep left 									|
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Speed limit (30km/h)			 				|
| 0.0				    | Speed limit (50km/h) 							|

For the second image, the model is relatively sure that this is a Keep Right sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep Right									| 
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Speed limit (30km/h)			 				|
| 0.0				    | Speed limit (50km/h) 							|
| 0.0				    | Speed limit (60km/h) 							|

For the third image, the model is relatively sure that this is a Stop sign (probability of 0.98). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| Stop 		   									| 
| 0.0     				| Traffic signals								|
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Bicycles crossing					 			|
| 0.0				    | General caution      							|

For the forth image, the model is relatively sure that this is a No entry sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| No entry   									| 
| 0.0     				| Turn left ahead 								|
| 0.0					| Stop  										|
| 0.0	      			| Turn right ahead					 			|
| 0.0				    | No passing for vehicles over 3.5 metric tons	|

For the fifth image, the model is relatively sure that this is a Double curve sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Double curve   								| 
| 0.0     				| Ahead only 									|
| 0.0					| Beware of ice/snow							|
| 0.0	      			| Speed limit (60km/h)			 				|
| 0.0				    | Speed limit (50km/h) 							|