# **Traffic Sign Recognition** 

---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1a]: ./project_pics/random_training_images.jpg "Random Images"
[image1b]: ./project_pics/visualization.jpg "Training Set Visualization"
[image1c]: ./project_pics/visualization.jpg "Validation Set Visualization"
[image1d]: ./project_pics/visualization.jpg "Test Set Visualization"
[image3]: ./project_pics/resampled_training_set.jpg "Training Data Resampled"
[image4]: ./Traffic_Signs/image1.jpg "Traffic Sign 1"
[image5]: ./Traffic_Signs/image5.jpg "Traffic Sign 2"
[image6]: ./Traffic_Signs/image2.jpg "Traffic Sign 3"
[image7]: ./Traffic_Signs/image3.jpg "Traffic Sign 4"
[image8]: ./Traffic_Signs/image4.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sidharth2189/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classes for the training, validation and test datasets. Also visualized are random images from the training set of the german traffic sign images.

![alt text][image1a]

![alt text][image1b]

Average number of training samples per class should be total number of training samples divided by number of classes. This comes to 809 samples roughly. From the above graph, most classes seem under-represented in the training dataset.

![alt text][image1c]

![alt text][image1d]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the images to grayscale and then normalize the image data. The image classes differ in content by geometry and shape, while color is not the definitive factor here for differentiation. Further, a well conditioned optimization problem is easier to solve for the optimizer. This means the input data has a distribution that is symmetric about the mean. After grayscaling, image data that come in pixel values of 0 to 255 have been normalized between 0.1 and 0.9 using the expression (0.1 + image_data_gray * (0.9 - 0.1)/255). Also, the training data set is shuffled to eliminate any possible bias that order of images may allow to creep in to the classifier.

I decided to generate additional data because most classes seem under-represented in the training dataset and imbalanced classes can put classifier accuracy [out of business](https://elitedatascience.com/imbalanced-classes).

The training data set is unevenly represented across the classes which can cause the classifier to become biased. One of the [ways](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) to solve imbalanced class data set problem is to oversample classes that are under-represented in the training data set, with replacement. This is done using [SMOTE()](https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167) imported from the [imblearn module](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE.fit_resample). SMOTE() finds the n-nearest neighbours in the minority class for each of the samples. Then it draws a line between the neighbours and generates random points on the lines.

After augmenting training data, the classes are evenly distributed as in graph below:

![alt text][image3]

The difference between the original data set and the augmented data set is that the number of training samples were increased from 34799 to 86430, spread out evenly across classes. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x40 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x80 					|
| Fully connected		| outputs = 500									|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| outputs = 200									|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| outputs = 43 									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture with TensorFlow. The initial validation accuracy was low. As such, I experimented with the dimension of the layers. For example increasing the filter depth to better extract features, along with using dropout at the fully connected layers helped increase validation accuracy. The ADAM optimizer was used over the SGD optimization for better accuracy. The batch size was set to 128, number of epochs to 20 and learning rate to 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.977 
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images might be difficult to classify because they are downloaded from the web and have different pixel height and width from what the network accepts. So, they are resized using cv2.resize() function and preprocessed by grayscaling normalizing

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

