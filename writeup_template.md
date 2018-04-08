# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./testImage/1.jpg "Traffic Sign 1"
[image2]: ./testImage/12.jpg "Traffic Sign 2"
[image3]: ./testImage/Stop.jpg "Traffic Sign 3"
[image4]: ./testImage/17.jpg "Traffic Sign 4"
[image5]: ./testImage/25.jpg "Traffic Sign 5"


---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The code is present in the zip file


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images so that the data has mean zero and equal variance. To do that I created a function named as normalize .

In this first I changed the image data to single channel image and then used this formula (pixel - 128)/ 128 on the resultant data so that the data has mean zero and equal variance.

After this i get a normalized image data which i will be using in training validation and testing of my images .

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

Currently the data is not augmentated in the project . If the augmentaion would have been implemented then the accuracy would have also increased .
When we  have few labels for a given data compared to the other data in training data then there may be a case in which if the netwrok come across the same image while testing then the accuracy of judging will be small enough to be taken into consideration . Therefore augmentation is helpful in such cases.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GreyScale Image				         
| Convolution 3x3     	| 1x1 stride, valid padding outputs 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, valid padding outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| Input 400 Weight(400,120) Bias(120) O/P 120	|
| RELU					|												|
| DROPOUT				|70%    										|          
| Fully connected		| Input 120 Weight(120,84 ) Bias(84) O/P 84 	|
| RELU					|												|
| DROPOUT				|80%    										|
| Fully connected		| Input 120 Weight(84,43 ) Bias(43) O/P 43   	|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet for the most part that was given, but I did add an additional dropout after the RELU of fully connected layer abd the AdamOptimizer with a learning rate of 0.001. The epochs used was 100 while the batch size was 128. Other important parameters I learned were important was the number and distribution of additional data generated. I played around with various different distributions of image class counts and it had a dramatic effect on the training set accuracy. It didn't really have much of an effect on the test set accuracy, or real world image accuracy. Even just using the default settings from the Udacity lesson leading up to this point I was able to get 94% accuracy with virtually no changes on the test set. When I finally stopped testing I got 93.8-94.5% accuracy on the test set though so I think it could have increased if i would have implemented the augmentation but because of the time constraint my focus was on implementing the network.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.9%
* test set accuracy of 94.4%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
  I used a very similar architecture to the one given in the classroom under the deep learning module. Because it helped to visualize and pratically implement the concepts that I learned in the classroom. 
  
* What were some problems with the initial architecture?
  The issue was the lack of Knowlege of all parameters. Due to which i was unable to get accuracy more than 90-92% so I shifted to LeNet architecture.
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.


I didn't alter much past adding a couple dropouts with a 70% probability for the first fully connected layer and then 80% probality for the second connected layer.


* Which parameters were tuned? How were they adjusted and why?
    
  Epoch, learning rate, batch size, and drop out probability were all parameters tuned For Epoch the main reason I tuned this was i was gettin accuracy in increasing order for the first 50 epoch so i had a strong confidence that if i increase it to 100 then I will get a better accuracy.
  
The batch size was not much altered since the size of good enough to provide a good accuracy . The learning rate is .001  as I am told it is a normal starting point, but I just wanted to try something different so .00097 was used. I think it mattered little. The dropout probability mattered a lot early on, but after awhile I set it to 70% for the first fully connected layer and 80% for the second fully connected layer and then i just left it. 



* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I think I could go over this project for another week and keep on learning. I think this is a good question and I could still learn more about that. I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

#
![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The road work  be difficult to classify because of the background present in the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road Work    			| Traffic Signal								|
| 30 km/h				| 30 km/h										|
| No entry	      		| No entry   					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| Road Work										|
| .05					| 30 km/h										|
| .04	      			| No entry  					 				|
| .01				    | Priority Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


