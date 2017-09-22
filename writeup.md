#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

####Dataset summary


---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (1,32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

first i watched some of the pictures in the DS.
Then i plotted the the counts of the different Labels in a horizontal bar graph.
at last i created a Lookup dict for the Label to explanation mapping

![Plottet BarChart][./barchart.png]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Normalizing the pixel Values helps a neural Network, because they dont Learn well when there a huge Numerical differences

I noticed huge differences in the counts of the classes that is why i created some dummy data. The process created a lot of memory errors because availible Ram und in the End it did decreased the Validation Accuracy. because of that i discarded this step


The difference between the original data set and the augmented data set is the following: Reduced Validation Accuracy


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 16x16x64 	|
| Leaky RELU					|												|
| dropout			|     									|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 8x8x128 	|
| Leaky RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 8x8x64 				|
| dropout			|     									|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 4x4x256 	|
| Leaky RELU					|												|
| dropout			|     									|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 2x2x512 	|
| Leaky RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 1x1x512 				|
| dropout			|     									|
| flatten			|   									|
| Fully connected		| Leaky RELU, 256 Nodes|
| dropout			|     									|
| Fully connected		| Leaky RELU, 128 Nodes   	|
| dropout			|     									|
| Fully connected		| Leaky RELU, 64 Nodes   	|
| dropout			|     									|
| Fully connected		| "logits", 45 Nodes   	|
| dropout			|     									|
| Softmax				|     									|




####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adamOptimizer
alpha = 0.01
dropout = 0.8
rate = 0.0006
EPOCHS = 30
BATCH_SIZE = 128
Finding the right hyperparameters was fiddling . the Graph seems to be very sensitive to learning rate and dropout.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.981
* validation set accuracy of 0.957
* test set accuracy of 0.942

If an iterative approach was chosen:
First i tried the leNet Modell from the Previous lesson, but the Accuracy was low <70%.
Then i completely reworked the architecture many times.
Finally i implemented the current Modell, and tweaked it with Leaky RELU's instead of the normal ones
* Leaky_RELU's gave the Modell a much better Accuracy
* 4 conv layers with 2 Maxpooling Layers enabled the Modell to detect Features Accurate
* 3 Dense Layers plus 1 Logit Layer for good classification abilities
* Mainly the learning_rate,EPOCHS and dropout-rate was modifies
* adding dropout layers effectively prevents overfitting

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![Achtung Gefahr Sign][./data/test/atg.jpg] ![Stop Sign][./data/test/stp.jpg] ![Priority Road][./data/test/vfs.jpg]
![No Entry][./data/test/vbt.jpg] ![Speedlimit][./data/test/tpl.jpg] ![Manipulated Stop Sign][./data/test/man.jpg]

Last 2 Pictures are Tricky to predict. The speedlimit Sign was Scaled badly and the 2. Stop sign was Manipulated

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General Caution      		| General Caution    									|
| Stop Sign     			| Stop Sign  								|
| Priority Road					| Priority Road									|
| No Entry    		| 	No Entry 				 				|
| 70 km/h			| Pedestrians      							|
| Manipulated Stop			| Stop Sign      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 94,2%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31th cell of the Ipython notebook.

For every Picture the Model was certain to predict the right label with almost 100% Probability
Even the Manipulated Sign was predicted right.
Suprisingly it had trouble to predict the Speedlimit.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .40         			| Pedestrians   									|
| .10     				| Speedlimit 20km/h 										|
| .08					| Children crossing											|
| .05	      			| EOSL 80kn/h					 				|
| .05				    | Speedlimit 120km/h    							|
