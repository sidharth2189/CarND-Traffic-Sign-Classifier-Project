## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project uses deep neural networks and convolutional neural networks to classify traffic signs. A model is trained and validated so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, it is tried on images of German traffic signs that is found on the web, to see if it can correctly predict traffic signs.

In the repository: 
* the file [Traffic_Sign_Classifier.ipynb](https://github.com/sidharth2189/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) is the Ipython notebook with the code
* the code is exported as an [html file](https://github.com/sidharth2189/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
* the [writeup](https://github.com/sidharth2189/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md) report discusses the approach to the objectives


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

* If working from the workspace, everytime the environment is set up, navigate to project repository and install the imblearn module using pip install imblearn

* Other modules required are, Tensorflow, Pandas, Numpy, Matplotlib and OpenCV

### Dataset and Repository

1. [Download the data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

[Reference Repository](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)

