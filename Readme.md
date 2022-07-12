Image Processing

* ImageProcessingMCD.ipynb : 
Given an image, you can choose between 3 different pipelineA, pipelineB, pipelineD. For each of these pipelines you can choose a strategy for feature extraction, such as GLCM,
HaarGLCM and LBPGLCM and perform preprocessing, candidate and features extraction. The results are saved in an output folder chosen by the user.
The whole process can take between 10 and 15 minutes. However for this demo, the image was downsample to 8bits, so the results are obtained faster.
The images were not downsized in the Micro Calcification Detection project.


Machine Learning

* MachineLearningMCD.ipynb *: This notebook contains the main code used for Machine Learning. The main strategies for dealing with Unbalanced Data are in the Preprocessing section (function rus()).
In the section *Training functions* the code for training with a 2-fold cross validation is showed, using two strategies: negative_pool (that uses Successive Enhancement Learning) and cv_classification() that
uses Random Undersampling. In this notebook, the functions to calculateFROC and drawCurve can be found in the senction of *Scores*.

* ignore big calcifications *: This notebook is used to check the candidates and remove the ones that match with calcifications greater than 15 pixels diameter. In this way, those are not used as part of the training.




Deep Learning

* DeepLearningMCD.ipynb *: This notebook contains the main code used in the Deep Learning part of the project. The 3 architectures are mentioned here, as well as the training and testing functions to obtain the results.
The probability maps generation are also considered, as well as the averaging of the results for the CNN ensemble architecture.. This function calls a python file name roi_project.py that contains the DataLoader for the project.
Architecture CNN_final corresponds to a CNN using an incremental and classification block, that will perform an ensemble. 
Architecture CD_CNN corresponds to a CNN using depthwise convolution.
Architecture Resnet50 is preloaded.

* proi_file_generator.ipynb *: File to generate the proi files that will be used by the DataLoader. These files are composed by the name or the key of the image, and the top-left most pixel of each patch, with its coordinates x and y. The patches are considering the breast mask, 
so it does not take patches from the background of the image.

* roi_cc_project.py *: DataLoader taken from professor Alessandro Bria. This file was slightly modified to meet our requirements.