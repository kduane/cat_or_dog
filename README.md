# Species Classifier

## Purpose:

I have multiple small animals underfoot while I'm coding, cats and dogs.  While my dog is canine shaped and behaved, she is a small breed and ruff-ly the same size or smaller than my cat.  

This project combines two open source datasets to create a single sample:  
    1. the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) which includes a limited number of samples for each category and over 120 separate dog breeds.  
    2. the [Kaggle Cat Breeds Dataset](https://www.kaggle.com/ma7555/cat-breeds-dataset) which includes a limited number of samples for 67 different cat breeds

While identifying which pet is at my feet is an easy task for me as a human, it becomes a larger challenge for a computer.  This project aims to build a binary image classification model to check whether a pet is a cat or a dog from a photo, with a stretch goal of applying real-time testing to a webcam.

### Phase 1 - Problem Definition  
    1.1 Broad Goals  
    1.2 Data Source  
    1.3 Problem Statement 

### Phase 2 - Data Gathering  
    2.1 Load Files

### Phase 3 - Exploratory Data Analysis  
    3.1 Dataset Shape
    3.2 Sample Images  
    3.3 Principal Component Analysis
 
### Phase 4 - Modeling  
    4.1 Train/Test/Split  
    4.2 Convolutional Neural Net    

### Phase 5 - Model Analysis  
    5.0 Baseline Score  
    5.1 Compare Accuracy Scores  

### Phase 6 - Conclusions  
    6.1 Revisit 1.3 Problem Statement  
    6.2 Conclusions  
    6.3 Recommendations for Further Research 


## Credits/References/Acknowledgements

Dataset Reference:

Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.

Joseph Redmon and Santosh Divvala and Ross Girshick and Ali Farhadi, You Only Look Once: Unified, Real-Time Object Detection, 2016

Code Inspiration:

Anmol Behl, Video Streaming Using Flask and Open CV, https://medium.com/datadriveninvestor/video-streaming-using-flask-and-opencv-c464bf8473d6