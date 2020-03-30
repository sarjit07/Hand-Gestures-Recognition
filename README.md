# Hand-Gestures-Recognition  
Hand Gestures Recognition using OpenCV, Keras, Convolutional Neural Networks

## Required Libraries:
    - python 3.6.7
    - openCV (pip install opencv-python)
    - matplotlib
    - keras
    - sklearn

## Repo Content:
- **cam_run.py** : Contains the main method which would start the Video Capture, segment the hand region, pass on the thresholded image to the CNN model, and predict the hand gesture from LIVE camera feed.

- **train_gestures.ipynb** : The Jupyter notebook which contains the data loading, creation and training of the 2D Convolutional Neural Network. 

- **my_dataset** : The Dataset consists of 6 types of gestures. Mainly, 
  - BLANK
  - OK
  - THUMBSUP
  - THUMBSDOWN
  - PUNCH
  - HIGH-FIVE

## USAGE
Details can also be found here: 
```https://www.kaggle.com/sarjit07/hand-gestures-recognition-with-opencv-and-cnn```
1. Download repo
2. Install required dependencies (Libraries)
3. Go to repo directory and run following command on terminal ```$ python cam_run.py ```

## DEMO
![](optimized.gif)
