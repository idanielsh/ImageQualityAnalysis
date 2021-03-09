# ImageQualityAnalysis

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General Info 
* ImageQualityAnalysis is a convinient-to-use interface that allows users to detect facial features of faces in an image.
* Also allows quick computation of the following features:
    * Detect face location relative to the image
    * Find the endpoint of a line if it were drawn from the person's nose to the image
    * Detect (x,y) angle of where the user is looking
## Requirements
* Python 3.6-3.8 **(Important)**
* Tensorflow 2.2 **(Also important)**
* Numpy
* opencv-python
## Installation
After installing all dependencies run
```
git clone https://github.com/idanielsh/ImageQualityAnalysis.git
```

## Example Starter Code
Run the following file from your favourite IDE:
```
examples/tester.py
```
And you should see the camera window launch with information about faces on the image. Note that the information is only
printed every 25 seconds to maintain readability. This rate can be changed in `examples/tester.py` but does not reflect 
the true performance of the app. 


## Credits

## Collaborators

