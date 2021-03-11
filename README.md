# ImageQualityAnalysis

## Table of contents
* [General info](#general-info)
* [Requirements](#Requirements)
* [Installation](#Installation)
* [Example Starter Code](#Example-Starter-Code)
* [Credits](#Credits)
* [Collaborators](#Collaborators)


## General Info 
* ImageQualityAnalysis is a convinient-to-use interface that allows users to detect facial features of faces in an image.
* Also allows quick computation of the following features:
    * Detect face location relative to the image
    * Find the endpoint of a line if it were drawn from the person's nose to the image
    * Detect (x,y) angle of where the user is looking
    * Detects if the users mouth is open
  
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

This project borrowed code from [this](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a)
TDS article by Vardan Agarwal. The reason we decided to make this project was to simplify the functions outlined above
into a single project which can be used by users.

We would also like to thank @KadenMc for being a great mentor throughout this development process. 
  

## Collaborators

The project was created as the conclusion to the 2020 cohort of LeanAI hosted by [UofTAi](https://www.uoft.ai/).

It was developed by @ushinghal19, @talicopanda, @oshalash38, @Addison-Weatherhead, and myself.


