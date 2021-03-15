

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

Import `src/image_analysis_services.py` at the top of the file which will give you access to all of the methods
defined in the file.

## Example Starter Code
Open the following file from your favourite IDE:
```
examples/tester.py
```
To see sample usage of this project.

Run the file and you should see the camera window launch with information about faces on the image. Note that this tester file 
can detect multiple faces in a single frame.



![sample usage](examples/ImageQualityAnalysis 2021-03-14 21-30-24_Trim.mp4) 






## Credits

This project borrowed code from [this](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a)
TDS article by Vardan Agarwal. The reason we decided to make this project was to simplify the functions outlined above
into a single project which can be used by users.

We would also like to thank [@KadenMc](https://github.com/KadenMc) for being a great mentor throughout this development process. 
  

## Collaborators

The project was created as the conclusion to the 2020 cohort of LeanAI hosted by [UofTAI](https://www.uoft.ai/).

It was developed by [@ushinghal19](https://github.com/ushinghal19), [@talicopanda](https://github.com/talicopanda), [@oshalash38](https://github.com/oshalash38), [@Addison-Weatherhead](https://github.com/Addison-Weatherhead), and myself.


