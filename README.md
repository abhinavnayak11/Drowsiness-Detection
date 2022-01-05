# Drowsiness Detection

## Table of Content
  * [Overview](#overview)
  * [Application](#application)
  * [Approach](#approach)
  * [Installation](#installation)
  * [Future scope](#future-scope)

## [Overview](#table-of-content)
This project will detect a drowsy person through live webcam, alert them while displaying that person's face. 

**<p align="center">App demo</p>**

![](images/demo.gif)

## [Applications](#table-of-content)
- Monitor drivers to avoid accidents
- Monitor students in class to sport the inattentive one (although, that would be mean :p)

## [Approach](#table-of-content)
1. Using dlib face detector, detect all the faces from live video.
2. Using dlib landmarks preditor, locate 68 landmarks on each face.
3. Calculate the average eye aspect ratio (EAR) for the eyes. 
4. Extract the face which has EAR < 0.22 (value got after trial and error) for more than 10 frames, display it while giving an alert sound

## [Installation](#table-of-content)
- You need Python (3.6) & git (to clone this repo)
- `pip install -r requirements.txt` : This will install all the dependencies. 
- `git clone https://github.com/abhinavnayak11/flight-prediction.git .` : To clone this repo
- `cd path/to/project`
- `python -m src.drowsiness-detection` : To run the script

