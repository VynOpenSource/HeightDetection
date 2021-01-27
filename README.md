# HeightDetection

## Table of contents

- [Introduction](#introduction)
- [Pipeline](#pipeline)
- [Folder Structure](#folder-structure-and-details)
- [Dataset](#dataset)
- [Steps to use project](#steps-to-use-project)
- [Requirements](#requirements)
- [Installation](#installation)
- [Execution](#execution-procedure)
- [Accuracy] (#accuracy)
- [Output](#output)
- [Dataset References](#dataset-references)
- [Creators](#creators)

## Introduction

Working at height is one of the biggest cause of fatalities and major injuries. It includes falls from high raised platforms, ladders or through fragile surfaces.

So we have trained a machine learning model to detect if a person is working at a height in given image.

## Pipeline
![Pipeline](https://github.com/ManavBansal/Vyntelligence/blob/master/src/readmeImages/pipe.png?raw=true)

## Dataset 

Dataset includes thousands of positive, negative and hard examples for the person working at height, collected from open sources.
This dataset is saved in drive, can be accessed from [here](https://drive.google.com/drive/folders/1_pDFkAN7P4u_QcJ4hScMMtl6UVNoMzEi?usp=sharing).

Some of the links from which images have been downloaded, are listed in [Dataset References](#dataset-references).

## Steps to use project


##### Step1: Get “Git” Link.

Clone or download > Copy the link!

##### Step2: Simply run below command.

"!git clone {link}"

##### Step3: Open the Folder in Google Drive.

Folder has a same name as the Github repo :)

##### Step5: Open The Notebook

Right Click > Open With > Colaboratory

##### Step6: Run

Now you are able to run Github repo in Google Colab.

## Accuracy

Two class Model-> Training Accuracy: 94% , Testing Accuracy: 91% (Height, Not Height)
Three class Model-> Training Accuracy: 83% , Testing Accuracy: 72% (Height, Not Height, Hard)


## Output
<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/src/outputImages/imagen1.jpg" width="400" height="250">

Not height, Probability [8.759406e-11]


<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/src/outputImages/imagen2.jpg" width="400" height="250">

Not height, Probability [2.0659362e-20]


<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/src/outputImages/imagep1.jpg" width="400" height="250">

Height, Probability [0.99990416]


<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/src/outputImages/imagep2.jpg" width="400" height="250">

Height, Probability [0.99976885]


## Dataset References

- Unsplash
- Pixabay
- Free Nature Stock
- Realistic Shots
- Life of pix
- Gratisography
- StockSnap
