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
- [Accuracy](#accuracy)
- [Output](#output)
- [Dataset References](#dataset-references)
- [Creators](#creators)

## Introduction

Working at height is one of the biggest cause of fatalities and major injuries. It includes falls from high raised platforms, ladders or through fragile surfaces.

So we have trained a machine learning model to detect if a person is working at a height in given image.

## Pipeline
![Pipeline](https://github.com/VynOpenSource/HeightDetection/blob/main/src/readmeImages/pipe.png)

## Dataset 

Dataset includes thousands of positive, negative and hard examples for the person working at height, collected from open sources.
This dataset is saved in drive, can be accessed from [here](https://drive.google.com/drive/folders/1_pDFkAN7P4u_QcJ4hScMMtl6UVNoMzEi?usp=sharing).

Some of the links from which images have been downloaded, are listed in [Dataset References](#dataset-references).

## Steps to use project


##### Step1: Clone this folder to you local PC.
( If in windows, also install MingW for 'make' commands.

##### Step2: Open command prompt, run "pip install -r requirements.txt"

##### Step3: Now for to download already trained models. 

- Run "make getModel2" for 2 class model in directory ./models.
- Run "make getModel3" for 3 class model in directory ./models.

##### Step4: To test these models:

- Insert images in testingImages folder, which you want to test.
- Then run "make test2" for 2 class model results.
- Or run "make test3" for 3 class model results.

##### Step5: To download dataset:

- Run "make getData2" to get data for 2 class model in directory ./data/data2.
- Run "make getData3" to get data for 3 class model in directory ./data/data3.

##### Step6: To train again, after downloading dataset:

- Run "make train2" and your new model2 will be saved in modelNew directory.
- Run "make train3" and your neew model3 will be saved in modelNew directory.

## Accuracy

***Two class Model***-> Training Accuracy: **94%** , Testing Accuracy: **91%** (Height, Not Height)

***Three class Model***-> Training Accuracy: **83%** , Testing Accuracy: **72%** (Height, Not Height, Hard)


## Output
<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/testingImages/imagen1.jpg" width="400" height="250">

Not height, Probability [8.759406e-11]


<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/testingImages/imagen2.jpg" width="400" height="250">

Not height, Probability [2.0659362e-20]


<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/testingImages/imagep1.jpg" width="400" height="250">

Height, Probability [0.99990416]


<img src="https://github.com/VynOpenSource/HeightDetection/blob/main/testingImages/imagep2.jpg" width="400" height="250">

Height, Probability [0.99976885]


## Dataset References

- Unsplash
- Pixabay
- Free Nature Stock
- Realistic Shots
- Life of pix
- Gratisography
- StockSnap
