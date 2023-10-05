# ImageEnhancement-cs7180

This is the submission for 

Image Enhancement - Project 1
for the course "CS7180 Advanced Perception"

20th September 2023

---------------------------------------------
Details

## Project Report

- FileName - "cs7180-AnirudhGugan-Project1.pdf"

## Authors
- Anirudh Muthuswamy, NUID - 002783250
- Gugan Kathiresan, NUID -002756523

## Operating System
- Google Colab Ubuntu - Tesla T4 GPU
- macOS for file management

## Required Packages
In your command line
> cd /path/to/your/project
> pip install -r requirements.txt

## Compilation Instructions for the "Master-AnirudhGugan-cs7180.ipynb" demo file.
- The experimentation and demo of code was performed inside the ipynb file named "Master-AnirudhGugan-cs7180.ipynb".
- The notebook is written in such a way that each section corresponds to a component of the project, and the user can just Run-all to understand the demo.
- While this is an ipynb file, each operation was defined as a user-defined function and called only in relevant locations.
- Certain aspects like Datasets and Model definition have been written in a class structure for easy inheritance and command line interfacing.
- The notebook was coded with the thought of reducing code duplication and maintaining software engineering principles in mind.
- But at the same time the ipynb was preferred to display outputs and visualizations beneficial for a research perspective

## Compilation Instructions for the script files
- Download the our version of the DIV2k dataset from the link https://www.kaggle.com/datasets/anirudhmuthuswamy/div2k-hr-and-lr
- Unzip the files and place it in the working directory
- Install all requirements from the "requirements.txt" file
- Run the "preprocess.py" file that generates all the necessary dataset files
- To train the base SRCNN model, run the train_SRCNN.py file
- To train the base SRCNN model with the VGG Loss, run the train_SRCNN_VGGLoss.py file
- To train the SRCNN2 model with the VGG Loss, run the train_SRCNN2_VGGLoss.py file


## Time Travel Days
- None Used
