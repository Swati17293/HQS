"Hierarchical Deep Multi-modal Network for MedicalVisual Question Answering"

1. Create the conda environment from the environment.yml file to install the dependencies.
    
    conda env create --name yourenvname --file=environment.yml

2. Activate the environment.

    conda activate yourenvname

3. Download and Prepare the dataset.
    
    -- download the dataset from the paper available at https://www.nature.com/articles/sdata2018251
    prepare the dataset by selecting the required fields and splitting it in train and test set

    for reference see the sample set in data/raw/ folder

4. Train/load the model and generate the predictions

    python3 main.py

5. Evaluate and compare the models

    python3 evaluate.py
	
# My new project

## Introduction

> An introduction or lead on what problem you're solving. Answer the question, "Why does someone need this?"

## Code Samples

> You've gotten their attention in the introduction, now show a few code examples. So they get a visualization and as a bonus, make them copy/paste friendly.

## Installation

> The installation instructions are low priority in the readme and should come at the bottom. The first part answers all their objections and now that they want to use it, show them how.
