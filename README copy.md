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
