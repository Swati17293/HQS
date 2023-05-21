# Hierarchical Deep Multi-modal Network for MedicalVisual Question Answering

## Introduction

> Visual Question Answering in Medical domain (VQA-Med) plays an important role in providing medical assistance to the end-users. These users are expected to raise either a straightforward question with a Yes/No answer or a challenging question that requires a detailed and descriptive answer. The existing techniques in VQA-Med fail to distinguish between the different question types sometimes complicates the simpler problems, or over-simplifies the complicated ones. It is certainly true that for different question types, several distinct systems can lead to confusion and discomfort for the end-users. To address this issue, we propose a hierarchical deep multi-modal network that analyzes and classifies end-user questions/queries and then incorporates a query-specific approach for answer prediction. We refer our proposed approach as Hierarchical Question Segregation based Visual Question Answering, in short HQS-VQA. 

## Requirements

  * pandas
  * scikit-learn
  * matplotlib
  * pillow
  * nltk
  * keras

## Dataset.
    
> Dataset is taken from the paper available at https://www.nature.com/articles/sdata2018251

## Input data format

```
image_id|question|answer
...
```
## Train/load the model and generate the predictions

> Download glove.6B.300d.txt file and place it in the data/external/glove folder.

```
python3 main.py
```

## Evaluate and compare the models

```
python3 evaluate.py
```
    
## Reference

If you are using this resource then please cite our paper:

Gupta, Deepak, Swati Suman, and Asif Ekbal. "Hierarchical deep multi-modal network for medical visual question answering." Expert Systems with Applications 164 (2020): 113993.

```
@article{gupta2020hierarchical,
  title={Hierarchical deep multi-modal network for medical visual question answering},
  author={Gupta, Deepak and Suman, Swati and Ekbal, Asif},
  journal={Expert Systems with Applications},
  volume={164},
  pages={113993},
  year={2020},
  publisher={Elsevier}
}
```

    
## License
> MIT License
