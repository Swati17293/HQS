## A Commonsense-Infused Language-Agnostic Learning Framework for Enhancing Prediction of Political Polarity in Multilingual News Headlines

## Abstract

> Predicting the political polarity of news headlines is a challenging task as they are inherently short, catchy, appealing, context-deficient, and contain only subtle bias clues. It becomes even more challenging in a multilingual setting involving low-resource languages. Our research hypothesis is that the use of additional knowledge, such as commonsense knowledge can compensate for a lack of adequate context. However, in a multilingual setting, it becomes ineffective as the majority of the underlying knowledge sources are available only in high-resource languages, such as English. To overcome this barrier, we propose to utilise the Inferential Commonsense Knowledge (IC_Knwl) via a Translate-Retrieve-Translate strategy to introduce a learning framework for the prediction of political polarity in multilingual news headlines. To evaluate the effectiveness of our framework, we present a dataset of multilingual news headlines.


## Dataset
    
> The dataset and its generation scripts are stored in the data folder.
Follow https://github.com/allenai/comet-atomic-2020/ to retrieve the Inferential Commonsense Knowledge (IC_Knwl)
Use https://cloud.google.com/translate for translations

## To generate the predictions use one of the following files 

```
python3 main.py
```
