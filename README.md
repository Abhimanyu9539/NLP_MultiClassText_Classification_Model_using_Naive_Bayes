# Multi-class Text Classification using Naive Bayes
This repository contains the code for text processing and Naive Bayes Model for multi-class text classification.

### Installation
To install the dependencies run:
```buildoutcfg
pip install -r requirements.txt
```

### Dataset
The dataset is a collection of complaints about consumer financial products and services that we sent to companies for response. The actual text of the complaint by the consumer is given in the `Consumer complaint narrative` column. The dataset also has a `product` column which contains the product for which the consumer is raising the complaint. We are going to build a model to predict the product given the complaint text. 

### Train the model
To train the model run:
```buildoutcfg
python Engine.py --file_name complaints.csv --input_path Input
```

### Predictions
To make prediction on a new review `Its the worst app ever I save my design lts not save`,  run:
```buildoutcfg
python predict.py --test_complaint "I was looking through my report and noticed a vehicle that I returned to the dealership where it's reporting completely incorrect. I had the opportunity to talk to several lawmakers and friends, and learned some basic laws in regards to voluntary or repossession of a vehicle. Under the laws of MASS and UCC 9.506 as well as State RISA and MVISA statutes, a deficiency can not be claimed unless all of the required notices were properly and timely given, and all of the allowable redemption and cure time limits were adhered to. PLEASE HAVE THEM IMMEDIATELY REMOVE"
```
