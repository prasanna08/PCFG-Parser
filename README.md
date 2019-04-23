# NLU Assignment 2
This code implements PCFG CKY parser.

## Prerequisites

### Requirements
Following dependencies are required to run this code.
* nltk
* numpy

Alternatively, you can use following commands to install necessary libraries.
```
pip install -r requirements.txt
``` 

## Training PCFG from scratch.
To train model from scratch use `Trainer.py`. Run following command:
``` 
python Trainer.py
```

## Model inference.
To evaluate model use following command:
```
python Parse.py  [--sentence="<SENTENCE>"]
```

Arguments:
*  `--sentence`: Specify the sentence to parse. Use appropriate escape characters as and when necessary. Place **sentence inside quotes**.
