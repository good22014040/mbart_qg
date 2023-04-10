# mbart_qg
use mbart model to train a chinese question generating model
## install requirements.txt
>pip install -r requirements.txt
## download training data
download train data from [Delta Reading Comprehension Dataset](https://github.com/DRCKnowledgeTeam/DRCD) and put the json files in "drcd" folder
```
mbart_qg
│───DRCD
│   │   DRCD_dev.json
│   │   DRCD_test.json
│   │   DRCD_training.json
│   README.md
│   config.py
│   ...
```
## set training parameters
set Hyperparameter in config.py

## training
run
>python train.py

train.py will read data from DRCD_dev.json and DRCD_training.json
and concatenate the context and answer into model input
label is the question in dataset.
the model for every epoch will save in "model" folder

## inference
run
>python predict.py

predict.py will load the "best_model" set in 
then input context and question

## Evaluation
run
>python rouge_test.py

rouge_test.py will read DRCD_test.json and generate question.
Then calculate ROUGE-L score of generate question and origi


