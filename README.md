# DeepFM_perfume
perfume recommendation using DeepFM

## Overall Flow
![image](https://user-images.githubusercontent.com/89527573/209938296-52031e64-dffa-40b8-8af5-4707912e2661.png)

## Data
-----
Data is crawled from parfumo.com.

## Code Description
config.py 
-----
contains essential informations to run codes such as file paths, column informations.

pre.py 
-----
creates dataset for train/predict task.
Task is paralleled using joblib.
For train  -> python pre.py train 
For predict -> python pre.py predict

rating.py 
-----
gathers rating informations frow website.

info.py
-----
gathers perfume informations from website.

clear.py 
-----
clears chrome caches

deepfm.py
-----
model class

train.py 
-----
train model. Can train model for 2 different task. 
For regression -> python train.py regression. 
For classification -> python train.py classification
You can see your training log at the log folder

optimization.py 
-----
Find the best seed for model. 
You can see your training log at the log folder

predict.py
-----
predict perfumes using pretrained model. Only update when the user's update status is 1.
Full shows predict output, rank and perfume index. Short only shows perfume index.
For Full -> python predict.py full
For Short -> python predict.py short

----

