# Traffic-Sign-Classification

This is an attempt to forecast the closing price of a stock given its previous closing prices.

This repository contains:

1. The [python code](Traffic_Sign_Classifier.py)
2. The [GUI Outputs](GUI.jpg)
3. [Train dataset .csv file](Train.csv)
4. [Train dataset](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/tree/main/Train)
5. [Test dataset .csv file](Test.csv)
6. [Test dataset](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/tree/main/Test)
7. [Input dataset](Input) for GUI
8. [ReadMe file](README.md) itself


## Table of Contents

- [About](#about)
- [GUI Output](#gui-output)
- [To Run](#to-run)


## About

The dataset used in this repo is the historical data of [Tata Consultancy Services](https://finance.yahoo.com/quote/TCS.NS/history?period1=1029110400&period2=1637971200&interval=1d&frequency=1d&filter=history) (a company listed in National Stock Exchange, India) from August 12, 2002 to November 26, 2021. The dataset was scraped from [Yahoo Finance](https://finance.yahoo.com) using Selenium. The data from Selenium is parsed using BeautifulSoup and converted to their respective datatypes before storing it in the Pandas dataframe.


## GUI Output
![Traffic Sign Classification GUI](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/blob/main/GUI.png "Title")


## To Run

Download the [Train dataset](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/tree/main/Train) and [.csv file](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/blob/main/Train.csv), [Test dataset](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/tree/main/Test) and [Test dataset](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/blob/main/Test.csv), [Input dataset](https://github.com/Viknesh-Rajaramon/Traffic-Sign-Classification/tree/main/Input) and save the downloaded folders in a folder titled 'omniglot'. Save the [python code](BlackBox.py) and [config file](config.json) in the same directory of 'omniglot'.


```

BlackBox.py
config.json
omniglot
│___  images_background
│___  images_evaluation
omniglot
│___  images_background
│___  images_evaluation
omniglot
│___  images_background
│___  images_evaluation

Make sure you have the following libraries installed to run the code.
```
numpy
pandas
matplotlib
scikit-learn
keras
tkinter
pillow
```

Download the following files in the same directory.
```
Code.py 
```

Run Traffic_Sign_Classifier.py
```
cd directory-where-the-files-are-saved
python3 Traffic_Sign_Classifier.py
```


## Team Members

Rithic Kumar
<br>
Sreedhar Arumugam
<br>
[Shresta M](https://github.com/shresta-m/Traffic_sign_classification)
<br>
R Shreja


# Black-Box-Meta-Learning

This repo implements and trains a memory augmented neural networks, a black-box meta-learner that uses a recurrent neural network for few shot classification

This repository contains:

1. The [python code](BlackBox.py)
2. The [config file](config.json)
3. [CS330 HW1 file](CS330_HW1.pdf)
4. And the [ReadMe file](README.md) itself

## Table of Contents

- [About](#about)
- [To Run](#to-run)
- [References](#references)


## About

### Dataset
The [Omniglot data](https://github.com/brendenlake/omniglot) set is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people. The Omniglot data set contains 50 alphabets. It is split into a background set of 30 alphabets and an evaluation set of 20 alphabets.

### Model
A stacked 2 layered-LSTM model is employed. The inputs from the support set are concatenated with their true lables one-hot encoded. Where as the inputs from the query set are concatenated with all zeroes. The model is expected to predict the true labels of the query set. Shown below is a stacked LSTM model. <br>
![demo](architecture.png) <br>
<br> More information on the training procedure could be found in [HW1 of CS330](https://github.com/siddarth-c/Black-Box-Meta-Learning/blob/main/CS330_HW1.pdf). The hyper-parameters can be changed in the [config file](config.json). <br>

##### Update 1 (05-08-2021) : Included support for Bidirectional-LSTM. Change 'bi_dir' to "true" in the config file to enable BiLSTM.

## To Run

Download the omniglot data [here](https://www.kaggle.com/watesoyan/omniglot/download) and save the downloaded folders in a folder titled 'omniglot'. Save the [python code](BlackBox.py) and [config file](config.json) in the same directory of 'omniglot'.


```

BlackBox.py
config.json
omniglot
│___  images_background
│___  images_evaluation    

```
Install the following libraries to run the code
```
torch
numpy
glob
PIL
matplotlib
```
Run BlackBox.py 
```
python3 BlackBox.py
```