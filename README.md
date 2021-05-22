DDatingApp (Deep Neural + Dating App)
======

DDatingApp is a project which aims to develop a NeuralNetwork which aim to "swipe" on people who you might like.

This project was developed using TensorFlow + Python.

It uses the SCUT-FBP5500 dataset for facial beauty prediction (many thanks to them!).

Background
------

Covid hit really hard, most of my day I spent coding but at the night I really wanted someone to talked, sadly, because of my work I can't really take my time to swipe on people on dating apps.

To be honest this was going to be a personal never to be published, but thanks to teacher it became an academic project, I don't expect to ever update this repo. 

Requirements
------

### Hardware
* More than 32GB of RAM
* Nvidia GPU

### Python Version
* Python 3.8.5

### Python Libraries
* TensorFlow
* Pickle
* Numpy
* PIL
* CV2
* zipfile
* gdown
* Jupyter
* pidgeon

I highly recommend using Anaconda since it comes with several libraries needed for the project

Training
------

### Downloading and preparing data

This project already comes with a tool to prepare data for the neural network, you can call it with from the root of the project:

```
(PS) > python .\neuralnetwork\prepareData.py
```

This script downloads and (obviously) prepares the data, you can add more training data using my own tool to label.

### Making more training data

This project aims to localize the NN to be used (virtually anywhere), the SCUT dataset is very limited to asian and white people, we wanted to add more diversity to the dataset, but we ended up deciding to make a tool for others to use. 

#### Scrap your images

You got a cool bot that can scrape thousands or hundreds of profiles? if yes then you are in luck, just make a python file with the name "data.py" inside the "classifier" folder and add your URLs inside an array call "data"

```
data = [...]
```

#### Downloading and Filtering images

Use downloadHelper.py and classifier.py to download then split into batches your photos

```
(PS) /classifier> downloadHelper.py
(PS) /classifier> classifier.py
```

This will produce a labelx.txt just load them using the "class" jupyter notebook, it has a really nice interface to label fast.

#### Adding them to the NN

After labeling your dataset, run "mergeLabels.py"

```
(PS) /classifier> mergeLabels.py
```

This will output a file call "User_labels.txt", copy this file to the SCUT dataset at "/SCUT-FBP5500_V2/train_test_files/", dont forget to add your photos in the Images folder of the SCUT dataset!

### Train NN

You have to ways of training the NN, using a python script or the jupyter notebook, both are same it is just preference

The files are in:

* /neuralnetwork/train.py
* /neuralnetwork/train.ipynb

Once your training has finished you need to copy .h5 to the common folder

## Running the bot

### Offline testing

If you don't want to use a bot and just want to use performance of the trained NN you can use:

* /neuralnetwork/test.py
* /tinderbot/offline.py

#### test.py

This script measures the perfomance overall of the NN, it uses the same dataset as training

#### offline.py

This script just predicts images from an array, useful to only predict a handful of images.

### Bot

Depending on your needs you can use a WebDriver or making API calls directly to tinder, it depends on how brave your are

#### Using Sellenium

Only SMS login is enabled, popus has not been handled

Run botS.py 

```
(PS) /tinderbot> botS.py 
```

#### Using Unofficial API

Needs functional token to work, you can extract it from a live session inside Tinder.

Run boyApi.py

```
(PS) /tinderbot> boyApi.py
```