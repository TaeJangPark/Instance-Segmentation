# Instance-Segmentation
This is an implementation of Mask R-cnn on Python 2.7 and Tensorflow. The model generates bounding boxes and segmentation masks for each Instance of an object in the image. It's based on [this code](https://github.com/matterport/Mask_RCNN) and I use MobileNet instead of ResNet101 as a backbone.

The repository includes:
* Source code of Mask R-cnn built on FPN and MobileNet.
* Training code for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step.
* Multi-GPU training.

# Getting Started
* ([model.py](/libs/nets/model.py), [utils.py](utils.py), [config.py](/libs/configs/config.py)): These files contain the main Mask RCNN implementation. 

* [Training_Data_Test.ipynb](/Training_Data_Test.ipynb). This notebook visualizes the different pre-processing steps
to prepare the training data.

* [inspect_model.ipynb](/inspect_model.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

# Training on MS COCO or your Own Dataset
you need MS COCO dataset and put the dataset in data folder, if you want to train this model.
To train the model on your own dataset, you will need to extend two classes:

'''Config''' 
This class contains the default configuration. Subclass it on modity the attributes you need to change.

'''Dataset''' 
This model allow you to use new datasets for training without having to change the code. The '''Dataset''' class it self is the base class. To use it, create a new class that inferits from it and adds functions specific to your dataset.
