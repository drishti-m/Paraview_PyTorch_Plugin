# Plugins for data-driven filters in Paraview
## Table of Contents

* [About the Project](#about-the-project)
    * [Done By](#done-by)
* [Instructions for installation](#instructions-for-installation)
* [Instructions to use filter plugin in Paraview](#instructions-to-use-filter-plugin-in-paraview)
* [Instructions to download & train the models](#instructions-to-download-&-train-the-models)
* [Example Results](#example-results)
* [Directory Structure](#directory-structure)


## About the Project
To couple the interface of **[Paraview](https://www.paraview.org/)**, *a visualization application*, and **[PyTorch](https://pytorch.org/)**, *a machine learning framework*, I have developed plugins in Paraview that allow users to load *pre-trained models* of their choice.  Pre-trained means that machine learning models have been already trained on some form of dataset and carry the resultant weights and biases. These plugins transform the input data by feeding it into the model and then visualize the model’s output. The filters builtin Paraview,  till the time of writing,  are based on deterministic algorithms to modify or extract slices of input data. These plugins extend the filters in Paraview to include algorithms based on data-driven transformations. This repository contains four plugins for the use cases: *image segmentation, image classification, fluid classification & fluid segmentation* respectively.
### Done By - 
[**Drishti Maharjan**](https://github.com/drishti-m) - d.maharjan@jacobs-university.de 


## Instructions for installation

*[Note to supervisor]: The installation of Paraview with Python support have been already set up in Titlis. The executable of Paraview is in* `/home/dmaharjan/paraview/paraview_build/bin`. Below are the instructions if you wish to build from scratch.

The plugins have been developed with *Paraview `5.8.1`* built against *Python `3.7.4`*. It should generally work for Python versions 3.3+. 
Follow instructions to build Paraview [here](https://gitlab.kitware.com/paraview/paraview/blob/master/Documentation/dev/build.md). Make sure to turn on `PARAVIEW_USE_PYTHON` option while building it to enable
Python support.

For installation of other packages and libraries:
```
conda create --name <env> python=3.7.4 --file requirements.txt
conda activate <env>
pip install -r pip-requirements.txt
```
Some packages are not available in conda so those packages are listed in pip-requirements and are to be 
installed via pip.


## Instructions to use filter plugin in Paraview
1. In the conda environment you built, open Paraview
2. Go to `Tools -> Manage Plugins -> Load New`.
3. Find the location of the code of plugins in your disk (here, inside `plugin-src/` directory)
4. Load all plugins required, eg, `plugin-src/fluid_classifier_plugin.py`
5. Open you source input file in Paraview by `File -> Open`.
6. After loading your input source, Go to `Filters -> Search`.
7. Type the name of the filter (named below), and give the user parameters and 'Apply' the filter.

The main plugins in this repository are named as follows:
* ML_Fluid_Classifier
* ML_Fluid_Segmentation
* ML_Img_Classifier
* ML_Img_Segmentation

The test plugins are to see the ground truth results of computed fluid classification and segmentation, and are used to compare with the ML based classification & segmentation. The test plugins in this repository are named as follows:
* Threshold_Fluid_Classify
* Threshold_Fluid_Segment

**Example: User parameters for ML_Fluid_Segmentation**:

1. *Trained Model's Path*: 
`./thesis-code/pre-trained-models/fluid-segment/fluid-segment(velocity).pth`
2. *Model's Class Defn Path*: `./thesis-code/pre-trained-models/fluid-segment/NN_fluid_segment.py`

**Example: User parameters for ML_Image_Classification**:
1. *Trained Model's Path*: 
`./thesis-code/pre-trained-models/img-classify/alexnet.pth`
2. *Model's Labels Path*: `./thesis-code/pre-trained-models/img-classify/imagenet_classes.txt`
 
*Note: The path above should be relative to the location of executable of Paraview, or absolute path from home.*


## Instructions to download & train the models
The pre-trained models for fluid classification & segmentation are already provided in the directory `pre-trained-models`. The pre-trained models for image classification & segmentation need to be downloaded from PyTorch. Run the script below.
```
cd pre-trained-models
python download_models.py
```
The downloaded models will be saved at `pre-trained-models`.

If you wish to train the model for fluid classification and segmentation, follow the instructions below.

```
cd model-training
python train_fluid_classifier.py
python train_fluid_segment.py
tensorboard --logdir runs/
```
Your training models will be saved at `model-training/models`.



## Example Results

The *"Surface view"* results of applying Fluid Segmentation Plugin for velocity data of a given timestep in Paraview are below.

**Original Velocity Magnitude View in Paraview**:

![Original](./results/velocity_R10-frame0.png )

**Above data Segmented by Plugin in Paraview**:

![Segmented](./results/seg-velocity_R10_frame0.png)

## Directory Structure
```
├── model-training                         # training of models
│   ├── datasets                           # training data 
│   │   ├── pressure                       # pressure data 
│   │   └── velocity                       # velocity data
│   ├── models                             # directory to save models
├── plugin-src                             # main directory with source code of plugins
│   ├── fluid_classifier_plugin.py         # plugin for fluid classification
│   ├── fluid_segment_plugin.py            # plugin code for fluid segmentation
│   ├── img_classifier_plugin.py           # plugin code for image classification
│   ├── img_segment_plugin.py              # plugin code for image segmentation
├── plugin-test                            # source code for test plugins (ground truth)
├── pre-trained-models                     # all pre-trained models 
│   ├── fluid-classify                     # pre-trained models for fluid classification
│   ├── fluid-segment                      # pre-trained models for fluid segmentation
│   ├── img-classify                       # pre-trained models for img classification
│   └── img-segment                        # pre-trained models for img segmentation
├── README.md                              # readme
└── requirements.txt                       # package requirements by conda
├── pip-requirements.txt                   # package requirements by pip
└── results                                # some resulting examples of plugins application

```