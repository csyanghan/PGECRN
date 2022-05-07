# Principal Graph Embedding Convolutional Recurrent Network for Traffic Flow Prediction

## Structure:

* data: including PEMSD4 and PEMSD8 dataset used in our experiments, which are released by and available at  [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data).

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our PGECRN model

## Requirements

Python 3.6.5, Pytorch 1.1.0, Numpy 1.16.3, argparse and configparser

To replicate the results in PEMSD4 and PEMSD8 datasets, you can run the the codes in the "model" folder directly. If you want to use the model for your own datasets, please load your dataset by revising "load_dataset" in the "lib" folder and remember tuning the learning rate (gradient norm can be used to facilitate the training).

Please cite our work if you find useful.
