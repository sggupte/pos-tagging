# Parts of Speech Tagging

The repo is broken down into two parts: one section for the discriminative model, and one for the generative model.

All code used to produce the results for the discriminative neural network are under the Discriminative folder.
There are three different scripts under this folder:

* utils.py -- This script contains all the auxiliary functions used for setting up the dataset, dataloader, etc.
* nn_training.py -- This script contains everything used to train the neural network
* architecture.py -- This script defines the model architecture for the neural network.

The code can be run through the command line, or copy-pasted and used in a notebook format (especially useful if you need to use Google Colab or something with computational power)
