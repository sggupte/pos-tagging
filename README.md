# Parts of Speech Tagging

**The repo is broken down into two parts: one section for the discriminative model, and one for the generative model.**

Type the following code to load install the correct dependencies into your pip virtual environment.

`
pip install -r requirements.txt
`

## Generative Model
All the code used to produce the results for the generative model are under the Generative Folder. Please run scripts from the main repository folder so that files save in the correct location. These are the scripts included.

* hmmModel.py -- main HMM model that is referenced in both textGenerator and Viterbi
* textGenerator.py -- Generates synthetic corpus and saves it in the out folder
* viterbi.py -- Runs the viterbi algorithm and saves the confusion matrices in the out folder

## Discriminative Model
All code used to produce the results for the discriminative neural network are under the Discriminative folder.
There are three different scripts under this folder:

* utils.py -- This script contains all the auxiliary functions used for setting up the dataset, dataloader, etc.
* nn_training.py -- This script contains everything used to train the neural network
* architecture.py -- This script defines the model architecture for the neural network.

The code can be run through the command line, or copy-pasted and used in a notebook format (especially useful if you need to use Google Colab or something with computational power)
