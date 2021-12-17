# SARMR

# Overview
Implement for Self-supervised Adversarial Regularization model for Medication Recommendation (SARMR)

This is the implement for our model in the paper Self-supervised Adversarial Regularization model for Medication Recommendation, which aims at recommending medications effectively. 

SARMR obtains informative patterns from raw EHRs for adversarial regularization to shape distributions of patient representations for Drug-Drug Interaction (DDI) reduction, and such a self-supervised adversarial regulrization requires no extra external knowledge about DDI. SARMR firstly obtains temporal information from historical admissions, and builds a key-value memory network with patient representations and corresponding medications. Then SARMR carries out multi-hop reading on the memory network to model interactions between patients and physicians while a graph neural network is used to embed the results into meaningful embeddings. Meanwhile, SARMR regulate the distribution of patient representations with a Generative Adversarial Network (GAN) to match it to a desired Gaussian distribution for DDI reduction.

# Requirment
Pytorch 1.1

Python 3.7

# Data

Experiments are carried out based on [MIMIC-III](https://mimic.physionet.org)， which is a real-world Electoric Healthcare Records (EHRs) dataset, and it collects clinical information related to over 45,000 patients. The diagnoses and procedures are used as inputs of SARMR, and the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

To prepare the datasets, get the following three tables from [MIMIC-III](https://mimic.physionet.org):

PRESCRIPTIONS.csv

DIAGNOSES_ICD.csv

PROCEDURES_ICD.csv

Then prepare the following datasets:

ndc2rxnorm_mapping.txt

ndc2atc_level4.csv

drug-atc.csv

drug_stitch2atc.csv

drug-DDI.csv

you can find the first four datasets in the file data, and drug-DDI.csv could be downloaded [here](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0)

put all these datasets in a file named as "data" under your Python project, and run PrepareDataset.py to get all the files.

# Code

Auxiliary.py: data loader

Networks.py: encoder(generator), decoder, and discriminator.

Optimization.py: basic modules that warp encoder and decoder for hyper-parameter tuning

MedRecOptimization.py: hyper-parameter tuning for MedRec, i.e. medication recommendation without GAN regularization.

DiscriminatorOptimization.py: hyper-parameter tuning for the discriminator.

Training.py: model training.

Evaluation.py: model evaluation.

PrepareDataset.py: prepare all the files you need to conducts experiments.

Parameters.py: global parameters for model.

After preparing all the required datasets:

(1) run the function MedRecTraining in Training.py to train SARMR with records of patients whose DDI rates are smaller then the preset threshold, you can find the saved models in the directory specified by "save_model_dir";

(2) run the function fit_distribution_from_encoder in Training.py to fit a Gaussian distribution based on the model obtained in (1), you can find the data sampled from the distribution in the directory specified by "real_data_save_path", which would act as real data to train the GAN model;

(3) run the function MedRecGANTraining in Training.py to train SARMR, you can find the model in the directory specified by "save_model_path".

# Hyper-parameters
We use bayesian optimization to tune hyper-parameters, and conduct the optimization using [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) speficially. Lists of all final parameters as well as range of values tried per parameter during development could be found in dictionary Hyper-parameters.


