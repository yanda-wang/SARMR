# ARMGA

# Overview
Implement for Adversarially Regularized Multi-hop Medication Recommendation with Graph Augmentation

This is the implement for our model in the paper Adversarially Regularized Multi-hop Medication Recommendation with Graph Augmentation, which aims at recommending medications effectively. ARMGA firstly obtains temporal information from historical admissions, and builds a key-value memory network with patient representations and corresponding medications. Then ARMGA carries out multi-hop reading on the memory network to model interactions between patients and physicians while a graph neural network is used to embed the results into meaningful embeddings. Meanwhile, ARMGA regulate the distribution of patient representations with a Generative Adversarial Network (GAN) to match it to a desired Gaussian distribution for DDI reduction

# Requirment
Pytorch 1.1

Python 3.7

# Data

Experiments are carried out based on [MIMIC-III](https://mimic.physionet.org)， which is a real-world Electoric Healthcare Records (EHRs) dataset, and it collects clinical information related to over 45,000 patients. The diagnoses and procedures are used as inputs of ARMR, and the medications prescribed in the first 24 hours of each admission are selected out as ground truths.

Patient records are firstly selected out from the raw data into a file, and each line contains the information for a single admission in the form of \[subject_id, hadm_id, admittime, medications, diagnoses, procedures\].You could find an example below.

\[17, 194023, 2134-12-27 07:15:00, A12A; C01C; B05C; N07A; A12C; A07A; N01A; C02D; M01A; A10A, 2724; 45829; 7455; V1259, 3571; 3961; 8872\]

After constructing the vocabulary for medical concepts, i.e., assigning a identical integer to each medical concepts, medications, diagnoses, and procedures are represented by corresponding integers, and patient records are transformed into a np.array, while each element in the array represents information for a single patient in the form \[adm_1, adm_2, ..., adm_n\]. For each adm_i, the form is \[\[med_1, med_2, ..., med_m\],\[diag_1, diag_2, ..., dig_d\],\[pro_1, pro_2, ..., pro_p\],\[ddi rate\]]. For instance, we could find an example below, and there are two admissions in the example. For the first admission, the medications are 3, 4, and 5, the diagnoses are 6 and 7, the procedures are 8 and 9, and the ddi rate is 0.3.

\[\[\[3, 4, 5\], \[6, 7\], \[8, 9\], \[0.3\]\], \[\[10 ,11\], \[12, 13, 14\], \[15, 16\], \[0.2\]\]\]

# Code

Auxiliary.py: data loader and data preprocessing.

Networks.py: encoder(generator), decoder, and discriminator.

Optimization.py: basic modules that warp encoder and decoder for hyper-parameter tuning

MedRecOptimization.py: hyper-parameter tuning for MedRec, i.e. medication recommendation without GAN regularization.

DiscriminatorOptimization.py: hyper-parameter tuning for the discriminator.

Training.py: model training.

Evaluation.py: model evaluation.

Parameters.py: global parameters for model.
