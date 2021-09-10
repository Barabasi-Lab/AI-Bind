![Logo](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a pipeline which uses unsupervised pre-training for learning chemical structures and leverages network science methods to predict protein-ligand binding using Deep Learning. 

## Usage

Use the Python notebook VecNet-User-Frontend.ipynb to make binding predictions on unseen ligands and targets.

## Data

Please download the data files from: https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADam6kAMLZ3vl-cRfjo6Cn5a?dl=0

## Shortcomings of existing deep models

Stat-of-the-art machine learning models like DeepPurpose (e.g., Transformer-CNN) used for drug repurposing learn the topology of the drug-target interaction network. Much simpler network model (configuration model) can acheive a similar test performance.

While predicting potential drugs for novel targets, drugs with more binding annotations in the training data appear more often. These models lack the ability to learn structural patterns of proteins and ligands, fixing which would enable ML models in the exploratory analysis over new targets.

The binding prediction task can be classified into 3 types: 

i) Unseen edges: When both ligand and target from the test dataset are present in the training data

ii) Unseen targets: When only the ligand from the test dataset is present in the training data

iii) Unseen nodes: When both ligand and target from the test dataset are absent in the training data

Despite performing well in predicting over unseen edges amd unseen targets, existing ML models and network model performs poorly on unseen nodes. 

AI-Bind learns from the chemical structures instead of learning the network topology and enables the binding exploration for new structures.

## Data preparation

Existing ML models for binding prediction use databases like DrugBank, BindingDB, Tox21, or Drug Target Commons. These datasets have a large bias in the amount of binding and non-binding information for ligands and targets.

The ML models learn the DTI network topology and use the degree information of the nodes while predicting. Many nodes having only positive or only negative pairs end up being predicting as always binding or always non-binding. 

We use network distance to generate negative samples for the nodes in DTI network which creates balance between the amount of positive and negative interactions for each ligand and target.

## Unsupervised Pre-training

Current protein-ligand binding prediction infrastructures train the deep models in an end-to-end fashion. This makes the task of generalizing to new chemical structures difficult. 

AI-Bind uses chemical embeddings trained on datasets way larger than only the binding dataset. This allows AI-Bind to extract meaningful structural patterns for completely new ligands and proteins. 

## VecNet

The primary ML model used is VecNet, which uses Mol2vec and ProtVec to embed the ligands and the proteins respectively. These embeddings are fed into a dense layer which acts as the decoder predicting the binding probability.

![VecNet](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/VecNet.PNG)

# Setup

Getting started using a conda env:

```shell
conda create --name ai-bind python=3.6
conda activate ai-bind
```

Installing packages:

```shell
pip install -r requirements.txt
```

# Files

The repository contains multiple folders: 

#### AIBind

AIBind.py: Contains the Python class for AI-Bind. Includes all the ML models. 

import_modules.py: Contains all the necessary Python modules to run AI-Bind models. 

#### Configuration Model - 5 fold

Configuration Model - Cross-Validation.ipynb: We execute a 5-fold cross-validation over unseen taregts and nodes on duplex configuration model using our data.

#### DeepPurpose - 5 fold

Deep Purpose - Final DataSet - Unseen Targets.ipynb: We execute a 5-fold cross-validation over unseen taregts on DeepPurpose using our data.

Deep Purpose - Final DataSet - Unseen Nodes.ipynb: We execute a 5-fold cross-validation over unseen nodes on DeepPurpose using our data.

#### DeepPurpose and Confuguration Model

DeepPurpose Rerun - Transformer CNN.ipynb: We retrain DeepPurpose here using BindingDB data. Multiple experiments on DeepPurpose have been carried out here, which includes randomly assigning chemical structures and degree analysis of DeepPurpose performance.

Configuration Models on DeepPurpose data.ipynb: We explore the performance of duplex configuration model on DeepPurpose train-test datasets.

runscriptposneg.m: Runs the network configuration model using ligand and target degree sequences. 

#### EigenSpokes

Eigen Spokes Analysis.ipynb - Runs Eigen Spokes analysis on combined adjacency matrix. 

#### Random Input Tests

VecNet-Uneen_Nodes-RANDOM.ipynb: Run VecNet on unseen nodes where ligand and target embeddings are replaced by Gaussian random inputs.

VecNet-Uneen_Nodes-T-RANDOM-Only.ipynb: Run VecNet on unseen nodes where target embeddings are replaced by Gaussian random inputs.

VecNet-Uneen_Targets-RANDOM.ipynb: Run VecNet on unseen targets where ligand and target embeddings are replaced by Gaussian random inputs.

VecNet-Uneen_Targets-T-RANDOM-Only.ipynb: Run VecNet on unseen targets where target embeddings are replaced by Gaussian random inputs.

#### Siamese

Siamese_Uneen_Targets.ipynb: We create the network-derived negatives and execute a 5-fold cross-validation using the Siamese model.

Siamese_Uneen_Nodes.ipynb: We execute a 5-fold cross-validation on unseen nodes here.

#### VAENet

VAENet-Uneen_Targets.ipynb: We create the network-derived negatives and execute a 5-fold cross-validation using VAENet.

VAENet-Uneen_Nodes.ipynb: We execute a 5-fold cross-validation on unseen nodes here.

#### Validation

SARS-CoV-2 Predictions Analysis VecNet.ipynb: Auto docking validation of top and bottom 100 predictions made by VecNet on SARS-CoV-2 and related human genes.

#### VecNet

VecNet-Uneen_Targets.ipynb: We create the network-derived negatives, execute a 5-fold cross-validation and make predictions on SARS-CoV-2 genes using VecNet.

VecNet-Uneen_Nodes.ipynb: We execute a 5-fold cross-validation on unseen nodes here.













