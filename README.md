![Logo](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a pipeline using unsupervised pre-training for learning chemical structures and leveraging network science methods to predict protein-ligand binding using Deep Learning. 

## Usage

Use the Python notebook VecNet-User-Frontend.ipynb to make binding predictions on unseen protein targets and both unseen ligands and targets.

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

#### DeepPurpose and Confuguration Model:

DeepPurpose Rerun - Transformer CNN.ipynb: We retrain DeepPurpose here using BindingDB data. Muleiple experiments on DeepPurpose have been carried out here, which includes randomly assigning chemical structures and degree analysis of DeepPurpose performance.

Configuration Models on DeepPurpose data.ipynb: We explore the performance of duplex configuration model on DeepPurpose train-test datasets.

#### VecNet

VecNet-Uneen_Targets.ipynb: We create the network-derived negatives, execute a 5-fold cross-validation and make predictions on SARS-CoV-2 genes using VecNet.

VecNet-Uneen_Nodes.ipynb: We execute a 5-fold cross-validation on unseen nodes here.

#### VAENet

VAENet-Uneen_Targets.ipynb: We create the network-derived negatives and execute a 5-fold cross-validation using VAENet.

VAENet-Uneen_Nodes.ipynb: We execute a 5-fold cross-validation on unseen nodes here.

#### Configuration Model - 5 fold

Configuration Model - Cross-Validation.ipynb: We execute a 5-fold cross-validation over unseen taregts and nodes on duplex configuration model using our data.

#### DeepPurpose - 5 fold

Deep Purpose - Final DataSet - Unseen Targets.ipynb: We execute a 5-fold cross-validation over unseen taregts on DeepPurpose using our data.

Deep Purpose - Final DataSet - Unseen Nodes.ipynb: We execute a 5-fold cross-validation over unseen nodes on DeepPurpose using our data.

# Data

Please downlad the related data from: https://www.dropbox.com/work/Foodome%20Team%20Folder/Chatterjee%2C%20Ayan/AI-Bind-Release/data/sars-busters-consolidated













