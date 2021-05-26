![Logo](https://github.com/ChatterjeeAyan/AI-Bind/tree/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a pipeline using unsupervised pre-training for learning chemical structures and leveraging network science methods to predict protein-ligand binding using Deep Learning. 

## Shortcomings of existing deep models

Stat-of-the-art machine learning models like DeepPurpose (e.g., Transformer-CNN) used for drug repurposing learn the topology of the drug-target interaction network. Much simpler network model (configuration model) can acheive a similar test performance.

While predicting potential drugs for novel targets, drugs with more binding annotations in the training data appear more often. These models lack the ability to learn structural patterns of proteins and ligands, fixing which would enable them for the exploratory analysis over new targets.

The binding prediction task can be classified into 3 types: 

i) Unseen edges: When both ligand and target are present in the training data

ii) Unseen targets: When only the ligand is present in the training data

iii) Unseen nodes: When both ligand and target are absent in the training data

Despite performing well in predicting over unseen edges amd unseen targets, existing ML models and network model performs poorly on unseen nodes. 

AI-Bind learns from the chemical structures instead of learning the network topology and enables the binding exploration for new structures.

## Data preparation

Existing ML models for binding prediction use databases like DrugBank, BindingDB, Tox21, or Drug Target Commons. These datasets have a large bias in the amount of binding and non-binding information for ligands and targets.

The ML models learn the DTI network topology and use the degree information of the nodes while predicting. Many nodes having only positive or only negative pairs end up being predicting as always binding or always non-binding. 

We use network distance to generate negative samples for the nodes in DTI network which creates balance between the amount of positive and negative interactions for each ligand and target.

## Unsupervised Pre-training

Current protein-ligand binding prediction tasks train the deep models in an end-to-end fashion. This makes the task of generalizing to new chemical structures difficult. 

AI-Bind uses chemical embeddings trained on datasets way larger than only the binding dataset. This allows AI-Bind to extract meaningful structural patterns from completely new ligands and proteins. 

## VecNet

The primary ML model used is VecNet, which uses Mol2vec and ProtVec to embed the ligands and the proteins respectively. These embeddings are fed into a dense layer which acts as the decoder predicting the binding probability.

![VecNet](https://github.com/ChatterjeeAyan/AI-Bind/tree/main/Images/VecNet.png)

# Setup

# Files

# Data














