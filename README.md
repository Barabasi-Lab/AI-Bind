![Logo](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a pipeline which uses unsupervised pre-training for learning chemical structures and leverages network science methods to predict protein-ligand binding using Deep Learning. 

## Set-up and Usage

Two usecases:

(1) Run predictions  

1. Download the docker file named "Predictions.dockerfile"
2. On your terminal, move to the directory with the dockerfile and run : 
	docker build -t aibindpred -f ./AIBind_Predict_v2.dockerfile ./
3. To run the image as a container, run : 
	docker run -it --gpus all --name aibindpredcontainer -p 8888:8888 aibindpred

	You may clone the git repository inside the container, or attach your local volume while running the container :
	docker run -it --gpus all --name aibindpredcontainer -p 8888:8888 -v ./local_directory:/home aibindpred

4. To execute additional shells inside the container, run : 
	docker exec -it aibindpredcontainer /bin/bash
5. To run a Jupyter notebook instance inside the container, run :
	jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
6. Run the notebook titled VecNet-User-Frontend.ipynb to make the binding predictions


(2) Reproducing the results 

Pull this docker image: https://hub.docker.com/r/omairs/foodome2

This docker image contains all the data files, notebooks, and environments required to reproduce the results. 

## Data

Data files are available here: https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADam6kAMLZ3vl-cRfjo6Cn5a?dl=0

## Description 

### Shortcomings of existing deep models

Stat-of-the-art machine learning models like DeepPurpose (e.g., Transformer-CNN) used for drug repurposing learn the topology of the drug-target interaction network. Much simpler network model (configuration model) can acheive a similar test performance.

While predicting potential drugs for novel targets, drugs with more binding annotations in the training data appear more often. These models lack the ability to learn structural patterns of proteins and ligands, fixing which would enable ML models in the exploratory analysis over new drugs and targets.

The binding prediction task can be classified into 3 types: 

i) Unseen edges (Transductive test): When both ligand and target from the test dataset are present in the training data

ii) Unseen targets (Semi-inductive test): When only the ligand from the test dataset is present in the training data

iii) Unseen nodes (Inductive test): When both ligand and target from the test dataset are absent in the training data

Despite performing well in predicting over unseen edges amd unseen targets, existing ML models and network model performs poorly on unseen nodes. 

AI-Bind learns from the chemical structures instead of learning the network topology and enables the binding exploration for new structures.

### Data preparation

Existing ML models for binding prediction use databases like DrugBank, BindingDB, Tox21, Davis, Kiba, Drug Target Commons etc. These datasets have a large bias in the amount of binding and non-binding information for ligands and targets.

The ML models learn the DTI network topology and use the degree information of the nodes while predicting. Many nodes having only positive or only negative pairs end up being predicting as always binding or always non-binding. 

We use network distance to generate negative samples for the nodes in DTI network which creates balance between the amount of positive and negative interactions for each ligand and target.

### Unsupervised Pre-training

Current protein-ligand binding prediction infrastructures train the deep models in an end-to-end fashion. This makes the task of generalizing to new chemical structures difficult. 

AI-Bind uses chemical embeddings trained on datasets way larger than the binding dataset. This allows AI-Bind to extract meaningful structural patterns for completely new ligands and proteins, beyond those present in training. 

### VecNet

The best performing ML model in AI-Bind implementation is VecNet, which uses Mol2vec and ProtVec to embed the ligands and the proteins, respectively. These embeddings are fed into a dense layer which acts as the decoder, predicting the binding probability.

![VecNet](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/GitHub_Diagram.pdf)

## Reproducing results

The repository contains multiple folders: 

### DeepPurpose and Confuguration Model

DeepPurpose Rerun - Transformer CNN.ipynb: We train-test DeepPurpose using the benchmark BindingDB data. Multiple experiments on DeepPurpose have been carried out here, which includes randomly assigning chemical structures and degree analysis of DeepPurpose performance.

Configuration Models on DeepPurpose data.ipynb: We explore the performance of duplex configuration model on DeepPurpose train-test datasets.

runscriptposneg.m: Runs the network configuration model using ligand and target degree sequences. Output files summat10.csv and summat01.csv is then used in calculating the performance of the configuration model.

### AIBind

AIBind.py: Contains the Python class for AI-Bind. Includes all the ML models. 

import_modules.py: Contains all the necessary Python modules to run AI-Bind models. 

### VecNet

VecNet-Uneen_Nodes.ipynb: We create the network-derived negatives, execute a 5-fold cross-validation on unseen nodes, and make predictions on SARS-CoV-2 genes using VecNet.

VecNet-Uneen_Targets.ipynb: We execute a 5-fold cross-validation on VecNet.

### DeepPurpose - 5 fold

Deep Purpose - Final DataSet - Unseen Targets.ipynb: We execute a 5-fold cross-validation over unseen taregts on DeepPurpose using the network-derived negatives.

Deep Purpose - Final DataSet - Unseen Nodes.ipynb: We execute a 5-fold cross-validation over unseen nodes on DeepPurpose using othe network-derived negatives.

### Configuration Model - 5 fold

Configuration Model - Cross-Validation.ipynb: We execute a 5-fold cross-validation over unseen taregts and nodes on duplex configuration model using the network-derived negatives.

### VAENet

VAENet-Uneen_Nodes.ipynb: We create the network-derived negatives and and execute a 5-fold cross-validation on unseen nodes here.

VAENet-Uneen_Targets.ipynb: We execute a 5-fold cross-validation on VAENet.

### Siamese

Siamese_Uneen_Nodes.ipynb: WWe create the network-derived negatives and and execute a 5-fold cross-validation on unseen nodes here.

Siamese_Uneen_Targets.ipynb: We execute a 5-fold cross-validation on the Siamese model.

### Validation

SARS-CoV-2 Predictions Analysis VecNet.ipynb: Auto docking validation of top and bottom 100 predictions made by VecNet on SARS-CoV-2 proteins and human genes associated with COVID-19.

### EigenSpokes

Eigen Spokes Analysis.ipynb - Runs Eigen Spokes analysis on combined adjacency matrix (square adjancecy matrix with both ligands and targets in rows and columns). 

### Random Input Tests

VecNet-Uneen_Nodes-RANDOM.ipynb: Run VecNet on unseen nodes where ligand and target embeddings are replaced by Gaussian random inputs.

VecNet-Uneen_Nodes-T-RANDOM-Only.ipynb: Run VecNet on unseen nodes where target embeddings are replaced by Gaussian random inputs.

VecNet-Uneen_Targets-RANDOM.ipynb: Run VecNet on unseen targets where ligand and target embeddings are replaced by Gaussian random inputs.

VecNet-Uneen_Targets-T-RANDOM-Only.ipynb: Run VecNet on unseen targets where target embeddings are replaced by Gaussian random inputs.

Queries and suggestions can be addressed to: chatterjee.ay@northeastern.edu











