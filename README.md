![Logo](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a deep-learning pipeline designed to provide reliable binding predictions for poorly annotated/unseen ligands  and proteins. Indeed, in recent years, deep-learning models have become progressively more popular in drug discovery, as they can offer rapid screening for large libraries of proteins and ligands, guiding the computationally expensive auto-docking simulations on selected pairs needing more accurate validation. Like many algorithms currently in the scientific literature, AI-bind leverages simple features encoded in the amino-acid sequence of a protein and in the isomeric SMILE of a ligand, known to drive the binding mechanism. The minimal structural information necessary to run the algorithm circumvents the general lack of available protein 3D structures. 

## Setting up AI-Bind and Predicting Protein-Ligand Binding (Guidelines for end users) 

1. Download the docker file named "Predictions.dockerfile".
2. On your terminal, move to the directory with the dockerfile and run : 
	docker build -t aibindpred -f ./AIBind_Predict_v2.dockerfile ./
3. To run the image as a container: 
	docker run -it --gpus all --name aibindpredcontainer -p 8888:8888 aibindpred

	You may clone the git repository inside the container, or attach your local volume while running the container :
	docker run -it --gpus all --name aibindpredcontainer -p 8888:8888 -v ./local_directory:/home aibindpred
4. To execute additional shells inside the container, run : 
	docker exec -it aibindpredcontainer /bin/bash
5. To run a Jupyter notebook instance inside the container, run :
	jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
The steps above will install all necessary packages and create the environment to run binding predictions using AI-Bind.
6. Organize your data file in a dataframe format with the colulmns 'InChiKey', 'SMILE' and 'target_aa_code'. Save this dataframe in a .csv file. 
7. Run the notebook titled VecNet-User-Frontend.ipynb to make the binding predictions. Predicted binidng probabilites will be available under the column header 'Averaged Predictions'.

# Why AI-Bind? 

## Shortcomings of Existing ML Models in Predicitng Protein-Ligand Binding

Our interest poorly annotated proteins and ligands, especially foodborne natural compounds, pushed our research team to evaluate the inductive test performance of several models, i.e., how well the algorithms like DeepPurpose perform when predicting binding for never-before-seen ligands and never-before-seen proteins. Indeed, only inductive test performance is a reliable metric to evaluate how well a model has learned the binding patterns encoded in the structural features describing proteins and ligands, and quantify its ability to generalize to novel structures. Unfortunately, our literature review showed how many models present mainly transductive test performance, measuring how well the algorithms predict unseen binding between known structures. We analytically derived how excellent performances in transductive tests can be achieved even with simple algorithms that do not require deep learning, and completely disregard the structural information characterizing proteins and ligands.
We present these configuration models, inspired by network science, as the baseline of any ML model implemented in AI-Bind. Their success in transductive test performance is driven by the underlying network structure contributing to the most used training datasets, which present an extremely biased picture of the binding classification task, with structures with a predominantly higher number of positive annotations (binding) compared to negative (non-binding), and vice-versa. In a scenario affected by annotation imbalance, AI models behave similarly to configuration models, disregarding structural information, and failing to perform well in inductive tests.

## What does AI-Bind offer?

We developed the AI-Bind pipeline with the goal to maximize inductive test performance. First, we mitigated annotation imbalance by including in the training set both positive and negative examples for each protein and ligand, balancing the exceeding number of positive annotations in the original data with network-derived negatives, i.e., pairs of proteins and ligands with high shortest path distance on the bipartite network induced by all pairs of proteins and ligands for which we collected binding experimental evidence. This step improves the training phase of all ML models, enhancing their ability to learn structural features. Second, by testing different architectures such as VecNet, VAENet, and Siamese, currently available to the user in different python notebooks, we understood how the best generalizing models are not trained end-to-end, but leverage the vectorial representation capturing the salient structural features of molecules and proteins, as learned on wider and more heterogeneous chemical libraries, not filtered according to the current binding evidence. In machine learning jargon, this implies the introduction of unsupervised pre-training of ligand and protein embeddings within the ML architecture.

The best performing architecture in AI-Bind implementation is VecNet, which uses Mol2vec and ProtVec to embed the ligands and the proteins, respectively. These embeddings are fed into a decoder (Multi-layer Perceptron), predicting the binding probability.
![VecNet](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/GitHub_Diagram.png)

## Interpretability of AI-Bind and Identifying Active Binding Sites

We mutate certain building blocks (amino acid trigrams) of the protein structure to recognize the regions influencing the binding predictions the most and identify them as the potential binding sites. Finally, we validate the AI-Bind predicted active binding sites on the human protein TRIM59 by visualizing the results of the auto-docking simulations and mapping the predicted sites to the amino acid residues where the ligands bind. AI-Bind predicted binding sites can guide the users in creating an optimal grid for the subsequent auto-docking simulations, further reducing simulation time. 

![trigram-study](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/trigram-study.png)

# Code and Data

## Data Files

All data files are shared via Dropbox: https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADam6kAMLZ3vl-cRfjo6Cn5a?dl=0

/data/sars-busters-consolidated/Database files: Contains protein-ligand binding data derived from DrugBank, BindingDB and DTC (Drug Target Commons). 
/data/sars-busters-consolidated/chemicals: Contains ligands used in training and testing of AI-Bind with embeddings.
/data/sars-busters-consolidated/GitData/DeepPurpose and Configuration Model: Train-test data related to 5-fold cross-validation of Transformer-CNN (DeepPurpose) and the Duplex Configuration Model.
/data/sars-busters-consolidated/GitData/interactions: Contains the network derived negatives dataset used in trainning of AI-Bind neural netoworks. 
/data/sars-busters-consolidated/GitData: Contains trained VecNet model, binding predictions on viral and human proteins associated with COVID-19, and summary of auto-docking simulations. 
/data/sars-busters-consolidated/master_files: Contains the absolute negative (non-binding) protein-ligand pairs used in testing of AI-Bind. 
/data/sars-busters-consolidated/targets: Contains the proteins used in training and testing of AI-Bind with associated embeddings. 
/data/sars-busters-consolidated/interactions: Contains the positive (binding) protein-ligand pairs derived from DrugBank, NCFD (Natural Compounds in Food Database), BindingDB and DTC. 
/data/sars-busters-consolidated/Auto Docking: Contains all files and results from the validation of AI-Bind on COVID-19 related viral and human proteins. 
/data/sars-busters-consolidated/Binding Probability Profile Validation: Contains the files visualizing the active binding sites from auto-dcoking simulations. 
/data/sars-busters/Mol2vec: Pre-trained Mol2vec and ProtVec models are available here. 

## Reproducing results

The repository contains multiple folders: 

### DeepPurpose and Confuguration Model

DeepPurpose Rerun - Transformer CNN.ipynb: We train-test DeepPurpose using the benchmark BindingDB data. Multiple experiments on DeepPurpose have been carried out here, which includes randomly assigning chemical structures and degree analysis of DeepPurpose performance.

Configuration Models on DeepPurpose data.ipynb: We explore the performance of duplex configuration model on DeepPurpose train-test datasets.

runscriptposneg.m: Runs the network configuration model using ligand and target degree sequences. Output files summat10.csv and summat01.csv are then used in calculating the performance of the configuration model.

### AIBind

AIBind.py: Contains the Python class for AI-Bind. Includes all the ML models. 

import_modules.py: Contains all the necessary Python modules to run AI-Bind models. 

### VecNet

VecNet-Unseen_Nodes.ipynb: We create the network-derived negatives, execute a 5-fold cross-validation on unseen nodes, and make predictions on SARS-CoV-2 genes using VecNet.

VecNet-Unseen_Targets.ipynb: We execute a 5-fold cross-validation on VecNet.

### DeepPurpose - 5 fold

Deep Purpose - Final DataSet - Unseen Targets.ipynb: We execute a 5-fold cross-validation over unseen targets on DeepPurpose using the network-derived negatives.

Deep Purpose - Final DataSet - Unseen Nodes.ipynb: We execute a 5-fold cross-validation over unseen nodes on DeepPurpose using the network-derived negatives.

### Configuration Model - 5 fold

Configuration Model - Cross-Validation.ipynb: We execute a 5-fold cross-validation over unseen targets and nodes on duplex configuration model using the network-derived negatives.

### VAENet

VAENet-Unseen_Nodes.ipynb: We create the network-derived negatives and and execute a 5-fold cross-validation on unseen nodes here.

VAENet-Unseen_Targets.ipynb: We execute a 5-fold cross-validation on VAENet.

### Siamese

Siamese_Unseen_Nodes.ipynb: We create the network-derived negatives and and execute a 5-fold cross-validation on unseen nodes here.

Siamese_Unseen_Targets.ipynb: We execute a 5-fold cross-validation on the Siamese model.

### Validation

SARS-CoV-2 Predictions Analysis VecNet.ipynb: Auto docking validation of top and bottom 100 predictions made by VecNet on SARS-CoV-2 proteins and human genes associated with COVID-19.

### EigenSpokes

Eigen Spokes Analysis.ipynb - Runs Eigen Spokes analysis on combined adjacency matrix (square adjancecy matrix with both ligands and targets in rows and columns). 

### Random Input Tests

VecNet-Unseen_Nodes-RANDOM.ipynb: Run VecNet on unseen nodes where ligand and target embeddings are replaced by Gaussian random inputs.

VecNet-Unseen_Nodes-T-RANDOM-Only.ipynb: Run VecNet on unseen nodes where target embeddings are replaced by Gaussian random inputs.

VecNet-Unseen_Targets-RANDOM.ipynb: Run VecNet on unseen targets where ligand and target embeddings are replaced by Gaussian random inputs.

VecNet-Unseen_Targets-T-RANDOM-Only.ipynb: Run VecNet on unseen targets where target embeddings are replaced by Gaussian random inputs.

## External Resources

Learn auto-docking using Autodock Vina: https://www.youtube.com/watch?v=BLbXkhqbebs
Learn to visualize active binding sites using PyMOL: https://www.youtube.com/watch?v=mBlMI82JRfI

# Cite AI-Bind

If you find AI-Bind useful in your research, please consider adding the following citation:

'''
@misc{chatterjee2021aibind,
      title={AI-Bind: Improving Binding Predictions for Novel Protein Targets and Ligands}, 
      author={Ayan Chatterjee and Omair Shafi Ahmed and Robin Walters and Zohair Shafi and Deisy Gysi and Rose Yu and Tina Eliassi-Rad and Albert-László Barabási and Giulia Menichetti},
      year={2021},
      eprint={2112.13168},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
'''