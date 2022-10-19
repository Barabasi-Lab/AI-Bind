![Logo](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a deep-learning  pipeline that provides interpretable binding predictions for never-before-seen proteins and ligands. AI-Bind is capable of rapid screening of large chemical libraries and guiding computationally expensive auto-docking simulations by prioritizing protein-ligand pairs for validation. The pipeline requires as input simple chemical features such as the amino-acid sequence of a protein and the isomeric SMILE of a ligand, which helps to overcome limitations associated with the lack of available 3D protein structures.

Preprint available at: https://arxiv.org/abs/2112.13168 

# Why AI-Bind? 

## Shortcomings of Existing ML Models in Predicting Protein-Ligand Binding

Our interest in predicting binding for never-before-seen proteins and ligands pushed us in splitting the test performances of the existing machine learning models (e.g., DeepPurpose) into three components:

(a) Transductive test: When both proteins and ligands from the test dataset are present in the training data,

(b) Semi-inductive test: when only the ligands from the test dataset are present in the training data, and

(c) Inductive test: When both proteins and ligands from the test dataset are absent in the training data.

We learn that only inductive test performance is a dependable metric for evaluating how well a machine learning model has learned binding from the structural features of proteins and ligands. We note that the majority of the models mainly present transductive test performance, which is related to predicting unseen links in the protein-ligand interaction network used in training. We explore how ML models achieve transductive performances comparable to much simpler algorithms (namely, network configuration models), which completely ignore the molecular structures and use the degree information to make binding predictions.

## What does AI-Bind offer?

AI-Bind pipeline maximizes inductive test performance by including network-derived negatives in the training data and introducing unsupervised pre-training for the molecular embeddings. The pipeline is validated via three different neural architectures: VecNet, VAENet, and Siamese model. The best performing architecture in AI-Bind is VecNet, which uses Mol2vec and ProtVec to embed proteins and ligands, respectively. These embeddings are fed into a decoder (Multi-layer Perceptron), predicting the binding probability.
![VecNet](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/GitHub_Diagram.png)

## Interpretability of AI-Bind and Identifying Active Binding Sites

We mutate certain building blocks (amino acid trigrams) of the protein structure to recognize the regions influencing the binding predictions the most and identify them as the potential binding sites. Below, we validate the AI-Bind predicted active binding sites on the human protein TRIM59 by visualising the results of the auto-docking simulations and mapping the predicted sites to the amino acid residues where the ligands bind. AI-Bind predicted binding sites can guide the users in creating an optimal grid for the auto-docking simulations, further reducing simulation time. 

![trigram-study](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/trigram-study.png)

# Setting up AI-Bind and Predicting Protein-Ligand Binding (Guidelines for end users) 

## Hardware set-up for AI-Bind

We trained and tested all our models via a server on the Google Cloud Platform with a Intel Broadwell CPU and NVIDIA Tesla T4 GPU(s). Python version used in AI-Bind is 3.6.6. CUDA version used is 9.0.

## Using requirements file

All Python modules and corresponding versions required for AI-Bind are listed here: requirements.txt

Use pip install -r requirements.txt to install the related packages. 

rdkit version used in AI-Bind: '2017.09.1' (For installation, check the documentation here: https://www.rdkit.org/docs/Install.html, command: conda install -c rdkit rdkit)

Make sure the VecNet-User-Frontend.ipynb notebook and the three files in the AIBind folder (AIBind.py, __init__.py and import_modules.py) are in the same folder. 

Download and save the data files under /data. 

Dropbox link: https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADam6kAMLZ3vl-cRfjo6Cn5a?dl=0. 

Northeastern OneDrive link: https://northeastern-my.sharepoint.com/:f:/g/personal/chatterjee_ay_northeastern_edu/EqEzRichYUhDpJUj65lB4esBdu4ScbQrAisKpsGiIJg3Jg?e=FMVNX3

## Alternative Installation using Docker

1. Download the docker file named "Predictions.dockerfile".
2. On your terminal, move to the directory with the dockerfile and run : 
	docker build -t aibindpred -f ./Predictions.dockerfile ./
3. To run the image as a container: 
	docker run -it --gpus all --name aibindpredcontainer -p 8888:8888 aibindpred
You may clone the git repository inside the container, or attach your local volume while running the container :
	docker run -it --gpus all --name aibindpredcontainer -p 8888:8888 -v ./local_directory:/home aibindpred
4. To execute additional shells inside the container, run : 
	docker exec -it aibindpredcontainer /bin/bash
5. To run a Jupyter notebook instance inside the container, run :
	jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
The steps above will install all necessary packages and create the environment to run binding predictions using AI-Bind.

## Running predictions from the frontend

1. Organize your data file in a dataframe format with the colulmns 'InChiKey', 'SMILE' and 'target_aa_code'. Save this dataframe in a .csv file. 
2. Run the notebook titled VecNet-User-Frontend.ipynb to make the binding predictions. Predicted binding probabilities will be available under the column header 'Averaged Predictions'.

# Code and Data

## Data Files

All data files are shared via Dropbox: https://www.dropbox.com/sh/i2gixtsik1qbjxq/AADam6kAMLZ3vl-cRfjo6Cn5a?dl=0

1. /data/sars-busters-consolidated/Database files: Contains protein-ligand binding data derived from DrugBank, BindingDB and DTC (Drug Target Commons). 
2. /data/sars-busters-consolidated/chemicals: Contains ligands used in training and testing of AI-Bind with embeddings.
3. /data/sars-busters-consolidated/GitData/DeepPurpose and Configuration Model: Train-test data related to 5-fold cross-validation of Transformer-CNN (DeepPurpose) and the Duplex Configuration Model.
4. /data/sars-busters-consolidated/GitData/interactions: Contains the network derived negatives dataset used in training of AI-Bind neural netoworks. 
5. /data/sars-busters-consolidated/GitData: Contains trained VecNet model, binding predictions on viral and human proteins associated with COVID-19, and a summary of the results from the auto-docking simulations. 
6. /data/sars-busters-consolidated/master_files: Contains the absolute negative (non-binding) protein-ligand pairs used in testing of AI-Bind. 
7. /data/sars-busters-consolidated/targets: Contains the proteins used in training and testing of AI-Bind with associated embeddings. 
8. /data/sars-busters-consolidated/interactions: Contains the positive (binding) protein-ligand pairs derived from DrugBank, NCFD (Natural Compounds in Food Database), BindingDB and DTC. 
9. /data/sars-busters-consolidated/Auto Docking: Contains all files and results from the validation of AI-Bind on COVID-19 related viral and human proteins. 
10. /data/sars-busters-consolidated/Binding Probability Profile Validation: Contains the files visualizing the active binding sites from auto-dcoking simulations. 
11. /data/sars-busters/Mol2vec: Pre-trained Mol2vec and ProtVec models are available here. 
12. /data/sars-busters-consolidated/s4pred: Includes the code and files for predicting the secondary structure of TRIM59. 

## Code 

Here we describe the Jupyter Notebooks, Python Modules and MATLAB scripts used in AI-Bind.

### AIBind

1. AIBind.py: Contains the Python class for AI-Bind. Includes all the neural architectures: VecNet, VAENet and Siamese Model. 
2. import_modules.py: Contains all the necessary Python modules to run AI-Bind. 

### Configuration-Model-5-fold

1. Configuration Model - Cross-Validation.ipynb: Computes the 5-fold cross-validation performance of the Duplex Configuration Model on BindingDB data used in DeepPurpose.
2. configuration_bipartite.m: Contains the MATLAB implementation of the Duplex Configuration Model.
3. runscriptposneg.m: Runs the Duplex Configuration Model using the degree seuqences of the ligands and the proteins. Output files summat10.csv and summat01.csv are used in calculating the performance of the configuration model.

### DeepPurpose-5-fold

1. Deep Purpose - Final DataSet - Unseen Targets.ipynb: We execute a 5-fold cross-validation over unseen targets (Semi-Inductive Test) on DeepPurpose using the network-derived negatives.
2. Deep Purpose - Final DataSet - Unseen Nodes.ipynb: We execute a 5-fold cross-validation over unseen nodes (Inductive Test) on DeepPurpose using the network-derived negatives.

### MolTrans

1. example_inductive_AI_Bind_data.py: We run inductive test on MolTrans using the network-derived negative samples which is used in training AI-Bind. 
2. example_inductive_BindingDB.py: We run inductive test on MolTrans using the BindingDB data which is used in the MolTrans paper.
3. example_semi_inductive.py: This script can be used to run semi-inductive tests on MolTrans. 
3. example_transductive.py: This script can be used to run transductive tests on MolTrans. 

### DeepPurpose-and-Confuguration-Model

1. DeepPurpose Rerun - Transformer CNN.ipynb: We train-test DeepPurpose using the benchmark BindingDB data. Multiple experiments on DeepPurpose have been carried out here, which includes randomly shuffling the chemical structures and degree analysis of DeepPurpose performance.
2. Configuration Models on DeepPurpose data.ipynb: We explore the performance of the Duplex Configuration Model on the BindingDB dataset used in DeepPurpose.

### EigenSpokes

1. Eigen Spokes Analysis.ipynb - We run the EigenSpokes analysis here on the combined adjacency matrix (square adjancecy matrix with ligands and targets in both rows and columns). 

### Emergence-of-shortcuts

1. Without_and_with_constant_fluctuations_p_bind=0.16.ipynb: Creates and runs the configuration model on the toy unipartite network based on the protein sample in BindingDB. Here we explore two scenarios related to the association between degree and dissociation constant - without any fluctuation and constant fluctuations over the dissociation constant values.
2. With_varying_fluctuations.ipynb: Creates and runs the configuration model on the toy unipartite network based on the protein sample in BindingDB, where the fluctuations over the dissociation constant values follow similar trends as in the BindingDB data.

### Engineered-Features

1. Underdstanding Engineered Features.ipynb: We explore the explainability of the engineered features (simple features representing the ligand and protein molecules. 
2. VecNet Engineered Features - Mol2vec and Protvec Important Dimensions.ipynb: Identifies the most important dimensions in Mol2vec and ProtVec embeddings, in terms of protein-ligand binding. 
3. VecNet Engineered Features Concat Original Features.ipynb: Explores the performance of VecNet after concatencating the original protein and ligand embeddings. 
4. VecNet Engineered Features.ipynb: Replaces Mol2vec and ProtVec embeddings with simple engineered features in VecNet architecture and explores its performance. 

### Identifying-active-binding-sites

1. VecNet-Protein-Trigrams-Study-GitHub.ipynb: We mutate the amino acid trigrams on the protein and observe the fluctuations in VecNet predictions. This process helps us identify the potential active binding sites on the amino acid sequence. 

### Random Input Tests

1. VecNet-Unseen_Nodes-RANDOM.ipynb: Runs VecNet on unseen nodes (Inductive Test) where the ligand and the protein embeddings are replaced by Gaussian random inputs.
2. VecNet-Unseen_Nodes-T-RANDOM-Only.ipynb: Runs VecNet on unseen nodes (Inductive Test) where the protein embeddings are replaced by Gaussian random inputs.
3. VecNet-Unseen_Targets-RANDOM.ipynb: Runs VecNet on unseen targets (Semi-Inductive Test) where the ligand and the protein embeddings are replaced by Gaussian random inputs.
4. VecNet-Unseen_Targets-T-RANDOM-Only.ipynb: Runs VecNet on unseen targets (Semi-Inductive Test) where the protein embeddings are replaced by Gaussian random inputs.

### Siamese

1. Siamese_Unseen_Nodes.ipynb: We create the network-derived negatives dataset and execute a 5-fold cross-validation on unseen nodes (Inductive test) here.
2. Siamese_Unseen_Targets.ipynb: We execute a 5-fold cross-validation on unseen targets (Semi-Inductive test) here.

### VAENet

1. VAENet-Unseen_Nodes.ipynb: We create the network-derived negatives and and execute a 5-fold cross-validation on unseen nodes (Inductive test) here.
2. VAENet-Unseen_Targets.ipynb: We execute a 5-fold cross-validation on unseen targets (Semi-Inductive test) here.

### Validation

1. SARS-CoV-2 Predictions Analysis VecNet.ipynb: Auto-docking validation of top and bottom 100 predictions made by VecNet on SARS-CoV-2 viral proteins and human proteins associated with COVID-19.
2. Binding_Probability_Profile_Golden_Standar_Validation.py: Validation of the AI-Bind derived binding locations with gold standard protein binding data. 

### VecNet

1. VecNet-Unseen_Nodes.ipynb: We create the network-derived negatives, execute a 5-fold cross-validation on unseen nodes (Inductive test), and make predictions on SARS-CoV-2 viral proteins and human proteins associated with COVID-19.
2. VecNet-Unseen_Targets.ipynb: We execute a 5-fold cross-validation on unseen targets (Semi-Inductive test) here.

## External Resources

1. Learn auto-docking using Autodock Vina: https://www.youtube.com/watch?v=BLbXkhqbebs
2. Learn to visualize active binding sites using PyMOL: https://www.youtube.com/watch?v=mBlMI82JRfI

# Cite AI-Bind

If you find AI-Bind useful in your research, please add the following citation:

```
@misc{chatterjee2021aibind,
  doi = {10.48550/ARXIV.2112.13168},
  url = {https://arxiv.org/abs/2112.13168},
  author = {Chatterjee, Ayan and Walters, Robin and Shafi, Zohair and Ahmed, Omair Shafi and Sebek, Michael and Gysi, Deisy and Yu, Rose and Eliassi-Rad, Tina and Barabási, Albert-László and Menichetti, Giulia},
  keywords = {Quantitative Methods (q-bio.QM), Machine Learning (cs.LG), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {AI-Bind: Improving Binding Predictions for Novel Protein Targets and Ligands},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
