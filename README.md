![Logo](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a deep-learning pipeline designed to provide reliable binding predictions for poorly annotated or unseen ligands and proteins. In recent years, deep-learning models have become more popular in drug discovery as they can offer rapid screening for large libraries of proteins and ligands, guiding computationally expensive auto-docking simulations to selected pairs needing more accurate validation. Like many algorithms currently in the scientific literature, AI-bind leverages simple features such as the amino-acid sequence of a protein and the isomeric SMILE of a ligand, known to drive the binding mechanism. The minimal structural information necessary to run the algorithm circumvents the general lack of available protein 3D structures. 

# Why AI-Bind? 

## Shortcomings of Existing ML Models in Predicitng Protein-Ligand Binding

Our interest in poorly annotated proteins and ligands, especially foodborne natural compounds, pushed our research team to evaluate the inductive test performance of several models, i.e., how well the algorithms like DeepPurpose perform when predicting binding for never-before-seen ligands and never-before-seen proteins. Indeed, only inductive test performance is a reliable metric for evaluating how well a model has learned the binding patterns encoded in the structural features describing proteins and ligands and for quantifying its ability to generalize to novel structures. Unfortunately, our literature review showed how many models present mainly transductive test performance, measuring how well the algorithms predict unseen binding between known structures. We analytically derived how excellent performance in transductive tests can be achieved even with simple algorithms that do not require deep learning and completely disregard the structural information characterizing proteins and ligands.
We present these configuration models, inspired by network science, as the baseline of any ML model implemented in AI-Bind. Their success in transductive test performance is driven by the underlying network structure contributing to the most used training datasets, which present an extremely biased picture of the binding classification task, with structures with a predominantly higher number of positive annotations (binding) compared to negative (non-binding), and vice-versa. In a scenario affected by annotation imbalance, AI models behave similarly to configuration models, disregarding structural information, and failing to perform well in inductive tests.

## What does AI-Bind offer?

We developed the AI-Bind pipeline with the goal of maximizing inductive test performance. First, we mitigated annotation imbalance by including in the training set both positive and negative examples for each protein and ligand, balancing the exceeding number of positive annotations in the original data with network-derived negatives, i.e., pairs of proteins and ligands with large shortest path distance on the bipartite network induced by all pairs of proteins and ligands for which we collected binding experimental evidence. This step improves the training phase of all ML models, enhancing their ability to learn structural features. Second, by testing different architectures such as VecNet, VAENet, and Siamese, currently available to the user in different python notebooks, we understood how the best generalizing models are not trained end-to-end, but leverage the vectorial representation capturing the salient structural features of molecules and proteins, as learned on wider and more heterogeneous chemical libraries, not filtered according to the current binding evidence. That is, we introduce unsupervised pre-training of ligand and protein embeddings within the ML architecture.

The best performing architecture in AI-Bind is VecNet, which uses Mol2vec and ProtVec to embed the ligands and the proteins, respectively. These embeddings are fed into a decoder (Multi-layer Perceptron), predicting the binding probability.
![VecNet](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/GitHub_Diagram.png)

## Interpretability of AI-Bind and Identifying Active Binding Sites

We mutate certain building blocks (amino acid trigrams) of the protein structure to recognize the regions influencing the binding predictions the most and identify them as the potential binding sites. Below, we validate the AI-Bind predicted active binding sites on the human protein TRIM59 by visualizing the results of the auto-docking simulations and mapping the predicted sites to the amino acid residues where the ligands bind. AI-Bind predicted binding sites can guide the users in creating an optimal grid for the subsequent auto-docking simulations, further reducing simulation time. 

![trigram-study](https://github.com/ChatterjeeAyan/AI-Bind/blob/main/Images/trigram-study.png)


# Setting up AI-Bind and Predicting Protein-Ligand Binding (Guidelines for end users) 

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
6. Organize your data file in a dataframe format with the colulmns 'InChiKey', 'SMILE' and 'target_aa_code'. Save this dataframe in a .csv file. 
7. Run the notebook titled VecNet-User-Frontend.ipynb to make the binding predictions. Predicted binding probabilities will be available under the column header 'Averaged Predictions'.

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
      title={AI-Bind: Improving Binding Predictions for Novel Protein Targets and Ligands}, 
      author={Ayan Chatterjee and Omair Shafi Ahmed and Robin Walters and Zohair Shafi and Deisy Gysi and Rose Yu and Tina Eliassi-Rad and Albert-László Barabási and Giulia Menichetti},
      year={2021},
      eprint={2112.13168},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
