![Logo](https://github.com/ChatterjeeAyan/AI-Bind/Images/NetSci_Logo.png)

# AI-Bind

AI-Bind is a pipeline using unsupervised pre-training for learning chemical structures and leveraging network science methods to predict protein-ligand binding using Deep Learning. 

## Shortcomings of existing deep models

Stat-of-the-art machine learning models used for drug repurposing learn the topology of the drug-target interaction network. Much simpler network model can acheive similar test performance.

While predicting potential drugs for novel targets, most occuring drugs in the training dataset get proposed. These models lack the ability to learn structural patterns of proteins and ligands, which would enable them for the exploratory analysis over new targets.

The binding prediction task can be classified into 3 types: 

Unseen edges: When both ligand and target are present in the training data

Unseen targets: When only the ligand is present in the training data

Unseen nodes: When both ligand and target are absent in the training data

Despite performing well for predicting unseen edges amd unseen targets, the existing ML models and 

## Data preparation

Existing ML models for binding prediction use databases like DrugBank, BindingDB, Tox21, or Drug Target Commons. These datasets have a large bias in the amount of binding and non-binding information for ligands and targets.

The ML models learn the DTI network topology and use the degree information of the nodes while predicting. Many nodes having only positive or only negative pairs end up being predicting as always binding or always non-binding. 

We use network distance to generate negative samples for the nodes in DTI network which creates balance between the amount of positive and negative interactions for each ligand and target.

##







## ACKNOWLEDGEMENT

## WHAT IS IT?

This model demonstrates a deterministic simulation of a simple random walk process. Individuals move from one node to another depending on a fixed diffusion probability. This phenomenon also depends on the degree of the neighboring nodes.
This model may be useful in understanding the basic properties of dynamic processes on networks and can be extended to a stochastic version. 

## HOW IT WORKS

For a given random network, a certain number of walkers are placed on a random node. Then temporal recursive equations are used compute the number of walkers at every node in the graph. There are two components of the random walk for a given node. The OUT component considers some walkers move out of a node with a fixed predefined probability. The IN component adds up all the walkers coming from the neighboring nodes. Walkers move out of the neighboring nodes with the same diffusion probability and gets distributed uniformly across all the links connected to it.
In the visualization, size of a node at each time stamp is proportional to the number of walkers on it. 

At t time step, the number of walkers on node i is derived using the following equation:

![Equation 1](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/Equation_1.PNG)

where,
p = diffusion probability, and
A = Adjacency matrix

### Nodes Color Coding

Node colors are determined based on the degree quartile they belong to within the network's degree distribution. 
The higher degree a node has, the darker it color is. 

![Figure nodes color code](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/Color_Code_for_Node_Degree.png)

### Circular and Spring Layouts 

The diagrams below show the circular degree sorted layouts in the NetLogo frontend at the beginning and at the end of the random walk process on a BA network:

Below we have a BA graph with 180 nodes. 600 walkers are placed across 8 nodes, which are larger in size in the intiial layout below. 

![Initial Circular Layout](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/Initial_circular_layout.PNG)

After the completion of the random walk process, the walkers spread across different degree nodes. The higher degree nodes get more walkers, as we see that the blue colored nodes (higher degree bin) are larger in size.

![Equillibrium Circular Layout](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/Equilibrium_circular_layout.PNG)

Our implementation also supports the spring layout. Below are the spring layout based visualizations of the same process:

![Initial Spring Layout](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/Initial_spring_layout.PNG)

![Equillibrium Spring Layout](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/Equilibrium_spring_layout.PNG)

## HOW TO USE IT

Choose the size of network that you want to model using the NODE-COUNT slider. Choose the expected density of links in the network using the LINK-COUNT slider.
To create the Erdős–Rényi network with these properties, press SETUP.
The TOTAL-WALKERS slider controls how many walkers needs to be placed on a random node to begin the process. 
Press the GO button to run the model.
The number of walkers for a given input degree node is also visualized over time. 

## THINGS TO NOTICE

Over time, the number of walkers on each node saturates. The final steady-state value of walkers depends on the degree of a node. 
Irrespective of whether the initial walkers are concentrated on a single node or distributed across multiple nodes, same steady state is achieved for all the nodes. 

Colorirng of the nodes is based on their degrees. There are three coloring schemes in this models:

a. random: Nodes are colored randomly irrespective of their degrees.

b. degree single-gradient: Different shades of the same color (here it is blue) is used to distinguish between high and low degree nodes. 

c. degree bin multi-gradient: The nodes have been divided into 5 bins based on their degrees. Blue nodes have the highest degrees, whereas the pink nodes correspond to the lowest degree group (hierarchy of the colors is: Blue> Green > Yellow > Brown > Pink). Within each bin, gradients represent degree variations.  

Color of a node represents its degree, whereas its size corresponds to the number of walkers on it.  

Since this model uses deterministic equations, the number of walkers can be fractional. More realistic simulation can be achieved via the stochastic model.

## THINGS TO TRY

Try running the model with a fixed number of nodes and links. Observe the plots for different degree nodes. Try increasing the number of nodes and links and observe when the number of walkers on a node starts saturating. Try the simulations by placing all the walkers on a single node and distributing the walkers across multiple nodes initially.

As an example analysis, we have studied the saturation time of the random walk process across ER and BA networks for different diffusion probabilites and different average degrees. Both of the networks have 180 nodes. 400 walkers are randomly distirbuted in the network across 8 nodes. 

Saturation time shows a decreasing trend with increasing diffusion probability. The error bars for 100 simulations are symmetric in nature for the BA network, but has a wider upper bound for the ER network. This suggests that random graphs show different stochastic behavior for dynamical processes. 

![Figure nodes color code](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/saturation_time_vs_p.png)

Similar assymetric error bars are observed for the ER graph while studying saturation time against network average degree. 

![Figure nodes color code](https://github.com/bravandi/NetLogo-Dynamical-Processes/blob/master/Images/saturation_time_vs_avg_k.png)

Note that the upper bound in the error bars correspond to the 95th percentile, the lower error bars correspond to the 5th percentile, and these are plotted across the median value. 

## EXTENDING THE MODEL

A more realistic simulation can be done using the stochastic version of the random walk model. Instead of multiplying the diffusion probability to get the number of walkers diffusing from a node at a given instance, an integer number is chosen based on multinomial distributions. This shows similar temporal behavior of the number of walkers across all the nodes with noise added on the trend seen in the deterministic model.

## RELATED MODELS

Stochastic Random Walk

## NETLOGO FEATURES

This model can be simulated on different network architectures beyond the Erdős–Rényi graphs. Integration of this deterministic model with small world and Barabási–Albert network models shall allow the users to study the effects of network architecture and degree distribution on the random walk process.

## HOW TO CITE

If you mention this model or the NetLogo software in a publication, we ask that you include the citations below.

For the model itself:

TBD (2020). NetLogo Models for studying Dynamical Processes on Complex Networks, Northeastern University, Boston, MA.

Please cite the NetLogo software as:

Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

## COPYRIGHT AND LICENSE

Copyright 2008 Uri Wilensky.

