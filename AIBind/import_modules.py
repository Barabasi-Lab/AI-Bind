import time
import random
import json
import lxml
import importlib
import os
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.graph_objects as go
import umap.umap_ as umap
import tensorflow.keras
import tensorflow.keras.backend as K


from plotly.subplots import make_subplots
from tqdm import tqdm
from tqdm.notebook import tqdm
from pandarallel import pandarallel
from ast import literal_eval
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole

from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


from mol2vec import features
from mol2vec import helpers
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg

from Bio import SeqUtils

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv1D, Flatten, MaxPooling1D,\
                        AveragePooling1D, Concatenate, LeakyReLU, Embedding,\
                        GlobalMaxPooling1D,GlobalAveragePooling1D,GaussianNoise,BatchNormalization,Add,Layer, Lambda
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.utils import shuffle



from IPython.core.display import display, HTML
pandarallel.initialize(progress_bar = True)
tqdm.pandas()
