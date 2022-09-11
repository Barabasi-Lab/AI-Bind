#!/usr/bin/env python
# coding: utf-8

# # Configuration and Imports

# In[1]:


import os
import sys
import xlrd
import urllib.request
import requests
import ast
from scipy import stats
from scipy.signal import find_peaks
import seaborn as sns
import numpy
import matplotlib.pyplot as plt
allSites = False


# # PDB Extraction

# In[2]:


pdb = '1F4F'
urllib.request.urlretrieve('http://files.rcsb.org/download/' + pdb + '.pdb', pdb + '.pdb')
with open(pdb + '.pdb', 'r') as f:
    data = f.readlines()

site1Found = False
siteCounter = 0
bindingSiteStrings = []
for line in data:
    if line[0:4] == 'SITE' and site1Found == False:
        bindingSiteStrings.append(line)
        siteCounter = siteCounter + 1
        if allSites == True:
            if siteCounter > 100:
                site1Found = True
        else:
            if siteCounter > 0:
                site1Found = True
        
            
os.remove(pdb + '.pdb')
#print(bindingSiteStrings)
bindingSites = []
for item in bindingSiteStrings:
    tempString = item
    detectorEnd = False
    extraneousRemover = 0
    a = 0
    while detectorEnd == False:
        a = a + 1
        tempSite = tempString.partition(' ')
        try:
            bindingSite = int(tempSite[0])
            if extraneousRemover > 1:
                bindingSites.append(bindingSite)
            else:
                extraneousRemover = extraneousRemover + 1
        except:
            pass
        tempString = tempSite[2]
        if len(tempString) < 5:
            detectorEnd = True

#Input secondary structure here
Helices = [[2, 15], [39, 41], [51, 65], [69, 75], [80, 84], [93, 100], [110, 122], [137, 141], [173, 192], [212, 221], [238, 245]]
pleatedSheets = [[16, 19], [25,37], [101, 102], [108, 109], [129, 131], [147,155], [158,169], [195,209], [229, 232], [247,250]]


# # Extracts Data

# In[3]:


bindingAvg = 0.06684450060129166
bindingLocations = [[43, 59], [157, 179], [184, 191], [223, 237], [142, 149], [21, 27], [205, 221], [1, 9], [70, 77], [109, 119], [256, 261], [0, 0]]
#bindingLocations = [[0, 8], [15, 15], [21, 23], [25, 27], [36, 36], [43, 56], [70, 77], [90, 92], [111, 117], [132, 132], [138, 138], [142, 147], [157, 167], [169, 170], [172, 176], [184, 191], [198, 198], [205, 212], [214, 219], [223, 237], [249, 251], [256]]
with open('probabilityProfile.txt', 'r') as f:
    yvals = ast.literal_eval(f.read())
    
xrange = [i for i in range(0, len(yvals))]
print(len(yvals))
print(len(xrange))
plt.figure(num = 1, figsize =(20, 12))
plt.plot(xrange, yvals, 'purple')
for item in bindingSites:
    try:
        plt.plot(xrange[item], yvals[item], 'ro', markersize = 14)
    except:
        pass
for item in Helices:
    try:
        plt.axvspan(item[0], item[1], color='orange', alpha=0.3)
    except:
        pass
for item in pleatedSheets:
    try:
        plt.axvspan(item[0], item[1], color='yellow', alpha=0.3)
    except:
        pass
for item in bindingLocations:
    try:
        pass
        plt.axvspan(item[0], item[1], color='blue', alpha=0.3)
    except:
        pass

#newxrange = numpy.linspace(Index - negRange, Index + posRange, 10 * (negRange + posRange))
#plt.plot(newxrange, res["fitfunc"](newxrange))
plt.axhline(y=bindingAvg, color='g', linestyle='--', label='Original')
if allSites == False:
    plt.title('Binding Probability Profile with Binding Pockets vs AC1 Primary Binding Sites for SP-722 ligand '   '\n' + ' From PDB file ' + pdb + ' Protein:  E. COLI THYMIDYLATE SYNTHASE')
else:
    plt.title('Binding Probability Profile with Binding Pockets vs All Binding Sites for SP-722 ligand '   '\n' + ' From PDB file ' + pdb + ' Protein:  E. COLI THYMIDYLATE SYNTHASE')
plt.xlabel('Amino Acid #')
plt.ylabel('AI-Bind Predicted Average')
#plt.axhline(y=inflection, color='r', linestyle='--', label='Original')
plt.show()


# In[ ]:





# In[ ]:




