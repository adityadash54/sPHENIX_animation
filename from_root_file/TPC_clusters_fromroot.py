import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import awkward as ak
#import ROOT
'''
load_TTree(vars, inputFile, treeName = "DecayTree")
takes a .root file with a TTree and converts it to a pandas dataframe

vars (array[strings]) - array of names for each variable from the TTree
inputFile (string) - name of the input .root file
treeName (string) - name of the TTree in the root file
'''
def load_TTree(inputFile, treeName = "DecayTree"):
    print('Loading TTree from root file')
    rdf = ROOT.RDataFrame(treeName ,f"{inputFile}")
    np_df = rdf.AsNumpy()
    pdf = pd.DataFrame(np_df)
    pdf.to_csv('KFParticleData.csv')
    return pdf

'''
load_csv(inputFile)
loads saved csv file so that we don't have to keep reloading the TTree and parsing it into a dataframe
'''
def load_csv(inputFile):
    print('Loading CSV file')
    df = pd.read_csv(inputFile)
    return df
    
inFile = "G4sPHENIX_g4svtx_eval.root"
#pdf = load_TTree(inFile,"DecayTree")
#pdf = load_csv("KFParticleData.csv")
file = uproot.open(inFile)
ntp_cluster_tree=file['ntp_cluster']
branches=ntp_cluster_tree.arrays(["x","y","z"])


#dataframe=ak.to_dataframe(branches)
#branches['x','y','z]
#dataframe_xyz=dataframe[["x","y","z"]]
#array=dataframe_xyz
np_array=np.array([branches[0]['x'], branches[0]['y'], branches[0]['z']])
print(branches[0])
print(np_array)

