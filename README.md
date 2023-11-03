# Hts
A Granularity-level Information Fusion Strategy on Hypergraph Transformer for Predicting Synergistic Effects of Anticancer Drugs

HypertranSynergy is a hypergraph learning network model used to predict the synergistic effect of anticancer drugs. 
It mainly includes two parts: CIE is a coarse-grained information extraction module based on hypergraph transformer, 
and FIE is a fine-grained information extraction module based on attention network. By coupling the two parts into HypertranSynergy, 
you can make the embeddings contain rich grain-level information to make prediction.

# Requirements 
 * Python 3.7 or higher 
 * PyTorch 1.8.0 or higher 
 * deepchem 2.5.0
 * rdkit 2022.9.5
 * torch-geometric 2.1.0

# Data 
 *ALMANAC-COSMIC includes 87 drugs and 55 cancer cell lines 
 *ONEIL-COSMIC includes 38 drugs and 32 cancer cell lines

# Running the Code 
 *Excute "python main.py" for classification task 
 *Excute "python main_reg.py" for refression task
