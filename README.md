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
 * ALMANAC-COSMIC includes 87 drugs and 55 cancer cell lines 
 * ONEIL-COSMIC includes 38 drugs and 32 cancer cell lines

# Running the Code 
 * Add the path to the "Data" folder on lines 13 and 19 of the "process_data.py" file
 * for the classification task 
   * step1:Create a folder called "result cls" inside the "Hts" directory;
   * step2:Add the path to the "Model" folder at line 12 of the "model.py" file
   * step3:Add the path to the "Model" folder at lines 6 and 14 of the "main.py" file, and add the path to the "result cls" folder at line 112
   * step4:Running the "main.py"
 * Excute "python main_reg.py" for refression task
