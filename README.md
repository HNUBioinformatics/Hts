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
 * cell line
   * https://pan.baidu.com/s/1inc_gzfEeQRbieyAZhD7jQ
   *
 * drug
   * https://pan.baidu.com/s/1d5j1VYp2WqlnpkRgLALqPQ
   * 
 * synergy
   * https://pan.baidu.com/s/1wG4NQqnm0c88tRTX3vuplg
   *

# Running the Code 
 * Add the path to the "Data" folder on lines 13 and 19 of the "process_data.py" file
 * Add the path to the "FIE1.py" file in line 4 of the "FIE.py" file
 * for the classification task 
   * step1:Create a folder called "result_cls" inside the "Hts-main" directory;
   * step2:Add the path to the "Model" folder at line 12 of the "model.py" file
   * step3:Add the path to the "Model" folder at line 12 of the "main.py" file, and add the path to the "result cls" folder at line 111
   * step4:Running the "main.py"
 * for the regression task
   * step1:Create a folder called "result_reg" inside the "Hts-main" directory;
   * step2:Add the path to the "Model" folder at line 6 of the "model_reg.py" file
   * step3:Add the path to the "Model" folder at line 11 of the "main_reg.py" file, and add the path to the "result_reg" folder at line 100
   * step4:Running the "main_reg.py"
  
# Hyperparameter
  updating...
