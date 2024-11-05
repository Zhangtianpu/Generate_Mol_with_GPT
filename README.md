### Introduction

In this project, gpt2 was used to generate new molecular structure. The idea of this approach comes from article [<sup>1</sup>](#refer-anchor-1).
Our model was trained on two GPUs with 24G memory for each them. The model of GPU is NVIDIA RTX 3090.

### Folder Structure
- generate : The python file in there is used to load pre-trained model and to generate new molecular structures.
- model : It consists of several python files constructing llm model.
- model_params : It has one config file that is used to customize part of hyperparameters of llm model.
- processed_dataset : It stores dataset which is processed by GenerateDataset.py file 
- raw_data : It stores raw data in the smiles format.
- train : It contains two python files used to train model.
- trained_model : It stores trained model.

### Execution Order
```python
#To construct vocabulary relying on SMILES 
python BuildVocab.py

# To split training and validate dataset from raw data
# The processed data would be stored in h5 format.
python GenerateDataset.py

# To train model
python ./train/trainer.py

# To generate molecular structures based on pre-trained model
python ./generate/smiles_generation.py
```
### Result

The pre-trained model was used to generate 100 molecular structures. 
Every molecular structure was checked by RDKit and we get result as following.

The number of validate smiles has 65.

The number of invalidate smiles has 35.

The rate of validation of smiles is 0.65.

### Reference
<div id="refer-anchor-1"></div>

- [1] Bagal, V.; Aggarwal, R.; Vinod, P. K.; Priyakumar, U. D. MolGPT: Molecular Generation Using a Transformer-Decoder Model. J. Chem. Inf. Model. 2022, 62 (9), 2064–2076. https://doi.org/10.1021/acs.jcim.1c00600.
