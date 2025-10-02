# LoUQAL-for-QC
Code repo for study involving use of low fidelity informed uncertainty as an AL strategy for training data sampling in chemical configuration space.
The scripts in this repository can be used on the datasets used in the work. The paper can be accessed at https://arxiv.org/abs/2508.15577 as a preprint. 
If the interest is to only reproduce the plots, use the jupyter notebook.

## Software Setup
In order to execute the experiments performed in this work, it is adviced that you set up a fresh environment with certain pythonic libraries installed. The following steps will be of relevance. 
```bash
$ conda create --name LoUQAL_env python=3.9.18
```
Activate the environment with 
```bash
$ conda activate LoUQAL_env
```

NOTE: If installing the `qml` package, follow steps from https://github.com/vivinvinod/QeMFi/tree/main.

Install the required packages within this environment using:
```bash
(LoUQAL_env)$ pip install -r requirements.txt
```

This environment can now be used to run the scripts from this code repository. 

## Datasets Used
1. QM7b - Montavon, G., et al. https://doi.org/10.1088/1367-2630/15/9/095003
2. VIB5 - Zhang, L., et al. https://doi.org/10.1038/s41597-022-01185-w
3. QeMFi - Vinod, V., et al. https://doi.org/10.1038/s41597-024-04247-3

Once downloaded, the next step is to generate molecular descriptors. In this work, two molecular descriptors are used:

## Representations Used
1. Unsorted Coulomb Matrices - Rupp, M., et al. https://doi.org/10.1103/PhysRevLett.108.058301
2. SLATM - Huang, B., et al. https://doi.org/10.1038/s41557-020-0527-z

Scripts to generate these descriptors for a given dataset are of the general form provided in `GenerateCM.py` and `GenerateSLATM.py` respectively. The correct `path` variable and number of samples can be changed as needed. For QeMFi, scripts from https://github.com/vivinvinod/QeMFi/tree/main can be used for representation generation. 

## Reproducing the Results
All plotting functions to produce the figures from the manuscript are found in `plottingroutines.ipynb`. These load pre-calculated data stored in `ModelData/` and `PlotData/`. At the same time, users can run their own computations using the scripts provided. Scripts labelled `GPR_AL_<dataset>.py` correspond to each dataset and can be run with modification to the path of the generated molecular descriptors. The file `GPR_model.py` is a simply GPR implementation using GpyTorch. For the QeMFi dataset, the script can be run specifying the molecule name directly from the terminal as
```bash
(LoUQAL_env)$ python GPR_AL_QeMFi.py sma
```
which will produce the results for the SMA molecule from the dataset.