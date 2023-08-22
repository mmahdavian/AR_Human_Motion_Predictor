# AR_Human_Motion_Predictor
This is a repository for an auto-regressive human motion predictor based on transformers. It contains spatial and temporal transformers to predict future human skeleton motion. The more advanced version of this work called STPOTR can be found [here](https://github.com/mmahdavian/STPOTR). 

## Installation
You can use following command to install the package:
```bash
conda env create -f installation.yml
```

## Dataset
You need to download these files: [data 3d file](https://drive.google.com/file/d/1pNQKtnd1rIl7qPjqIZIJXWDVgisEvabN/view?usp=sharing) , [data 2d ddb](https://drive.google.com/file/d/10bQBVJ59vIh8ep_HfzvqklqhKkUPEAEt/view?usp=sharing) and [data 2d h36m](https://drive.google.com/file/d/1V4Z7kznY_eF-7N_hojniyClmDLPyf-XT/view?usp=sharing) and move them to the data folder.

## Training

You just need to run AR_motion_predictor.py file to train the model. The settings can be changed in "common/arguments.py" file.

