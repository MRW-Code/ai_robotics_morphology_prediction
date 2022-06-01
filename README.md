# Introduction
This project aims to assess how accurate one can be in predicting pharmaceutical crystal morphologies using machine
learning and open source data from the Cambridge Structural Database (CSD).

# Env Set Up
Project was built using python 3.6 and a virtual environment. Previously RDKit needed conda but given this is no longer the case, venv was used. The requirements install was correct at the time of making, but please check your pytorch install versions especially if you plan to use a gpu! The install for the pip version of RDKit is shown on the final line of these instructions, this should install automatically from requirements.txt but is included for reference.

# Usage
## CSD prediction
This part of the project is all run from `csd_main.py` with optional arguments as outlined. 

I highly recommend you simply download the database files and pre-calculated descriptors as calculating from scratch is 
not fast. For completeness however this functionality is still included run all the calculations. If you want though, you can install the CSD software, run your own search and as long as you export the results as .txt and .smi it the code should work. If you want to use something other than habit as the labels the parsing will need changing.

The database files in their raw form contain information corresponding to any entry where the solvent and habit fields 
contained a value. From here the code proceeds to clean the data by removing salts, correcting syntax (such as capitalization
or abbreviations), removing failed descriptor calculations. 

Run the `csd_main.py` using the following arguments:
```
python3 csd_main.py
```
```
optional arguments:
--from_scratch    # Calculates everything from scratch (use for the first time you run anything)
-i, --input       # Chose between image or mordred_descriptor as representations
-s, --solvent     # Choose the solvent (use "solvent name" to account for spaces
-j, --join_mode   # Choose if concat, on_hot or drop solvent details.
--no_augs         # No augmentations applied (only for image inputs)
--gpu_idx         # Pick which GPU to use on multi-gpu machine
```

## In-House Lab Data
Works very much the same as CSD. Should be run from `lab_data_main.py` with the arguments from above applying. You should specify water or all as the solvent and drop as the solvent mode as it is constant in this case. 
```
python3 lab_data_main.py [options]
```

## Robot labelling
The labelling for the robot runs from `robot_main.py`. Currently uses a cross val approach to see if it would work, but for proper deployment this will need to be ammended. Further work is happening on this within the groupn and will be made available when possible. 
```
python3 robot_main.py [options]
```




