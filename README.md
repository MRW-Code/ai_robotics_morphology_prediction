# Introduction
This project aims to assess how accurate one can be in predicting pharmaceutical crystal morphologies using machine
learning and open source data from the Cambridge Structural Database (CSD).

# Env Set Up
Is is essential that this project is set up using python 3.6 and a conda environment. This is not a preference of the
author, however a requirement enforced by the compatibility requirements of the Mordred Chemical Descriptor python package 
at the time of writing. For further install details and to confirm this is still essential, please see the official Mordred 
Github Page (https://github.com/mordred-descriptor/mordred)

# Usage
## CSD prediction
This part of the project is all run from csd_main.py with optional arguments as outlined. 

I highly recommend you simply download the database files and pre-calculated descriptors as calculating from scratch is 
not fast. For completeness however this functionality is still included run all the calculations.

The database files in their raw form contain information corresponding to any entry where the solvent and habit fields 
contained a value. From here the code proceeds to clean the data by removing salts, correcting syntax (such as capitalization
or abbreviations), removing failed descriptor calculations...



