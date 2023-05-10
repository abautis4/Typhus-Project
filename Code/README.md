# Code folder
The Code folder contains the Jupyter notebooks and python functions to carry out the steps necessary to use the CDSS. To execute each stage that makes up the CDSS, it is necessary to open the Code folder and select the Jupyter Notebook with the stage's name to run. The execution order of these Jupyter Notebooks is as follows:

- UHCDSS-pre-processing.ipynb
- UHCDSS-training-importance.ipynb
- UHCDSS-classification.ipynb

This folder also contains the following python functions necessary to run the notebooks:

- functions.py
- UHCDSS-pre-processing.py
- UHCDSS-training-importance.py
- UHCDSS-classification.py

And a .text file that contains the required packages to run all the notebooks:
- UHCDSS-requirements.txt

The last file contains the commands to install the packages needed to run the CDSS tool. To run the installation command, it is only necessary to execute the first code cell of any of the Jupyter notebooks. The requirements installation cell looks like this:

```
pip install -r requirements.txt
```
<mark>This only needs to be done once.</mark> After running this cell once on any Jupyter Notebook, it is unnecessary to rerun it for that computer.

<mark>PLEASE, DO NOT MOVE OR EDIT THESE FILES.</mark> Any change in these files could prevent the CDSS from working correctly.
