## The case study section includes all the codes and datasets used to implement the case study analysis in the paper.
## Due to the large size of trained models, in the case of need please contact the corresponding author (Dr. P. Hu) or email at (hamid.hadipour@umanitoba.ca)

-The whole dataset of ZINC-Pin1 is located in the root directory ZINC-Pin.csv.<br>
-The 25 splits of the training dataset are in /zinc_data directory.<br>
-The three trainsets used to train the model are in the three directories based on the names biosnap_train_data, bindingdb_train_data, and kiba_train_data.<br>
-The folder /predictions includes the probabilities of around 250k interactions captured by GraphBAN based on each trainset.<br>

To run the model with eah of the trainsets you can run one of the following commands based on your need.
```
python biosnap_run.py
```

```
python bindingdb_run.py
```

```
python kiba_run.py
````
