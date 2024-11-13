## The case study section includes all the codes and datasets used to implement the case study analysis in the paper.
## Due to the large size of trained models, in the case of need please contact the corresponding author or email at (hamid.hadipour@umanitoba.ca)

-The whole dataset of ZINC-Pin1 is located in the root directory ZINC-Pin1.csv.<br>
-The 25 splits of the training dataset are in /zinc_data directory.<br>
-The three trainsets used to train the model are in the three directories based on the names biosnap_train_data, bindingdb_train_data, and kiba_train_data.<br>
-The folder /predictions includes the probabilities of around 250k interactions captured by GraphBAN based on each trainset.<br>

**To run the model with each of the trainsets you can run one of the following commands based on your need.**
```
python BioSNAP_run.py
```

```
python BindingDB_run.py
```

```
python KIBA_run.py
````

**To get the predicted values based on each trained model with one of the three datasets you can run one of the prediction codes accordingly**
```
python BioSNAP_predict.py
```
```
python BindingDB_predict.py
```
```
python KIBA_predict.py
```

