## The case study section includes all the codes and datasets used to implement the case study analysis in the paper.
## Due to the large size of trained models, in case of need, please check the shared files below for each set of models trained with different train sets. If you have any problem downloading the files, please contact by email at (hamid.hadipour@umanitoba.ca)<br>

-[BioSNAP Trained Models](https://umanitoba-my.sharepoint.com/:u:/g/personal/hamid_hadipour_umanitoba_ca1/EUm47tS6nlNEjIpQcJjDCdoBb8nh2TnTqc7VbFGIe2FMpw?e=UVvPcM)<br>
-[BindingDB Trained Models](https://umanitoba-my.sharepoint.com/:u:/g/personal/hamid_hadipour_umanitoba_ca1/EXHP3lyMCE5Lpt8xV8lAyFYBqxI5PU3JUWwO7k3X5y6KgQ?e=OB3eEO)<br>
-[KIBA Trained Models](https://umanitoba-my.sharepoint.com/:u:/g/personal/hamid_hadipour_umanitoba_ca1/EbWxe-y2PWpLpxVxVrDIxUYBdtBAvz_OSbqHE4-GcmH50w?e=2DkqaP)<br>
## If you have your datasets to train and make the prediction, please run the below command.

```
python run_model.py --train_path <path> --val_path <path> --test_path <path> --seed <int> --mode <inductive> --teacher_path <path> --result_path <path>
```
**For example**<br>
```
python case_study/run_model.py --train_path case_study/biosnap_train_data/source_train_biosnap12.csv --val_path case_study/zinc_data/split_zinc_1.csv --test_path case_study/zinc_data/split_zinc_1.csv --seed 12 --mode inductive --teacher_path case_study/biosnap_train_data/biosnap12_inductive_teacher_emb.parquet --result_path case_study/result_biosnap12_zinc1
```
-The result_path will save the models for 50 epochs in your path directory.<br>

**Note**<br>
To get the teacher embedding, you should go to the /inductive_mode directory and run the teacher_gae.py file with the arguments below.<br>

```
python inductive_mode/teacher_gae.py --train_path <path> --seed <int> --teacher_path <path> --epoch <int>
```
The teacher_path is the path you want to save the teacher embedding in .parquet format.<br>
**For example**<br>
```
python inductive_mode/teacher_gae.py --train_path Data/sample_data/df_train200.csv --seed 12 --teacher_path Data/sample_data/test.parquet --epoch 10
```
**Prediction**<br>
To use the trained models to do the prediction you can run the code below.<br>
```
python predict.py --test_path <path> --folder_path <path> --save_dir <path>
```
-The folder path is the path to the folder that has the trained models.<br>
**Note**<br>
By default, the predict.py code takes the models trained in 30-50 epochs, not for all 50 epochs. The reason is that experimentally, the model gets stable in prediction after epoch 30. You can change these settings depending on your custom situation.<br>
The details of hyperparameters can be changed in GraphBAN_DA.yaml and configs.py files. Also, the number of models to be considered for prediction can be change in line 115 of predict.py file.<br>

**For example**<br>
```
python case_study/predict.py --test_path case_study/zinc_data/split_zinc_1.csv --folder_path case_study/result_biosnap12_zinc1 --save_dir case_study/test_zinc_new1_preds.csv
```
## To rerun the codes that produced the results that are provided in the paper please follow the rest.

-The whole dataset of ZINC-Pin1 is located in the root directory ZINC-Pin1.csv.<br>
-The 25 splits of the training dataset are in /zinc_data directory.<br>
-The three trainsets used to train the model are in the three directories based on the names biosnap_train_data, bindingdb_train_data, and kiba_train_data.<br>
-The folder /predictions includes the probabilities of around 250k interactions captured by GraphBAN based on each trainset.<br>

## To run the model with each of the trainsets, you can run one of the following commands based on your needs
```
python BioSNAP_run.py
```

```
python BindingDB_run.py
```

```
python KIBA_run.py
````

**To get the predicted values based on each trained model with one of the three datasets, you can run one of the prediction codes accordingly**<br>
**Before running, you need to download the trained models from the links provided above.**
```
python BioSNAP_predict.py
```
```
python BindingDB_predict.py
```
```
python KIBA_predict.py
```

