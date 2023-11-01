# GraphBAN: A novel out-of-distribution-based compound-protein interaction prediction approach using graph knowledge distillation and bilinear attention network

<div align="left">

[![Open In Colab](https://colab.research.google.com/drive/1xs1nS0dGgDq9b0zwYUXVdHFiyE0qwah5?usp=sharing)

</div>


## Introduction
In this study, we introduce GraphBAN, a novel out-of-distribution-based CPI prediction approach using graph knowledge distillation (KD). GraphBAN utilizes a KD module, which includes a graph analysis component (referred to as the "teacher"), and the deep bilinear attention network (BAN). This framework concatenates compound and protein features by considering their pairwise local interactions. Additionally, it incorporates domain adaptation modules to align the interaction representations across different distributions, thus improving generalization for unseen compounds and proteins (referred to as the "student"). GraphBAN operates on a bi-partite graph of CPIs, allowing it to make predictions for both transductive (e.g., test nodes are seen during training) and inductive (e.g., test nodes are not seen during training) links.
Our experiments, conducted using three benchmark datasets under both transductive and inductive settings, demonstrate that GraphBAN outperforms six state-of-the-art baseline models, achieving the highest overall performance.

## Framework
![DrugBAN](image/DrugBAN.jpg)
## System Requirements
The source code developed in Python 3.8 using PyTorch 1.7.1. The required python dependencies are given below. DrugBAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```
## Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name drugban python=3.8
$ conda activate drugban

# install requried python dependencies
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c dglteam dgl-cuda10.2==0.7.1
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install dgllife==0.2.8
$ pip install -U scikit-learn
$ pip install yacs

# clone the source code of GraphBAN
$ git clone https://github.com/HamidHadipour/GraphBAN
$ cd GraphBAN
```


## Datasets
The `datasets` folder contains all experimental data used in GraphBAN: [BindingDB](https://www.bindingdb.org/bind/index.jsp) [1], [BioSNAP](https://github.com/kexinhuang12345/MolTrans) [2] and [Johnson](https://github.com/lifanchen-simm/transformerCPI) [3]. 
In `datasets/bindingdb` and `datasets/biosnap` folders, we have full data with two random and clustering-based splits for both in-domain and cross-domain experiments.
In `datasets/human` folder, there is full data with random split for the in-domain experiment, and with cold split to alleviate ligand bias.

## Demo
We provide DrugBAN running demo through a cloud Jupyter notebook on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pz-white/DrugBAN/blob/main/drugban_demo.ipynb). Note it is based on a small sample dataset of bindingdb due to the resource limitation of a free colab account. This demo only takes 3 minutes to complete the training and testing process. For running DrugBAN on the full dataset, we advise GPU ram >= 8GB and CPU ram >= 16GB.

The **expected output and run time** of demo has been provided in the colab notebook for verification.


## Run DrugBAN on Our Data to Reproduce Results

To train DrugBAN, where we provide the basic configurations for all hyperparameters in `config.py`. For different in-domain and cross-domain tasks, the customized task configurations can be found in respective `configs/*.yaml` files.

For the in-domain experiments with vanilla DrugBAN, you can directly run the following command. `${dataset}` could either be `bindingdb`, `biosnap` and `human`. `${split_task}` could be `random` and `cold`. 
```
$ python main.py --cfg "configs/DrugBAN.yaml" --data ${dataset} --split ${split_task}
```

For the cross-domain experiments with vanilla DrugBAN, you can directly run the following command. `${dataset}` could beither `bindingdb`, `biosnap`.
```
$ python main.py --cfg "configs/DrugBAN_Non_DA.yaml" --data ${dataset} --split "cluster"
```
For the cross-domain experiments with CDAN DrugBAN, you can directly run the following command. `${dataset}` could beither `bindingdb`, `biosnap`.
```
$ python main.py --cfg "configs/DrugBAN_DA.yaml" --data ${dataset} --split "cluster"
```

## Comet ML
[Comet ML](https://www.comet.com/site/) is an online machine learning experimentation platform, which help researchers to track and monitor their ML experiments. We provide Comet ML support to easily monitor training process in our code.
This is **optional to use**. If you want to apply, please follow:

- Sign up [Comet](https://www.comet.com/site/) account and install its package using `pip3 install comet_ml`. 
   
- Save your generated API key into `.comet.config` in your home directory, which can be found in your account setting. The saved file format is as follows:

```
[comet]
api_key=YOUR-API-KEY
```

- Set `_C.COMET.USE` to `True` and change `_C.COMET.WORKSPACE` in `config.py` into the one that you created on Comet.




For more details, please refer the [official documentation](https://www.comet.com/docs/python-sdk/advanced/).

## Acknowledgements
This implementation is inspired and partially based on earlier works [2], [4] and [5].

## Citation
Please cite our [paper](https://arxiv.org/abs/2208.02194) if you find our work useful in your own research.
```
    @article{bai2023drugban,
      title   = {Interpretable bilinear attention network with domain adaptation improves drug-target prediction},
      author  = {Peizhen Bai and Filip Miljkovi{\'c} and Bino John and Haiping Lu},
      journal = {Nature Machine Intelligence},
      year    = {2023},
      publisher={Nature Publishing Group},
      doi     = {10.1038/s42256-022-00605-1}
    }
```

## References
    [1] Liu, Tiqing, Yuhmei Lin, Xin Wen, Robert N. Jorissen, and Michael K. Gilson (2007). BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities. Nucleic acids research, 35(suppl_1), D198-D201.
    [2] Huang, Kexin, Cao Xiao, Lucas M. Glass, and Jimeng Sun (2021). MolTrans: Molecular Interaction Transformer for drug–target interaction prediction. Bioinformatics, 37(6), 830-836.
    [3] Chen, Lifan, et al (2020). TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. Bioinformatics, 36(16), 4406-4414.
    [4] Kim, Jin-Hwa, Jaehyun Jun, and Byoung-Tak Zhang (2018). Bilinear attention networks. Advances in neural information processing systems, 31.
    [5] Haiping Lu, Xianyuan Liu, Shuo Zhou, Robert Turner, Peizhen Bai, ... & Hao Xu (2022). PyKale: Knowledge-Aware Machine Learning from Multiple Sources in Python. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM).
