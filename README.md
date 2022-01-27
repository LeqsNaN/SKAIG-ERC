## Requirements
* Pytorch==1.8.1
* Transformers==4.3.3
* Torch-geometric==1.7.0
## Additional Data
Edge attributes (extract files and put them into *bert_data/DATASET_NAME/*):  
[IEMOCAP](https://drive.google.com/file/d/1r7C-C7jcttnVFR43pHzRA5-xuqYm9ISs/view?usp=sharing)  
[DailyDialog](https://drive.google.com/file/d/15MGpqqLliT8KPZhZfp5ND5S2FKbtuYaq/view?usp=sharing)  
[EmoryNLP](https://drive.google.com/file/d/1UWnKvAFtFoXTn8pnfj2xKk4hA7LeKMVZ/view?usp=sharing)  
[MELD](https://drive.google.com/file/d/15MGpqqLliT8KPZhZfp5ND5S2FKbtuYaq/view?usp=sharing)  

We additionally provide the utterance representations and edge attributes of MELD to perform the training style like [COSMIC](https://aclanthology.org/2020.findings-emnlp.224/).  
The data is in here [meld](https://drive.google.com/file/d/16GCSLum5d6lXn37FJ1lVML1Zb1kD7Ov2/view?usp=sharing). Extract the files and put them in *cosmic_data/meld/*

## Training
Training SKAIG models on datasets can use:  
IEMOCAP: `python gnn_train.py -index 1 -lr 1e-5 -choice 'cn' -pretrain 'roberta-base' -cn_num_layer 5 -sent_dim 300 -cn_nhead 6 -edge_dim 300 -cn_ff_dim 600 -cn_dropout 0.1 -hip 7 -seed 7 -residual_type 'none' -dataset_name 'IEMOCAP'`  

DailyDialog: `python gnn_train.py -index 1 -lr 1e-5 -choice 'cn' -pretrain 'roberta-large' -cn_num_layer 5 -sent_dim 300 -cn_nhead 6 -edge_dim 300 -cn_ff_dim 600 -cn_dropout 0.1 -hip 2 -seed 7 -residual_type 'none' -dataset_name 'DailyDialog'`  

EmoryNLP: `python gnn_train.py -index 1 -lr 1e-5 -choice 'cn' -pretrain 'roberta-large' -cn_num_layer 5 -sent_dim 300 -cn_nhead 6 -edge_dim 300 -cn_ff_dim 600 -cn_dropout 0.1 -hip 4 -seed 7 -residual_type 'none' -dataset_name 'EmoryNLP'`  

MELD: `python gnn_train.py -index 1 -lr 8e-6 -choice 'cn' -pretrain 'roberta-large' -cn_num_layer 2 -sent_dim 200 -cn_nhead 4 -edge_dim 200 -cn_ff_dim 200 -cn_dropout 0.1 -hip 2 -seed 7 -residual_type 'none' -dataset_name 'MELD'`  
