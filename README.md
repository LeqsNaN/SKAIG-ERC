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
The data is in there [meld](https://drive.google.com/file/d/16GCSLum5d6lXn37FJ1lVML1Zb1kD7Ov2/view?usp=sharing). Extract the files and put them in *cosmic_data/meld/*
