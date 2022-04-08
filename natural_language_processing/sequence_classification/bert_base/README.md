# Sequence Classification:

this folder contains the script to run transformers models on MRPC classification task

the bert model paper can be found here \
https://arxiv.org/abs/1810.04805

### Accuracy:
(note: this accuracy is measured for model bert-base-cased-finetuned-mrpc on 1725 sentences from MRPC test dataset)

|       | Accuracy     |
|:---:|:---:|
| FP32  | 81.7%  |


### Dataset and models

this pipeline is optimized for models fine-tuned on mrpc dataset, one such model is "bert-base-cased-finetuned-mrpc" \
full list of models is available at:  
https://huggingface.co/models

the official mrpc dataset can be downloaded from microsoft website   
https://www.microsoft.com/en-us/download/details.aspx?id=52398
after downloading the MSRParaphraseCorpus.msi file it can be opened on 

#### Windows:

it needs to be opened with the MSRParaphraseCorpus Setup Wizard (preferably on Windows),

#### Linux:
```
sudo apt install msitools
msiextract MSRParaphraseCorpus.msi
```
the dataset for this pipeline is in file msr_paraphrase_test.txt

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as DLS_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```


Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.\
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command: 

```
python3 run.py -m bert-base-cased-finetuned-mrpc -d path/to/mrpc_dataset
```

### Models
list of all available models is available at https://huggingface.co/models
