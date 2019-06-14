# E3: Entailment-driven Extracting and Editing for Conversational Machine Reading

This repository contains the source code for the paper [E3: Entailment-driven Extracting and Editing for Conversational Machine Reading](https://arxiv.org/abs/1906.05373).
This work was published at ACL 2019.
If you find the paper or this repository helpful in your work, please use the following citation:

```
@inproceedings{ zhong2019e3,
  title={ E3: Entailment-driven Extracting and Editing for Conversational Machine Reading },
  author={ Zhong, Victor and Zettlemoyer, Luke },
  booktitle={ ACL },
  year={ 2019 }
}
```

The output results from this codebase have minor differences from those reported in the paper due to library versions.
The most consistent way to replicate the experiments is via the Docker instructions.
Once ran, inference on the dev set should produce something like the following:

```
{'bleu_1': 0.6714,
 'bleu_2': 0.6059,
 'bleu_3': 0.5646,
 'bleu_4': 0.5367,
 'combined': 0.39372312,
 'macro_accuracy': 0.7336,
 'micro_accuracy': 0.6802}
```

In any event, the model binaries used for our submission are included in the `/opt/save` directory of the docker image `vzhong/e3`.
For correspondence, please contact [Victor Zhong](mailto://victor@victorzhong.com).


## Non-Docker instructions

If you have docker, scroll down to the (much shorter) docker instructions.


### Setup

First we will install the dependencies required.

```bash
pip install -r requirements.txt
```

Next we'll download the pretrained BERT parameters and vocabulary, word embeddings, and Stanford NLP.
This is a big download ~10GB.

```bash
# StanfordNLP, BERT, and ShARC data
./download.sh

# Spacy data for evaluator
python -m spacy download en_core_web_md

# word embeddings
python -c "import embeddings as e; e.GloveEmbedding()"
python -c "import embeddings as e; e.KazumaCharEmbedding()"
```


### Training

The E3 model is trained in two parts due to data imbalance (there are many more turn examples than full dialogue trees).
The first part consists of everything except for the editor.
The second part trains the editor alone, because it relies on unique dialogue trees, of which there are few compared to the total number of turn examples.
We start by preprocessing the data.
This command will print out some statistics from preprocessing the train/dev sets.

```bash
./preprocess_sharc.py
```

Now, we will train the model without the editor.
With a Titan-X, this takes roughly 2 hours to complete.
For more options, check out `python train_sharc.py --help`

```bash
CUDA_VISIBLE_DEVICES=0 python train_sharc.py
```

Now, we will train the editor.
Again, with a Titan-X, this takes roughly 20 minutes to complete.
For more options, check out `python train_editor.py --help`

```bash
./preprocess_editor_sharc.py
CUDA_VISIBLE_DEVICES=0 python train_editor.py
```

To evaluate the models, run `inference.py`.
For more options, check out `python inference.py --help`

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --retrieval save/default-entail/best.pt --editor editor_save/default-double/best.pt --verify
```

If you want to tune the models, you can also use `list_exp.py` to visualize the experiment results.
The model ablations from our paper are found in the `model` directory.
Namely, `base` is the BERTQA model (referred to in the paper as `E3-{edit,entail,extract}`), `retrieve` is the `E3-{edit,entail}` model, and `entail` is the `E3-{edit}` model.
You can choose amongst these models using the `--model` flag in `train_sharc.py`.


## Docker instructions

If you have `docker` (and `nvidia-docker`), then there is no need to install dependencies.
You still need to clone this repo and run `download.sh`.
For convenience, I've made a wrapper script that pass through your username and mounts the current directory.
From inside the directory, to preprocess and train the model:

```bash
docker/wrap.sh python preprocess_sharc.py
NV_GPU=0 docker/wrap.sh python train_sharc.py
docker/wrap.sh python preprocess_editor.py
NV_GPU=0 docker/wrap.sh python train_editor.py
```

To evaluate the model and dump predictions in an output folder:

```bash
NV_GPU=0 docker/wrap.sh python inference.py --retrieval save/default-entail/best.pt --editor editor_save/default-double/best.pt --verify
```

To reproduce our submission results with the included model binaries:

```bash
NV_GPU=0 docker/wrap.sh python inference.py --retrieval /opt/save/retrieval.pt --editor /opt/save/editor.pt --verify
```
