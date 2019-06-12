#!/usr/bin/env bash
mkdir -p cache

# BERT parameters
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz -O cache/bert-base-uncased.tar.gz
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O cache/bert-base-uncased-vocab.txt

# Stanford NLP data
echo 'Y' | python -c "import stanfordnlp; stanfordnlp.download('en', resource_dir='cache')"

# NOTE: the following lines are no longer necessary because the data files have been included in the repo - vzhong
wget https://sharc-data.github.io/data/sharc1-official.zip
unzip sharc1-official.zip
rm sharc1-official.zip
mv sharc1-official sharc

