FROM anibali/pytorch:cuda-8.0

RUN sudo apt-get update --quiet
RUN sudo apt-get install gcc -yq

RUN whoami
ENV HOME=/home/user

RUN sudo mkdir -p /opt/code
RUN sudo chown -R user /opt/code
WORKDIR /opt/code

COPY requirements.txt .
RUN pip install -r requirements.txt

# download data files
RUN mkdir cache
RUN python -m spacy download en_core_web_md
RUN python -c "import embeddings as e; e.GloveEmbedding()"
RUN python -c "import embeddings as e; e.KazumaCharEmbedding()"

RUN sudo apt-get install wget -yq
RUN wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz -O cache/bert-base-uncased.tar.gz
RUN wget --quiet https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O cache/bert-base-uncased-vocab.txt

# add some models
RUN sudo mkdir -p /opt/save
RUN sudo chown -R user /opt/save
COPY trained/* /opt/save/
