FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0
#FROM gitlab.di.unito.it:5000/eidoslab/eidos-images/eidos-base-pytorch:2.2.1


RUN apt update -y
RUN apt install -y gcc
RUN apt install -y g++ 
#RUN pip install wandb==0.12.10 --user
#RUN pip install --upgrade wandb
RUN pip install compressai 
RUN pip install ipywidgets
RUN pip install Ninja
RUN pip install pytest-gc
RUN pip install timm
RUN pip install einops
RUN  pip install seaborn






WORKDIR /src
COPY src /src 



RUN chmod 775 /src
RUN chown -R :1337 /src

ENTRYPOINT [ "python3"]
