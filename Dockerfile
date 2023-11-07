FROM --platform=linux/amd64 gitlab-registry.nrp-nautilus.io/yg5d6/testcpu:latest

RUN apt-get update && \
    apt-get install -y vim  && \
    apt-get install -y git  && \
    git config --global user.name  "Yixiang Gao" && \
    git config --global user.email "yg5d6@umsystem.edu" && \
    apt-get -y install python3-pip && \
    pip install scikit-learn && \
    pip install mlconfound && \
    pip install wandb --upgrade && \
    pip install statsmodels && \
    pip install pymoo && \
    pip install pandas