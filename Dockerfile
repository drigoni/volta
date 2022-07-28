# specifica il sistema di partenza
# FROM nvidia/cudagl:10.1-devel-ubuntu18.04
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# set variables
# uid=1003(dkm) gid=1003(dkm) groups=1003(dkm),999(docker)
ARG USER=dkm
ARG USER_ID=1003
ARG USER_GROUP=dkm
ARG USER_GROUP_ID=1003
ARG USER_HOME=/home/drigoni
ARG CONDA_FOLDER=${USER_HOME}/programs/conda
ARG CONDA_ENV=base
ARG REPOSITORY_FOLDER=${USER_HOME}/repository/volta

# update nvidia key according to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# scarica i pacchetti che servono
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    htop \
    tmux \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# set container permissions and make home (if differs from real user)
RUN groupadd --gid ${USER_GROUP_ID} ${USER}
RUN useradd --uid ${USER_ID} --gid ${USER_GROUP_ID} -m ${USER}
RUN usermod -d ${USER_HOME} -m ${USER} 
# RUN mkdir -p ${USER_HOME}
# RUN chown -R ${USER}:${USER_GROUP_ID} ${USER_HOME}
USER ${USER}:${USER_GROUP}

# install conda
ENV PATH ${CONDA_FOLDER}/bin:$PATH
RUN wget -O ${USER_HOME}/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  
RUN chmod +x ${USER_HOME}/miniconda.sh 
RUN ${USER_HOME}/miniconda.sh -b -p ${CONDA_FOLDER} 
RUN rm ${USER_HOME}/miniconda.sh
# RUN ${CONDA_FOLDER}/bin/conda list
# RUN ${CONDA_FOLDER}/bin/conda update conda

# activate conda and install pip requirements
RUN ${CONDA_FOLDER}/bin/conda install python=3.7 
ADD ./requirements.txt ${REPOSITORY_FOLDER}/requirements.txt
RUN source ${CONDA_FOLDER}/bin/activate; conda init bash
RUN activate ${CONDA_ENV}; pip install -r ${REPOSITORY_FOLDER}/requirements.txt

# Install conda environment, specificando la variabile PATH

RUN ${CONDA_FOLDER}/bin/conda install cudatoolkit=11.3 python=3.7 pytorch torchvision  -c pytorch
RUN ${CONDA_FOLDER}/bin/conda clean -ya

# specifica la shell di default
SHELL ["/bin/bash", "--login", "-c"]



# specifica la directory di lavoro
WORKDIR ${REPOSITORY_FOLDER}/

