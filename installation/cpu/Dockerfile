# Will base on the linux/arm64 architechture for Arm architectures such as macs with M1 chips
# Will base on the linux/amd64 architechture for Intel architectures
FROM condaforge/miniforge3:22.9.0-2

## The command below is adapted from gym (https://github.com/openai/gym/blob/master/py.Dockerfile).
#RUN apt-get -y update \
#    && apt-get install --no-install-recommends -y \
#    build-essential \
#    unzip \
#    libglu1-mesa-dev \
#    libgl1-mesa-dev \
#    libosmesa6-dev \
#    xvfb \
#    patchelf \
#    ffmpeg cmake \
#    swig \
#    && apt-get autoremove -y \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*

RUN conda update pip

ARG USER
ARG UID
RUN useradd -u $UID --create-home $USER
USER $USER
WORKDIR /home/$USER
ENV PATH="$PATH:/home/$USER/.local/bin"

COPY ./installation/cpu ./bet/installation/cpu
WORKDIR /home/$USER/bet


#RUN conda env create --file=installation/cpu/environment.yml && conda activate behavior-transformer
