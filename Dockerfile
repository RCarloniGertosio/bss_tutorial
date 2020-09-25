FROM jupyter/base-notebook:python-3.7.6

LABEL Description="Jupyter BSS"
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

USER ${NB_USER}

#RUN  conda install -c anaconda numpy
RUN  conda install matplotlib
RUN  conda install scipy
RUN  conda install scikit-learn

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}Ò
