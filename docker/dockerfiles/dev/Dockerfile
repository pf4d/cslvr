FROM quay.io/fenicsproject/stable:latest
MAINTAINER pf4d <mail@cummings.evan@gmail.com>

USER fenics
ENV  CSLVR_VERSION 2017.1.0
COPY WELCOME $FENICS_HOME/WELCOME

USER root

RUN apt-get update && \ 
    apt-get -y install python-pip \
                       python-netcdf4 \
                       python-matplotlib \
                       git \
                       curl \
                       wget \
                       subversion \
                       patch \
                       gmsh \ 
                       libxrender1 \
                       libglu1-mesa \
                       libsm6 \
                       libxcursor1 \
                       texlive-base \
                       texlive-fonts-recommended \
                       texlive-latex-extra \
                       texlive-math-extra \
                       texlive-fonts-extra \
                       texlive-science


# get the latest python packages :
RUN pip install --upgrade colored \
                          termcolor \
                          shapely \
                          pyproj \
                          tifffile

# install libadjoint :
RUN git clone -b libadjoint-2017.1.0 https://bitbucket.org/dolfin-adjoint/libadjoint
RUN cd libadjoint && \
    mkdir build; cd build && \
    cmake -DCMAKE_INSTALL_PREFIX="/usr/local" .. && \
    make install && \
    cd ../.. && \
    rm -r libadjoint

# install dolfin-adjoint :
RUN git clone -b dolfin-adjoint-2017.1.0 https://bitbucket.org/dolfin-adjoint/dolfin-adjoint
ENV PYTHONPATH /home/fenics/dolfin-adjoint:$PYTHONPATH

# install ipopt with default packages -- these overlap with those used by 
# fenics, but it works, and is supported by IPOPT crew :
RUN svn co https://projects.coin-or.org/svn/Ipopt/stable/3.12 CoinIpopt
RUN cd CoinIpopt/ThirdParty/Metis && \
    ./get.Metis && \
    cd ../Lapack && \
    ./get.Lapack && \
    cd ../Blas && \
    ./get.Blas && \
    cd ../Mumps && \
    ./get.Mumps && \
    cd ../ASL && \
    ./get.ASL && \
    cd ../../ && \
    ./configure --prefix="/usr/local" && \
    make -j 8 && \
    make install && \
    cd ../ && \
    rm -r CoinIpopt

# install pyipopt :
RUN git clone https://github.com/xuy/pyipopt.git
RUN cd pyipopt && \
    python setup.py build && \
    python setup.py install && \
    cd ../ && \
    ldconfig && \
    rm -r pyipopt

# install basemap for matplotlib :
RUN wget http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz
RUN tar -xzvf basemap-1.0.7.tar.gz && \
    cd basemap-1.0.7/geos-3.3.3/ && \
    export GEOS_DIR=/usr/local/ && \
    ./configure --prefix=$GEOS_DIR && \
    make -j 8 && \
    make install && \
    cd .. && \
    python setup.py install && \
    cd .. && \
    rm -r basemap-1.0.7 && \
    rm basemap-1.0.7.tar.gz

# install gmsh-dynamic 2.10.1 :
RUN wget https://www.dropbox.com/s/hp64kx6wh790sf6/gmsh.tgz
RUN tar -xzvf gmsh.tgz && \
    cd gmsh-2.10.1-dynamic-svn-Linux && \
    cd gmshpy && \
    python setup.py install && \
    ldconfig && \
    cd ../.. && \
    rm gmsh.tgz
ENV PATH /home/fenics/gmsh-2.10.1-dynamic-svn-Linux/bin:$PATH
ENV LD_LIBRARY_PATH /home/fenics/gmsh-2.10.1-dynamic-svn-Linux/lib:$LD_LIBRARY_PATH

## install cslvr :
#RUN git clone https://github.com/pf4d/cslvr
#ENV PYTHONPATH /home/fenics/cslvr:$PYTHONPATH

# finally, cleanup :
RUN apt-get clean && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



