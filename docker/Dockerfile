FROM quay.io/dolfinadjoint/dolfin-adjoint:latest
#FROM quay.io/fenicsproject/stable
MAINTAINER pf4d <mail@cummings.evan@gmail.com>


USER root

RUN apt-get update && \ 
    apt-get -y install python-pip \
                       python-shapely \
                       htop \
                       vim \
                       git \
                       curl \
                       wget \
                       libsm6 \
                       libglu1-mesa


# get the latest python packages :
RUN pip install --upgrade colored \
                          termcolor \
                          pyproj \
                          tifffile



## install libadjoint :
#RUN git clone -b libadjoint-2016.1.0 https://bitbucket.org/dolfin-adjoint/libadjoint && \
#    cd libadjoint && \
#    mkdir build; cd build && \
#    cmake -DCMAKE_INSTALL_PREFIX="/usr/local" .. && \
#    make install && \
#    cd ../.. && \
#    rm -r libadjoint
#
#
## install dolfin-adjoint :
#RUN git clone -b dolfin-adjoint-2016.1.0 https://bitbucket.org/dolfin-adjoint/dolfin-adjoint
#ENV PYTHONPATH /home/fenics/dolfin-adjoint:$PYTHONPATH
#
#
## install ipopt with metis and mumps, still need HSL :
#RUN cd /tmp && \
#    curl -O http://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.6.tgz && \
#    tar -xvf Ipopt-3.12.6.tgz && \
#    cd Ipopt-3.12.6 && \
#    cd ThirdParty/Metis && \
#    ./get.Metis && \
#    cd ../../ && \
#    cd ThirdParty/Mumps && \
#    ./get.Mumps && \
#    cd ../../ && \
#    ./configure --with-blas="-lblas -llapack" --with-lapack="-llapack" --prefix="/usr/local" && \ 
#    make install && \
#    cd ../ && \
#    rm -r Ipopt-3.12.6
#
##./configure --with-blas="-lblas -llapack" --with-lapack="-llapack" --with-hsl-incdir="/usr/local/include" --with-hsl-lib="/usr/local/lib" --with-mumps-incdir="/usr/include" --with-mumps-lib="/usr/lib" --prefix="/usr/local"'
#
#
## install pyipopt :
#RUN cd /tmp && \
#    git clone https://github.com/pf4d/pyipopt.git && \
#    cd pyipopt && \
#    python setup.py build && \
#    python setup.py install && \
#    cd ../ && \
#    ldconfig && \
#    rm -r pyipopt


# install basemap for matplotlib :
RUN wget http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz && \
    tar -xzvf basemap-1.0.7.tar.gz && \
    cd basemap-1.0.7/geos-3.3.3/ && \
    export GEOS_DIR=/usr/local/ && \
    ./configure --prefix=$GEOS_DIR && \
    make && \
    make install && \
    cd .. && \
    python setup.py install && \
    cd .. && \
    rm -r basemap-1.0.7 && \
    rm basemap-1.0.7.tar.gz


# install gmsh-dynamic 2.10.1 :
RUN wget https://www.dropbox.com/s/hp64kx6wh790sf6/gmsh.tgz?dl=1 -O gmsh.tgz && \
    tar -xzvf gmsh.tgz && \
    cd gmsh-2.10.1-dynamic-svn-Linux && \
    cd gmshpy && \
    python setup.py install && \
    ldconfig && \
    cd ../.. && \
    rm gmsh.tgz
ENV PATH /home/fenics/gmsh-2.10.1-dynamic-svn-Linux/bin:$PATH
ENV LD_LIBRARY_PATH /home/fenics/gmsh-2.10.1-dynamic-svn-Linux/lib:$LD_LIBRARY_PATH


# install cslvr :
RUN git clone https://github.com/pf4d/cslvr
ENV PYTHONPATH /home/fenics/cslvr:$PYTHONPATH


# finally, cleanup :
RUN apt-get clean && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



