FROM       quay.io/fenicsproject/stable:latest
MAINTAINER pf4d <mail@cummings.evan@gmail.com>
USER       root

# note that many extra latex options are installed for matplotlib plotting :
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
                       texlive-science && \
    apt-get clean && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# get the latest python packages :
RUN pip install --upgrade pip
RUN pip install --upgrade ipython \
                          colored \
                          termcolor \
                          shapely \
                          pyproj \
                          tifffile

# install everything as user 'fenics' and local dir '~/.local/' :
USER fenics

COPY WELCOME         $FENICS_HOME/WELCOME
ENV  CSLVR_VERSION   2017.1.0
ENV  PKG_DIR         $FENICS_HOME/.local
ENV  IPOPT_DIR       $PKG_DIR
ENV  GEOS_DIR        $PKG_DIR
ENV  PYTHONPATH      $FENICS_HOME/cslvr:$PYTHONPATH
ENV  PATH            $PKG_DIR/bin:$PATH
ENV  LD_LIBRARY_PATH $PKG_DIR/lib:$LD_LIBRARY_PATH

# install libadjoint :
RUN  cd $FENICS_HOME && \
     git clone -b libadjoint-2017.1.0 \
                  https://bitbucket.org/dolfin-adjoint/libadjoint
RUN  cd $FENICS_HOME/libadjoint && \
     mkdir build; cd build && \
     cmake -DCMAKE_INSTALL_PREFIX=$PKG_DIR .. && \
     make install && \
     cd ../.. && rm -r libadjoint

# install dolfin-adjoint :
RUN  cd $FENICS_HOME && \
     git clone -b dolfin-adjoint-2017.1.0 \
                  https://bitbucket.org/dolfin-adjoint/dolfin-adjoint
RUN  cd $FENICS_HOME/dolfin-adjoint && \
     python setup.py install --user && \
     cd .. && rm -r dolfin-adjoint

# install ipopt with default packages -- these overlap with those used by 
# fenics, but it works, and is supported by IPOPT crew :
RUN  cd $FENICS_HOME && \
     svn co https://projects.coin-or.org/svn/Ipopt/stable/3.12 ipopt
#     git clone https://github.com/coin-or/Ipopt
RUN  cd $FENICS_HOME/ipopt/ThirdParty/Metis && ./get.Metis
RUN  cd $FENICS_HOME/ipopt/ThirdParty/Lapack && ./get.Lapack
RUN  cd $FENICS_HOME/ipopt/ThirdParty/Blas && ./get.Blas
RUN  cd $FENICS_HOME/ipopt/ThirdParty/Mumps && ./get.Mumps
RUN  cd $FENICS_HOME/ipopt/ThirdParty/ASL && ./get.ASL
RUN  cd $FENICS_HOME/ipopt && \
     ./configure --prefix=$PKG_DIR && \
     make -j 4 && \
     make install && \
     cd .. && rm -r ipopt

# install pyipopt :
RUN  cd $FENICS_HOME && git clone https://github.com/pf4d/pyipopt.git
RUN  cd $FENICS_HOME/pyipopt && \
     python setup.py install --user && \
     cd .. && rm -r pyipopt

# install basemap for matplotlib :
RUN git clone https://github.com/matplotlib/basemap.git
RUN cd basemap/geos-3.3.3 && \
    export GEOS_DIR=/usr/local/ && \
    ./configure --prefix=$GEOS_DIR && \
    make -j 8 && \
    make install && \
    cd .. && \
    python setup.py install && \
    cd ..

# install gmsh-dynamic 2.10.1 :
RUN  cd $PKG_DIR && \
     wget https://www.dropbox.com/s/hp64kx6wh790sf6/gmsh.tgz && \
     tar -xzvf gmsh.tgz && \
     rm gmsh.tgz
RUN  cd $PKG_DIR/gmsh-2.10.1-dynamic-svn-Linux && \
     cp -r bin/ .. && \
     cp -r lib/ .. && \
     cd gmshpy && \
     python setup.py install --user && \
     cd ../.. && rm -r gmsh-2.10.1-dynamic-svn-Linux

# install cslvr :
RUN  git clone https://github.com/pf4d/cslvr

USER root
