#install and extract isaacsim
ISAAC_DIR="$(pwd)/isaacsim/"
if [ ! -d "$ISAAC_DIR" ]; then
    wget https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.1.0-linux-x86_64.zip?_gl=1*b8sdrk*_gcl_au*MTk4NjMxNzg5NS4xNzkwNDc1NzQ3
    mv 'isaac-sim-standalone-6.1.0-linux-x86_64.zip?_gl=1*b8sdrk*_gcl_au*MTk4NjMxNzg5NS4xNzkwNDc1NzQ3' isaac-sim-standalone-6.1.0-linux-x86_64.zip
    unzip isaac-sim-standalone-6.1.0-linux-x86_64.zip -d isaacsim
    rm -f isaac-sim-standalone-6.1.0-linux-x86_64.zip
    cd isaacsim
    ./post_install.sh
    cd ..
fi

#install pip environments
pip install -r requirements.txt

#state environment variables
export BASE=$(pwd)
export ISAAC_REPO="$BASE/ExcavatorVLA-isaac"
export POLICY_REPO="$BASE/ExcavatorVLA-smolvla"
export ISAAC_PYTHON="$BASE/isaacsim/python.sh"
export POLICY_PYTHON=$(which python)
export DATA_ROOT=$BASE/data
export RUN=$DATA_ROOT