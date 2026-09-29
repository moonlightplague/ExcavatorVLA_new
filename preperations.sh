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
