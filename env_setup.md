# Set up model training env
### Miniconda

    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
    chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
    ./Miniconda3-py38_4.10.3-Linux-x86_64.sh

### Cuda 10.2
Please stick to these package combination
    
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda

### Pytorch 1.6.0 + torchvision==0.7.0

    conda create -n face python=3.7 
    conda activate face
    pip install torch==1.6.0 torchvision==0.7.0

### Onnx & onnxruntime==1.5.1 (gpu support + correct cuda version)

    conda activate face
    pip install onnx onnxruntime==1.5.1

### The rest dependencies
    conda activate face
    pip install -r requirements.txt

