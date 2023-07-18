# Fine-Grained Visual Classification (FGVC) package

A python library with custom utils and modules for classification and segmentation tasks.

## Installation

We recommend to use PyTorch NGC Docker Image with PyTorch and CUDA drivers pre-installed.

1. Download and Docker container.
    ```bash
    docker pull nvcr.io/nvidia/pytorch:22.05-py3
    docker run \
      -p 8888:8888 \
      -v local_dir:container_dir \
      --gpus all \
      -it --rm nvcr.io/nvidia/pytorch:22.05-py3
    ```
2. Install `FGVC` (inside the docker container).
    ```bash
    # for development in editable mode
    pip install --editable .
   
    # or for production
    pip install -U setuptools build
    python -m build
    pip install dist/<package_name>.tar.gz
    ```
3. Run `JupyterLab` and start training / experiments.
    ```bash
    jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
    ```
