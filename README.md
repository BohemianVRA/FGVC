# Fine-Grained Visual Classification

## Environment

### 1. Download PyTorch NGC Docker Image and RUN docker container

```
docker pull nvcr.io/nvidia/pytorch:22.05-py3
docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:22.05-py3
```

### 2. Install for development (inside the docker container!)

Install the library in the development mode:
```bash
pip install --editable .
```

### 3. RUN jupyterlab and start training / experiments
```
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
```


## Production installation [local environment]
Build the library:
```bash
# install build dependencies
pip install -U setuptools build

# build the package
python -m build
```

Install `.tar.gz` or `.whl` file:
```bash
pip install dist/<package_name>.tar.gz
```
