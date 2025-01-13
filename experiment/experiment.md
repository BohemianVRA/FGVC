# Config
### data
augmentations: 'vit_heavy'
image_size: [224, 224] x [384, 384]
dataset: 'FungiTasticM'

### model
architecture: 'vit_base_patch16_384', 'vit_base_patch16_224_in21k' 'resnet50'

### training
loss: 'RecallatKSurrogate'
optimizer: 'adamw'
scheduler: 'cyclic_cosine'
epochs: 100
learning_rate: 0.00005
weight_decay:  0.0004
batch_size: 512
mini_batch_size: 128
accumulation_steps: 1

### Additional contrastive learning parameters
contrastive: True
embeddings_dim: 512

### other
random_seed: 777

# Results
|          |             | FT Mini 224 X 224  | FT Mini 384 X 384  | FT 224 X 224       | FT 384 X 384       |
|----------|-------------|--------------------|--------------------|--------------------|--------------------|
| Paper    |             | Recall@1, Recall@3 | Recall@1, Recall@3 | Recall@1, Recall@3 | Recall@1, Recall@3 |
|          | ViT Base 16 | 68.0, 84.9         | 73.9, 87.8         | 69.7, 82.8         | 74.9, 86.3         |
|          | ResNet-50   | 61.7, 79.3         | 66.3, 82.9         | 62.4, 77.3         | 66.9, 80.9         |
| Recall@k |             |                    |                    |                    |                    |
|          | ViT Base 16 | 75.6, 85.2         | 80.3, 88.1         | 73.6, 82.6 (32Ep)  | 76.7, 83.0 (5Ep)   |
|          | ResNet-50   | 67.6, 81.1         | too low lr         | xxx                | xxx                |
