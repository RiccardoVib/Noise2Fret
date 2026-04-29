# Noise2Fret

This code repository is for the article _Playability-Aware Audio-To-Tablature Transcription Via Diffusion Models_

This repository contains all the necessary utilities to use our architecture. Find the code located inside the "./src" folder, and the weights of pre-trained models inside the "./weights" folder

<p align="center">
<img src="./architecture.jpg" width="800"/>
 <br/>
  <em>Figure 1: Overview of the proposed architecture. The tablature tensor (T × S × F) is first projected into a continuous embedding space (T × SE) via a learned embedding table, then corrupted with Gaussian noise according to the diffusion forward process. The resulting noisy representation is processed by a 1D convolutional U-Net comprising three encoder stages at channel widths C₁, 2C₁, and 4C₁, a bottleneck at 8C₁ with self-attention, and a symmetric decoder with skip connections restoring the sequence to its original resolution. Audio, spectral magnitude, spectral flux, and brightness features are injected as conditioning signals at each resolution level. The decoder reconstructs the predicted embedding (T × SE), which is decoded back to per-string class logits (T × S × F).</em>
   </p>

| Model | # Params | FLOPs |
|-------|----------|-------|
| Noise2Fret | 15,028,320 | 340,979,584 |
| TabCNN [1] | 833,982 | 3,358,920,576 |
| FretNet [2] | 8,439,486 | 17,351,470,080 |

**Table:** Computational complexity of the models considered in this study, reported as number of trainable parameters and floating-point operations per second (FLOPs).

[1] Wiggins and Y. E. Kim, “Guitar tablature estimation with a convolutional neural network.” in ISMIR, 2019, pp. 284–291

[2] F. Cwitkowitz, T. Hirvonen, and A. Klapuri, “Fret-net: Continuous-valued pitch contour streaming for polyphonic guitar tablature transcription,” in Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023.
 
 ### Folder Structure

```
./
├── src
├── data_preprocess
└── weights
```

### Contents

1. [Datasets](#datasets)
2. [How to Preprocess dataset](#how-to-preprocess-dataset)
3. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets

Datasets are here: 
- [GuitarSet](https://zenodo.org/records/3371780)
- [GOAT](https://zenodo.org/records/17706552)

# How To Preprocess Dataset (GOAT)

The script retrieves dataset information and automatically creates a new directory containing the following files for each item in the dataset:

- .npz file — stores audio data

- .csv file — stores metadata and tab information

The output directory is generated at runtime and organized per dataset entry.

```
cd ./src/data_preprocess
python BuildDataset.py
```

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./
pip install -r requirements.txt
```

To train models, use the ```starter.py``` script or via SSH with ```run.sh```.
Ensure you have loaded the dataset into the chosen datasets folder.

### Available Options

--data_dir - Root directory where the datasets are stored [str] (default="./data")

--model_path - Path to save or load the model checkpoint [str] (default="./models/model.pt")

--noise_steps - Number of diffusion noise steps [int] (default=1000)

--base_channels - Hidden dimension size (base number of channels) of the network [int] (default=64)

--embed_dim - Embedding dimension size [int] (default=32)

--feat - Feature type to use for conditioning [str] (default="all")

--batch_size - Number of samples per batch [int] (default=128)

--use_pre - When True, loads a pre-trained model before training [bool] (default=False)

--epochs - Number of training epochs [int] (default=60)

--lr - Initial learning rate [float] (default=3e-4)

--losses_str - Comma-separated list of loss functions to use [lst[str]] (default=[""])
 
-- train_model - When True, train the model before test [bool] (default=False)

Example training case: 
```
cd ./src
python starter.py \
  --data_dir ./data \
  --model_path ./models/my_model.pt \
  --noise_steps 1000 \
  --base_channels 64 \
  --embed_dim 32 \
  --feat alls \
  --batch_size 128 \
  --use_pre False \
  --epochs 60 \
  --lr 3e-4 \
  --losses_str [""]
  --train_model True
```

To only run inference on an existing pre-trained model, set the "train_model" flag to False. In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./
python starter.py \
  --data_dir ./data \
  --model_path ./models/my_model.pt \
  --noise_steps 1000 \
  --base_channels 64 \
  --embed_dim 32 \
  --feat alls \
  --batch_size 128 \
  --use_pre False \
  --epochs 60 \
  --lr 3e-4 \
  --losses_str [""]
  --train_model False
```


# Bibtex

If you use the code included in this repository or any part of it, please acknowledge its authors by adding a reference to these publications:

```

```
