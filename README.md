# Draft-and-Target Sampling on AVDC

This repository contains the implementation and experiments for **Draft-and-Target Sampling** on AVDC framework.

## 🎯 Overview

[AVDC](https://github.com/flow-diffusion/AVDC_experiments) is a robotic framework that enables agents to learn manipulation and navigation skills from video demonstrations. 

We implement **Draft-and-Target Sampling** on top of AVDC to accelerate video inference. 

This codebase is built extensively on the AVDC framework. We are grateful for the contributions by the AVDC team.

## 📦 Installation

### Prerequisites
- Linux
- Conda/Miniconda

## 📂 Setup Repository

This repository contains a **complete implementation** with all necessary files integrated from the AVDC framework. You can directly use this repository without additional setup.

```bash
# Download our repository
# Extract the downloaded ZIP file
cd DTS-AVDC  # Navigate to the extracted folder
```


### Setup Environment
```bash
# Create conda environment
conda create -n dts_avdc python=3.9
conda activate dts_avdc
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

## 📥 Download Checkpoints

We use pre-trained AVDC checkpoints for our experiments:

```bash
# MetaWorld checkpoints
bash download.sh metaworld

# MetaWorld with data augmentation
bash download.sh metaworld-DA

# iTHOR checkpoints  
bash download.sh ithor
```

## 📁 Expected Repository Structure
After setup, the `DTS-AVDC-114514` directory should look like:

```
DTS-AVDC-114514/
├── ckpts/                          # Pre-trained model checkpoints
├── demo/                          # Video demonstrations from simulator
├── experiment/                    # Our Draft-Target experiment scripts
├── flowdiffusion/                # Core diffusion models
│   └── flowdiffusion/
│       ├── goal_diffusion.py     # Main diffusion with Draft-Target
├── metaworld/                    # MetaWorld Benchmark  
├── download.sh                  # Checkpoint download script
├── LICENSE                      # MIT License
└── README.md                    # Current file
└── requirements.txt            # Our dependencies
└── ...                          # Other files
```

## 🚀 Running Experiments

**Important**: Make sure you have activated the conda environment:
```bash
conda activate dts_avdc
```

**Note**: If you encounter issues running bash scripts on Windows/WSL, you may need to convert line endings to LF format:
```bash
# Convert line endings if needed (only if scripts fail to run)
dos2unix experiment/*.sh
dos2unix download.sh
```

Then, navigate to the experiment directory:
```bash
cd experiment
```

### MetaWorld Manipulation Tasks
We run the MetaWorld experiments on the data augmentation checkpoint.

```bash
# Run with DA checkpoint
# make sure you have the checkpoint ../ckpts/metaworld_DA/model-24.pt
bash benchmark_mw_DA.sh 0
```

### iTHOR Navigation Tasks

```bash
# Run on iTHOR
# make sure you have the checkpoint ../ckpts/ithor/model-16.pt
bash benchmark_thor.sh 0
```

## 🤝 Acknowledgments

This codebase is modified from the following repositories:

- [AVDC](https://github.com/flow-diffusion/AVDC_experiments): Learning to Act from Actionless Videos through Dense Correspondences
- [UniMatch](https://github.com/autonomousvision/unimatch)
- [Imagen-PyTorch](https://github.com/lucidrains/imagen-pytorch)
- [Guided-Diffusion](https://github.com/openai/guided-diffusion)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
