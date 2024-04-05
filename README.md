# sprites

## Setup

```
conda create -n sprites python=3.10
pip install -r requirements.txt
```

### Dataset download

```
# Download dataset
mkdir -p data/retro-pixel-characters-generator && cd data/retro-pixel-characters-generator
kaggle datasets download -d calmness/retro-pixel-characters-generator
unzip retro-pixel-characters-generator.zip
```
