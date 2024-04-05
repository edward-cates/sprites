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
# Find and combine pictures into 1 folder (recursively find), renaming to image count `0000.png`.
mkdir ../sprites
i=0; for f in $(find ./data -name '*.png'); do cp $f ../sprites/$(printf "%04d" $i).png; i=$((i+1)); done
```
