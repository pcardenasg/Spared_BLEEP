# Spared_BLEEP
[BLEEP](https://github.com/bowang-lab/BLEEP/tree/main) state-of-the-art model adapted for [SpaRED](https://arxiv.org/abs/2407.13027#) datasets

## Environment set up
Create environment:
```
cd Spared_BLEEP
conda env create -f environment.yml -n bleep_spared
conda activate bleep_spared
pip install -r requirements.txt
```

## Running the complete EGN framework (exemplar building + gene expression prediction)
```
python BLEEP_main.py --dataset [SpaRED_dataset_name] --lr [learning_rate]
```
We invite you to explore the arguments available and their description in `utils.py` in order to modify their values as needed.