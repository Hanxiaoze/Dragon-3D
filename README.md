# Dragon-3D

<div align=center>
<img src="./figs/Fig1.png" width="90%" height="90%" alt="TOC" align=center />
</div>

## Section 1: Setup Environment
software/environment versions:
```
pytorch	2.4.0
pytorch-cuda	11.8
torchaudio	2.4.0
torchvision	0.19.0
torchtriton	3.0.0
torch-geometric	2.3.0
torch-cluster	1.6.3
torch-scatter	2.1.2
torch-sparse	0.6.18
torch-spline-conv	1.2.2
scikit-learn	1.7.2
scikit-image	0.25.2
tensorboard	2.20.0
absl-py	2.4.0
rdkit	2025.09.5
librdkit	2025.09.5
openbabel	3.1.1	
biopython	1.86
cctbx-base	2026.1
plip	3.0.0
mrcfile	1.5.4
munkres	1.1.4
networkx	3.4.2
numpy	2.2.6
scipy	1.15.2
pandas	2.3.3
matplotlib-base	3.10.8
sympy	1.14.0
pyvista	0.47.0
vtk	9.6.0
pillow	12.1.1
```

You can follow the instructions to setup the conda environment

```shell
conda env create -f dragon-3D_env.yml -n dragon-3D
conda activate dragon-3D
```


## Section 2: Generation

### Run Dragon-3D on the test examples
1. Please setup the env dependencies
2. Just change to the base directory and run the `zzx_Generate.py` with prepared yml file

**for denovo generation**
```shell
python zzx_Generate.py
```


### Run Dragon-3D on your own targets

For the denovo generation task, prepare your receptor PDB file and modify the example `./configs/zzx_gen_A_B_dual_example.yml` file.

- setting the `output_dir`, `receptor_A` and `receptor_B` parameters to your designated output path and receptor files.
- specifying the `x_A`, `y_A`, and `z_A` parameters as the center coordinates of the pocket_A.
- specifying the `x_B`, `y_B`, and `z_B` parameters as the center coordinates of the pocket_B.

```shell
python zzx_Generate.py --config ./configs/your.yml
```

## Section 3: Training your own dual-target drug models
1. Please modify the training file dependency paths according to your local environment.

2. Just run the `zzx_pocED_2_ligED_train_biNet_0.py`, `zzx_GPPM_train_accelerate.py`, `zzx_GFPM.py` and `zzx_GAPM_train_AMP.py` sequentially.


## Section 4: Training Dataset
The training data is located in the `datasets_and_splits` directory of this project repository.


## Random Seeds
All the random seeds used in the `zzx_pocED_2_ligED_train_biNet_0.py`, `zzx_GPPM_train_accelerate.py`, `zzx_GFPM.py` and `zzx_GAPM_train_AMP.py` is `42`



## Section 5: License

MIT


## Acknowledgement

This project is partially inspired by and built upon the ED2Mol project:

ED2Mol: https://github.com/pineappleK/ED2Mol

ED2Mol: Nat Mach Intell 7, 1355–1368 (2025). https://doi.org/10.1038/s42256-025-01095-7

We thank the original authors for making their code publicly available.

