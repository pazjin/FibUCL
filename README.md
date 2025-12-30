# Uncertainty-Guided Curriculum Learning for Automated Liver Fibrosis Staging onHeterogeneous MRI
This repository implements the Uncertainty-Guided Curriculum Learning (FibUCL) framework for automated liver fibrosis staging.  FibUCL progressively incorporates samples with higher prediction uncertainty during training, allowing the model to learn from easy to hard cases while maintaining stable optimization in early stages.
## Repository Structure
```markdown
LiFS/
├── Non-Contrast/
│        ├── main.py                                         # Main entry point
│        ├── preprocess.py                                   # Preprocessing: segmentation, ROI extraction, slicing
│        ├── model_inference.py                              # Inference with pretrained models
│        └── model/                                          # Pretrained models：T1,T2,DWI_800
│              ├── best_model_{phase}{Subtask}.pt
│              └── best_attention_moe{Subtask}.pt
├── Contrast/
│        ├── main.py                                         # Main entry point
│        ├── preprocess.py                                   # Preprocessing: segmentation, ROI extraction, slicing
│        ├── model_inference.py                              # Inference with pretrained models
│        └── model/                                          # Pretrained models:GED1,GED2,GED3,GED4
│             ├── best_model_{phase}{Subtask}.pt
│             └── best_attention_moe{Subtask}.pt
```

## Repository Structure
### Clone the repo
```bash
git clone https://github.com/pazjin/FibUCL.git
cd FibUCL
```
### Install dependencies
```bash
pip install -r requirements.txt
```
## Environment

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3 (for GPU inference)

## Dataset Preparation
We follow the CARE-Liver LiFS dataset format.
```css
DATA_ROOT/
├── Vendor_A/
│       ├── patientid_A_{label}/
│               ├── T1.nii.gz
│               ├── T2.nii.gz
│               ├── DWI_800.nii.gz
│               ├── GED1.nii.gz
│               ├── GED2.nii.gz
│               ├── GED3.nii.gz
│               └── GED4.nii.gz
│       ├── patientid_A_{label}/
|               └── ...
├── Vendor_B/
|      └── ...
├── Label.csv
```

### Installation

```bash
conda create -n fibucl python=3.8 -y
conda activate fibucl
pip install -r requirements.txt
```

## Pipeline Overview
### 1️ Preprocessing (preprocess.py)

#### Reads 3D liver .nii.gz images from input folder

#### Performs automatic liver segmentation (TotalSegmentator)
```bash
/output/tempt/processed/mask
```
#### Extracts & normalizes liver ROI
```bash
/output/tempt/processed/Processed_data_cut/Image.nii.gz
```
#### Converts 3D ROI into 2D slices (PNG)
```bash
/output/tempt/processed/slices
```
#### Generates CSV metadata files (e.g., dataset_T1.csv) for model input
```bash
/output/tempt/processed/
```
### 2️⃣ Inference (model_inference.py)
#### Loads generated CSVs as model input
#### Loads pretrained model weights from:
```bash
NonContrast/model/   # T1, T2, DWI_800
Contrast/model/      # GED1-GED4
```
#### Produces prediction results:
```bash
/output/LiFS_pred.csv
```
## Run Instructions
### Preprocessing
```bash
python preprocess.py --input /path/to/liver_dataset --output /path/to/output
```
### Inference
```bash
python model_inference.py --input /path/to/output/processed_csv --output /path/to/output --task NonContrast
python model_inference.py --input /path/to/output/processed_csv --output /path/to/output --task Contrast
```
### Full Pipeline
```bash
python main.py --input /path/to/liver_dataset --output /path/to/output --task NonContrast
python main.py --input /path/to/liver_dataset --output /path/to/output --task Contrast
```
⚠️ Both tasks generate /output/LiFS_pred.csv; running the second task in the same output folder will overwrite the previous file.
## Citation
```bibtex
@inproceedings{Jin2025FibUCL,
  title     = {Uncertainty-Guided Curriculum Learning for Automated Liver Fibrosis Staging on Heterogeneous MRI},
  author    = {Yuxin Jin and Fengjun Zhao and Yanrong Chen and Xuelei He},
  booktitle = {Proceedings of the MICCAI 2025 Workshop on CARE-Liver},
  year={2025}
}
```
