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
### Required packages include
torch>=1.13.0,torchvision,pandas,numpy,nibabel,SimpleITK,Pillow,tqdm,totalsegmentator>=2.0,<3.0,nnunetv2==2.5.1
## Pipeline Overview
### 1️ Preprocessing (preprocess.py)
#### Reads 3D liver .nii.gz images from input folder
#### Performs automatic liver segmentation (TotalSegmentator)
→ /output/tempt/processed/mask< br >

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
@article{Jin2025FibUCL,
  title     = {Uncertainty-Guided Curriculum Learning for Automated Liver Fibrosis Staging on Heterogeneous MRI},
  author    = {Yuxin Jin and Fengjun Zhao and Yanrong Chen and Xuelei He},
  affiliation = {Northwest University, Xi’an, China; The First Affiliated Hospital of Xi’an Jiaotong University, Xi’an, China},
  year      = {2025},
  note      = {⋆ Corresponding author: Xuelei He (xueleihe@nwu.edu.cn)},
  url       = {https://github.com/pazjin/FibUCL}
}
```
