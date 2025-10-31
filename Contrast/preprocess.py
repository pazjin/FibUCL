'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-30 14:23:44
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-04 11:17:12
FilePath: /MICCAI_workshop/Test/Contrast_Docker/preprocess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
from PIL import Image

os.environ['TOTALSEGMENTATOR_NO_DOWNLOAD'] = 'True'

def segment_mask(nifti_path, save_path):
    wrong_list = []
    phase = nifti_path.split('/')[-1]

    # 确保权重存在
    expected_model_dir = os.path.join(os.environ.get('TOTALSEGMENTATOR_PATH', ''), 'nnunet', 'results')
    if not os.path.exists(expected_model_dir):
        print(f"⚠️ 模型路径不存在：{expected_model_dir}")
    
    try:
        print("TotalSegmentator model path:", os.environ.get('TOTALSEGMENTATOR_PATH'))
        totalsegmentator(nifti_path, save_path, task="total_mr", roi_subset=["liver"])
    except Exception as e:
        print(f"Error processing {nifti_path}: {str(e)}")
        wrong_list.append(nifti_path)

    if wrong_list:
        phase_dir = os.path.join('/output/tempt', f'Segmentator_wronglist_{phase}.csv')
        pd.DataFrame(wrong_list).to_csv(phase_dir, index=False)
        print(f"Wrong list saved to {phase_dir}")

def cut_roi(data):
    silce_list = []
    for silce in range(len(data)):
        if np.sum(data[silce])>0:
            silce_list.append(silce)
    y_list = []
    for y in range((data).shape[1]):
        if np.sum(data[:,y,:])>0:
            y_list.append(y)
    x_list = []
    for x in range((data).shape[-1]):
        if np.sum(data[:,:,x])>0:
            x_list.append(x)
    if (len(silce_list)>0)&(len(y_list)>0)&(len(x_list)>0):
        return silce_list[0],silce_list[-1],y_list[0],y_list[-1],x_list[0],x_list[-1]
    else:
        return False,False,False,False,False,False

def extract_roi(nifti_img, mask_img, roi_dir):
    wrong_list = []
    roi_path = os.path.join(roi_dir, 'Image.nii.gz')
    phase = roi_dir.split('/')[-1]
    try:
        phase_data = sitk.GetArrayFromImage(sitk.ReadImage(nifti_img))
        phase_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_img))
        phase_mask[phase_mask>=1] = 255
        z_min,z_max,y_min,y_max,x_min,x_max = cut_roi(phase_mask)
        if z_min|z_max|y_min|y_max|x_min|x_max:
            roi_data = (phase_data*(phase_mask/255).astype('uint8'))[z_min:z_max,y_min:y_max,x_min:x_max]
            roi = sitk.GetImageFromArray(roi_data)
            sitk.WriteImage(roi,roi_path)
    except:
        wrong_list.append(roi_dir)
    pd.DataFrame(wrong_list).to_csv(f'/output/tempt/Processed_data_wronglist_{phase}.csv',index=False)
    return roi_path

def save_slices(image_path, patient_id, save_dir):
    # 检查 image_path 是否存在
    saved_paths = []
    try:
        image_data = nib.load(image_path)
        image_data = image_data.get_fdata()
        for i in range(image_data.shape[-1]):
            slice_img = image_data[:, :, i]
            normalized = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255)
            img_data = normalized.astype(np.uint8)
            img = Image.fromarray(img_data)
            img = img.convert('L')
            filename = os.path.join(save_dir, f"{patient_id}_slice_{i}.png")
            img.save(filename)
            saved_paths.append(filename)
    except Exception as e:
        print(f"An error occurred while processing '{image_path}': {e}")

    return saved_paths

def process_single_patient(patient_dir, save_dir, record_list, phase, wrong_list):
    nifti_path = os.path.join(patient_dir, f"{phase}.nii.gz")
    if not os.path.exists(nifti_path):
        wrong_list.append(nifti_path)
        return
    patient_id = os.path.basename(patient_dir.strip("/"))
    center_id = patient_dir.split('/')[-2]
    mask_dir = os.path.join(save_dir, 'mask', center_id, patient_id, phase)
    os.makedirs(mask_dir, exist_ok=True)

    segment_mask(nifti_path, mask_dir)
    mask_path = os.path.join(mask_dir, "liver.nii.gz")

    roi_dir = os.path.join(save_dir, 'Processed_data_cut', center_id, patient_id, phase)
    os.makedirs(roi_dir, exist_ok=True)

    roi_path = extract_roi(nifti_path, mask_path, roi_dir)

    slice_dir = os.path.join(save_dir, "slices", center_id, patient_id, phase)
    os.makedirs(slice_dir, exist_ok=True)

    slice_paths = save_slices(roi_path, patient_id, slice_dir)
    
    for path in slice_paths:
        record_list.append({"patient_id": patient_id, "img_path": path})  
    pd.DataFrame(wrong_list).to_csv(f'/output/tempt/no_{phase}.csv',index=False)

def process_all_data(raw_dir, output_dir, csv_path, phase):
    record_list = []
    case_path = []
    wrong_list = [] 
    for vendor in os.listdir(raw_dir):
        vendor_path = os.path.join(raw_dir, vendor)
        for case in os.listdir(vendor_path):
            case_path.append(os.path.join(vendor_path, case))
    for p_dir in tqdm(case_path):
        process_single_patient(p_dir, output_dir, record_list, phase, wrong_list)

    df = pd.DataFrame(record_list)
    df.to_csv(csv_path, index=False)
