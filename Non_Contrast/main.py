'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-30 15:26:28
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-11 09:54:49
FilePath: /MICCAI_workshop/Test/Contrast_Docker/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from preprocess import process_all_data
from model_inference import run_inference
import os
import pandas as pd
import torch

if __name__ == "__main__":
    input_dir = "/input"
    output_dir = "/output/tempt/processed"
    os.makedirs(output_dir, exist_ok=True)
    # Step 1: 数据预处理（分割、裁剪、切片、表格生成）
    for phase in ['T1', 'T2', 'DWI_800']:
        csv_path = os.path.join(output_dir, f"dataset_{phase}.csv")
        process_all_data(input_dir, output_dir, csv_path, phase)
    # Step 2: 模型预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_T1_path = os.path.join(output_dir, f"dataset_T1.csv")
    csv_T2_path = os.path.join(output_dir, f"dataset_T2.csv")
    csv_DWI_800_path = os.path.join(output_dir, f"dataset_DWI_800.csv")
    for task in ['Subtask_1','Subtask_2']:
        if task == 'Subtask_1':
            best_model_path = './model/best_attention_moe_Subtask_1.pt'
            model_T1_path = "./model/best_model_T1_Subtask_1.pt"
            model_T2_path = "./model/best_model_T2_Subtask_1.pt"
            model_DWI_800_path = "./model/best_model_DWI_800_Subtask_1.pt"
            subtask_1 = run_inference(device, best_model_path,
                          csv_T1_path, csv_T2_path, csv_DWI_800_path,
                          model_T1_path, model_T2_path,model_DWI_800_path,
                          bn_iters=3, tent_steps=1)
        if task == 'Subtask_2':
            best_model_path = './model/best_attention_moe_Subtask_2.pt'
            model_T1_path = "./model/best_model_T1_Subtask_2.pt"
            model_T2_path = "./model/best_model_T2_Subtask_2.pt"
            model_DWI_800_path = "./model/best_model_DWI_800_Subtask_2.pt"
            subtask_2 = run_inference(device, best_model_path,
                          csv_T1_path, csv_T2_path, csv_DWI_800_path,
                          model_T1_path, model_T2_path,model_DWI_800_path,
                          bn_iters=3, tent_steps=1)
    subtask_1.rename(columns={'Attention_MOE_prediction': 'Subtask1_prob_S4'}, inplace=True)
    subtask_2.rename(columns={'Attention_MOE_prediction': 'Subtask2_prob_S1'}, inplace=True)
    df_merT = pd.merge(subtask_1, subtask_2, on=['patient_id', 'Setting'], how='outer', suffixes=('_sub1', '_sub2'))
    df_merT.rename(columns={'patient_id': 'Case'}, inplace=True)
    # 重新排列列的顺序
    df_merT = df_merT[['Case', 'Setting', 'Subtask1_prob_S4', 'Subtask2_prob_S1']]
    # 保存合并后的表格
    df_merT.to_csv('/output/LiFS_pred.csv', index=False)
    
