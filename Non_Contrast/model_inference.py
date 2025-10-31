import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import numpy
import os

class Dataset_loader(torch.utils.data.Dataset):
    """dataset."""

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 读取图片
        image = Image.open(self.dataframe.iloc[idx, 1])
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def predict_with_meta(model, dataloader, df_meta, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = [x.to(device) for x in batch]  # or handle as needed
            outputs = torch.sigmoid(model(*inputs)).squeeze(1)  # unpack if model expects multiple args
            preds.extend(outputs.cpu().numpy())
    df_meta = df_meta.copy()
    df_meta['pred'] = preds
    return df_meta

def slice_preds_to_patient_preds(df_preds, pred_col='pred'):
    # 按 patient_id 分组，对所有切片的预测取平均（或 max）
    df_patient = df_preds.groupby('patient_id').agg({
        pred_col: 'mean',   # 平均，也可以换成 'max'
    }).reset_index()
    return df_patient

# 3. Load model
def load_model(path, device):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)
    torch.serialization.add_safe_globals({
    'numpy.core.multiarray.scalar': numpy.core.multiarray.scalar
    })
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)  # 将模型参数和缓冲区也移动到 GPU
    return model

# 4. Attention-based Gating Network
class AttentionGatingNet(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.attn_fc = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        attn_logits = self.attn_fc(x)
        # 把缺失模态（mask=0）的位置设为 -1e9，使得 softmax 后趋近于0
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        weighted = x * attn_weights
        out = self.classifier(weighted)
        return out, attn_weights
    
def finetue_moedel(best_model_path,device):
    model = AttentionGatingNet().to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    return model

def bn_adapt(model, dataloader, device, bn_iters):
    """
    BN 自适应（可多次迭代），只更新 BN 层的均值方差，不更新权重。
    """
    model.train()  # BN 更新需要训练模式
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()  # 重置均值/方差
            m.momentum = None  # 用累积均值
    with torch.no_grad():
        for _ in range(bn_iters):
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                if torch.is_tensor(inputs):
                    inputs = inputs.to(device)
                else:
                    raise ValueError("Unexpected input type in bn_adapt")
                _ = model(inputs)  # 只前向更新 BN 统计
    return model


def tent_adapt(model, dataloader, device, tent_iters, lr):
    """
    TENT (Test-time Entropy Minimization)
    不改变 dataloader 输出格式
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(tent_iters):
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # 计算熵最小化损失
            loss = - (torch.softmax(outputs, dim=1) *
                      torch.log_softmax(outputs, dim=1)).sum(1).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    return model



def run_inference(device, best_model_path, csv_T1_path, csv_T2_path, csv_DWI_800_path,
                  model_T1_path, model_T2_path, model_DWI_800_path, bn_iters, tent_steps):
    

    model_T1 = load_model(model_T1_path,device)
    model_T2 = load_model(model_T2_path,device)
    model_DWI_800 = load_model(model_DWI_800_path,device)

    df_T1 = pd.read_csv(csv_T1_path)
    df_T2 = pd.read_csv(csv_T2_path)
    df_DWI_800 = pd.read_csv(csv_DWI_800_path)
    T1_data = Dataset_loader(csv_file=csv_T1_path,
                                transform=transforms.Compose([
                                    transforms.Resize([128, 128]),
                                    transforms.ToTensor()
                                ]))
    T2_data = Dataset_loader(csv_file=csv_T2_path,
                                transform=transforms.Compose([
                                    transforms.Resize([128, 128]),
                                    transforms.ToTensor()
                                ]))
    DWI_800_data = Dataset_loader(csv_file=csv_DWI_800_path,
                                transform=transforms.Compose([
                                    transforms.Resize([128, 128]),
                                    transforms.ToTensor()
                                ]))

    data_loader = {
    'T1': torch.utils.data.DataLoader(T1_data, batch_size=len(T1_data), shuffle=False, num_workers=0),
    'T2': torch.utils.data.DataLoader(T2_data, batch_size=len(T2_data), shuffle=False, num_workers=0),
    'DWI_800': torch.utils.data.DataLoader(DWI_800_data, batch_size=len(DWI_800_data), shuffle=False, num_workers=0)
    }

    for T1_data_all in data_loader['T1']:
        T1_dataset =  torch.utils.data.TensorDataset(T1_data_all)
    for T2_data_all in data_loader['T2']:
        T2_dataset =  torch.utils.data.TensorDataset(T2_data_all)
    for DWI_800_data_all in data_loader['DWI_800']:
        DWI_800_dataset =  torch.utils.data.TensorDataset(DWI_800_data_all)


    data_loader = {
    'T1': torch.utils.data.DataLoader(T1_dataset, batch_size=128, shuffle=False, num_workers=0),
    'T2': torch.utils.data.DataLoader(T2_dataset, batch_size=128, shuffle=False, num_workers=0),
    'DWI_800': torch.utils.data.DataLoader(DWI_800_dataset, batch_size=128, shuffle=False, num_workers=0)
    }

    T1_loader_full = DataLoader(T1_data, batch_size=len(T1_data), shuffle=False)
    T2_loader_full = DataLoader(T2_data, batch_size=len(T2_data), shuffle=False)
    DWI_loader_full = DataLoader(DWI_800_data, batch_size=len(DWI_800_data), shuffle=False)

    # OOD Adapt
    if bn_iters > 0:
        model_T1 = bn_adapt(model_T1, T1_loader_full, device, bn_iters)
        model_T2 = bn_adapt(model_T2, T2_loader_full, device, bn_iters)
        model_DWI_800 = bn_adapt(model_DWI_800, DWI_loader_full, device, bn_iters)
    if tent_steps > 0:
        model_T1 = tent_adapt(model_T1, T1_loader_full, device, tent_steps, lr=1e-4)
        model_T2 = tent_adapt(model_T2, T2_loader_full, device, tent_steps, lr=1e-4)
        model_DWI_800 = tent_adapt(model_DWI_800, DWI_loader_full, device, tent_steps, lr=1e-4)
    
    pred_T1_slice = predict_with_meta(model_T1, data_loader['T1'], df_T1, device)
    pred_T2_slice = predict_with_meta(model_T2, data_loader['T2'], df_T2, device)
    pred_DWI_800_slice = predict_with_meta(model_DWI_800, data_loader['DWI_800'], df_DWI_800, device)


    pred_T1 = slice_preds_to_patient_preds(pred_T1_slice, pred_col='pred')
    pred_T2 = slice_preds_to_patient_preds(pred_T2_slice, pred_col='pred')
    pred_DWI_800 = slice_preds_to_patient_preds(pred_DWI_800_slice, pred_col='pred')


    pred_T1.rename(columns={'pred': 'pred_T1'}, inplace=True)
    pred_T2.rename(columns={'pred': 'pred_T2'}, inplace=True)
    pred_DWI_800.rename(columns={'pred': 'pred_DWI_800'}, inplace=True)


    df_merge = pred_T1.merge(pred_T2, on=['patient_id'], how='outer', suffixes=('', '_T2'))
    df_merge = df_merge.merge(pred_DWI_800, on=['patient_id'], how='outer', suffixes=('', '_DWI_800'))


    df_merge['pred_T1'] = df_merge['pred_T1'].fillna(0)
    df_merge['pred_T2'] = df_merge['pred_T2'].fillna(0)
    df_merge['pred_DWI_800'] = df_merge['pred_DWI_800'].fillna(0)


    df_merge['mask_T1'] = df_merge['pred_T1'].apply(lambda x: 1 if x != 0 else 0)
    df_merge['mask_T2'] = df_merge['pred_T2'].apply(lambda x: 1 if x != 0 else 0)
    df_merge['mask_DWI_800'] = df_merge['pred_DWI_800'].apply(lambda x: 1 if x != 0 else 0)



    X = df_merge[['pred_T1', 'pred_T2', 'pred_DWI_800']].values.astype(np.float32)
    mask = df_merge[['mask_T1', 'mask_T2', 'mask_DWI_800']].values.astype(np.float32)
    
    model_ft = finetue_moedel(best_model_path, device)
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(device)
        M_tensor = torch.tensor(mask).to(device)
        final_preds, attn_weights = model_ft(X_tensor, M_tensor)
        final_preds = final_preds.cpu().numpy().squeeze(1)

    df_merge['Attention_MOE_prediction'] = final_preds
    df_merge['Setting'] = 'NonContrast'
    return df_merge

