import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 1. 엑셀 데이터 전처리 (결측치 컬럼 제거 포함) ---
def load_and_merge_excel(excel_files):
    dfs = []
    for f in excel_files:
        if not os.path.exists(f): continue
        df = pd.read_excel(f)
        df['time'] = pd.to_datetime(df['time'])
        df = df.select_dtypes(include=[np.number, 'datetime64', 'bool'])
        dfs.append(df.set_index('time'))
    
    merged_df = pd.concat(dfs, axis=1).sort_index()
    
    # [중요] 모든 값이 NaN인 컬럼 삭제 (NaN 에러의 주범)
    merged_df = merged_df.dropna(axis=1, how='all')
    
    # 나머지 결측치 처리
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    # 여전히 NaN이 남아있는 행(전체 데이터가 비어있는 초기 시점 등) 삭제
    merged_df = merged_df.dropna()
    
    print(f"Excel Merge Complete. Final Features: {merged_df.shape[1]}")
    return merged_df.reset_index()

# --- 2. 데이터셋 클래스 ---
class MultiModalManufacturingDataset(Dataset):
    def __init__(self, json_file, img_dir, excel_df, sensor_keys, transform=None, scaler_json=None, scaler_excel=None):
        self.img_dir = img_dir
        self.transform = transform
        self.excel_df = excel_df
        self.sensor_keys = sensor_keys
        
        with open(json_file, 'r') as f:
            self.json_data = json.load(f)
        
        self.time_keys = sorted(list(self.json_data.keys()))
        self.scaler_json = scaler_json
        self.scaler_excel = scaler_excel

    def __len__(self):
        return len(self.time_keys)

    def __getitem__(self, idx):
        current_time_str = self.time_keys[idx]
        record = self.json_data[current_time_str]
        
        # 1. 시간 윈도우 설정
        current_time = pd.to_datetime(current_time_str)
        if idx < len(self.time_keys) - 1:
            next_time = pd.to_datetime(self.time_keys[idx+1])
        else:
            next_time = current_time + pd.Timedelta(milliseconds=500)

        # 2. 엑셀 데이터 슬라이싱
        mask = (self.excel_df['time'] >= current_time) & (self.excel_df['time'] < next_time)
        excel_window = self.excel_df.loc[mask].drop(columns=['time']).values
        
        if len(excel_window) == 0:
            closest_idx = (self.excel_df['time'] - current_time).abs().idxmin()
            excel_window = self.excel_df.iloc[[closest_idx]].drop(columns=['time']).values

        # 3. 정규화 및 텐서 변환
        excel_context = self.scaler_excel.transform(excel_window)
        
        json_vals = record.get("Sensor_values", {})
        json_sensors = np.array([float(json_vals.get(k, 0)) for k in self.sensor_keys])
        json_target = self.scaler_json.transform(json_sensors.reshape(1, -1))[0]

        # 4. 이미지 로드
        img_filename = os.path.basename(record['Images'][0])
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(excel_context, dtype=torch.float32), torch.tensor(json_target, dtype=torch.float32)

# --- 3. 모델 정의 ---
class VQVAE_ANP_Predictor(nn.Module):
    def __init__(self, vqvae_model, excel_dim, target_dim=49, hidden_dim=128):
        super(VQVAE_ANP_Predictor, self).__init__()
        self.vqvae = vqvae_model
        # 엑셀의 차원을 JSON 타겟 차원으로 매핑하는 투영층
        self.excel_projection = nn.Linear(excel_dim, target_dim)
        
        from ANP.anp import NeuralProcess
        # Context X: 이미지 특징(128) + 시간 인덱스(1)
        self.anp = NeuralProcess(x_dim=128 + 1, y_dim=target_dim, hidden_dim=hidden_dim, latent_dim=hidden_dim)
        self.anp.norm_x = nn.Identity()
        self.anp.norm_y = nn.Identity()

    def forward(self, img, excel_seq, target_y_label=None):
        batch_size, seq_len, _ = excel_seq.shape
        # VQ-VAE 특징 추출
        with torch.no_grad():
            qt, qb, _, _, _ = self.vqvae.encode(img)
            z_img = torch.cat([qt.mean([2, 3]), qb.mean([2, 3])], dim=-1) # [B, 128]
        
        z_img_seq = z_img.unsqueeze(1).expand(-1, seq_len, -1)
        time_idx = torch.linspace(0, 1, seq_len).view(1, seq_len, 1).expand(batch_size, -1, -1).to(img.device)
        
        context_x = torch.cat([z_img_seq, time_idx], dim=-1) # [B, L, 129]
        context_y = self.excel_projection(excel_seq)        # [B, L, 49]
        
        target_x = context_x[:, -1:, :] # 마지막 시점 정보를 쿼리로 사용
        target_y = target_y_label.unsqueeze(1) if target_y_label is not None else None
        
        return self.anp(context_x, context_y, target_x, target_y=target_y)

# --- 4. 메인 학습 루프 ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SENSOR_KEYS = [
    "Q_VFD1_Temperature", "Q_VFD2_Temperature", "Q_VFD3_Temperature", "Q_VFD4_Temperature",
    "M_Conv1_Speed_mmps", "M_Conv2_Speed_mmps", "M_Conv3_Speed_mmps", "M_Conv4_Speed_mmps",
    "I_R01_Gripper_Pot", "I_R01_Gripper_Load", "I_R02_Gripper_Pot", "I_R02_Gripper_Load",
    "I_R03_Gripper_Pot", "I_R03_Gripper_Load", "I_R04_Gripper_Pot", "I_R04_Gripper_Load",
    "I_R01_GripperLoad_lbf", "I_R02_GripperLoad_lbf", "I_R03_GripperLoad_lbf", "I_R04_GripperLoad_lbf",
    "M_R01_SJointAngle_Degree", "M_R01_LJointAngle_Degree", "M_R01_UJointAngle_Degree", 
    "M_R01_RJointAngle_Degree", "M_R01_BJointAngle_Degree", "M_R01_TJointAngle_Degree",
    "M_R02_SJointAngle_Degree", "M_R02_LJointAngle_Degree", "M_R02_UJointAngle_Degree", 
    "M_R02_RJointAngle_Degree", "M_R02_BJointAngle_Degree", "M_R02_TJointAngle_Degree",
    "M_R03_SJointAngle_Degree", "M_R03_LJointAngle_Degree", "M_R03_UJointAngle_Degree", 
    "M_R03_RJointAngle_Degree", "M_R03_BJointAngle_Degree", "M_R03_TJointAngle_Degree",
    "M_R04_SJointAngle_Degree", "M_R04_LJointAngle_Degree", "M_R04_UJointAngle_Degree", 
    "M_R04_RJointAngle_Degree", "M_R04_BJointAngle_Degree", "M_R04_TJointAngle_Degree",
    "I_SafetyDoor1_Status", "I_SafetyDoor2_Status", "Q_Cell_CycleCount", "I_MHS_GreenRocketTray", "CycleState"
    ]

    # 경로 설정
    EXCEL_FILES = ["D:/archive/Sensor_Data(Label)/EDT/sample/Conveyor_Signals_sample.xlsx", 
                   "D:/archive/Sensor_Data(Label)/EDT/sample/FFCell_CycleManagement_sample.xlsx",
                   "D:/archive/Sensor_Data(Label)/EDT/sample/FFCellSafetyManagement_sample.xlsx",
                   "D:/archive/Sensor_Data(Label)/EDT/sample/R01_Data_sample.xlsx",
                   "D:/archive/Sensor_Data(Label)/EDT/sample/R02_Data_sample.xlsx",
                   "D:/archive/Sensor_Data(Label)/EDT/sample/R03_Data_sample.xlsx",
                   "D:/archive/Sensor_Data(Label)/EDT/sample/R04_Data_sample.xlsx"] # 모든 엑셀파일 리스트
    
    JSON_FILE_NORMAL = r"D:\archive\Batch 1\data_1000.json"
    IMG_DIR_NORMAL = r"D:\archive\Batch 1\BATCH1000"
    JSON_FILE_ERROR = r"D:\archive\Batch 1\data_24000.json"
    IMG_DIR_ERROR = r"D:\archive\Batch 1\BATCH24000"

    # 1. 엑셀 데이터 로드 및 정제
    merged_excel_df = load_and_merge_excel(EXCEL_FILES)
    excel_scaler = StandardScaler().fit(merged_excel_df.drop(columns=['time']).values)

    # 2. JSON 센서 데이터 로드 및 정규화
    def get_json_sensors(path):
        with open(path, 'r') as f: data = json.load(f)
        return [[float(v["Sensor_values"].get(k, 0)) for k in SENSOR_KEYS] for v in data.values()]
    
    all_json_sensors = get_json_sensors(JSON_FILE_NORMAL) + get_json_sensors(JSON_FILE_ERROR)
    json_scaler = StandardScaler().fit(all_json_sensors)

    # 3. 데이터셋 통합
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    ds_normal = MultiModalManufacturingDataset(JSON_FILE_NORMAL, IMG_DIR_NORMAL, merged_excel_df, SENSOR_KEYS, transform, json_scaler, excel_scaler)
    ds_error = MultiModalManufacturingDataset(JSON_FILE_ERROR, IMG_DIR_ERROR, merged_excel_df, SENSOR_KEYS, transform, json_scaler, excel_scaler)
    
    dataloader = DataLoader(ConcatDataset([ds_normal, ds_error]), batch_size=1, shuffle=True)

    # 4. 모델 훈련
    from VAE.vqvae import VQVAE
    vqvae = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512).to(DEVICE)
    model = VQVAE_ANP_Predictor(vqvae, excel_dim=merged_excel_df.shape[1]-1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for img, excel_seq, json_target in pbar:
            img, excel_seq, json_target = img.to(DEVICE), excel_seq.to(DEVICE), json_target.to(DEVICE)
            
            # [수치 안정성 체크]
            if torch.isnan(excel_seq).any(): continue

            optimizer.zero_grad()
            _, losses, _ = model(img, excel_seq, target_y_label=json_target)
            loss = losses['loss']
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    torch.save(model.state_dict(), "robust_sensor_model.pth")