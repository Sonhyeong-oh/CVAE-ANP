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
import torch.nn.functional as F
import copy
import joblib

# --- CVAE 및 ANP 어텐션 모듈 import ---
from VAE.cvae_v2 import ConditionalVAE_v2
from ANP.atten import Attention

# --- 하이퍼파라미터 및 설정 ---
IMAGE_RESOLUTION = 256
ORIGINAL_IMAGE_DIMS = (1080, 720)
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 4
LATENT_DIM = 128
BETA_CVAE_LOSS = 0.5
BEST_MODEL_SAVE_PATH = "best_cvae_anp_model.pth"

# --- 경로 설정 ---
BASE_DATA_PATH = "D:/archive/Sensor_Data(Label)/EDT/sample"
PATHS = {
    'train': {
        'excel_files': [os.path.join(BASE_DATA_PATH, 'train', 'FFCell_CycleManagement_train.xlsx'), 
                        os.path.join(BASE_DATA_PATH, 'train', 'FFCellSafetyManagement_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'Conveyor_Signals_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R01_Data_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R02_Data_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R03_Data_train.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'train', 'R04_Data_train.xlsx')
                       ],
        'normal_jsons': ["D:/archive/Batch 1/data_1000.json", "D:/archive/Batch 1/data_2000.json"],
        'normal_imgs': ["D:/archive/Batch 1/BATCH1000", "D:/archive/Batch 1/BATCH2000"],
        'error_jsons': ["D:/archive/Batch 1/data_24000.json", "D:/archive/Batch 1/data_25000.json"],
        'error_imgs': ["D:/archive/Batch 1/BATCH24000", "D:/archive/Batch 1/BATCH25000"],
    },
    'val': {
        'excel_files': [os.path.join(BASE_DATA_PATH, 'val', 'FFCell_CycleManagement_val.xlsx'), 
                        os.path.join(BASE_DATA_PATH, 'val', 'FFCellSafetyManagement_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'Conveyor_Signals_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R01_Data_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R02_Data_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R03_Data_val.xlsx'),
                        os.path.join(BASE_DATA_PATH, 'val', 'R04_Data_val.xlsx')
                       ],
        'normal_jsons': ["D:/archive/Batch 1/data_3000.json"],
        'normal_imgs': ["D:/archive/Batch 1/BATCH3000"],
        'error_jsons': ["D:/archive/Batch 1/data_26000.json"],
        'error_imgs': ["D:/archive/Batch 1/BATCH26000"],
    },
}
# --- 상수 정의 (기존과 동일) ---
COMPONENT_ORDER = ['MHS', 'R01', 'R02', 'R03', 'R04', 'Conv1', 'Conv2', 'Conv3', 'Conv4']
NUM_COMPONENTS = len(COMPONENT_ORDER)
COORD_DIM = NUM_COMPONENTS * 4
BOOLEAN_SENSOR_KEYS = [ "I_SafetyDoor1_Status", "I_SafetyDoor2_Status", "I_HMI_EStop_Status", "I_MHS_GreenRocketTray", "I_Stopper1_Status", "I_Stopper2_Status", "I_Stopper3_Status", "I_Stopper4_Status", "I_Stopper5_Status" ]
_BASE_KEYS = [ "Q_VFD1_Temperature", "Q_VFD2_Temperature", "Q_VFD3_Temperature", "Q_VFD4_Temperature", "M_Conv1_Speed_mmps", "M_Conv2_Speed_mmps", "M_Conv3_Speed_mmps", "M_Conv4_Speed_mmps", "I_R01_Gripper_Pot", "I_R01_Gripper_Load", "I_R02_Gripper_Pot", "I_R02_Gripper_Load", "I_R03_Gripper_Pot", "I_R03_Gripper_Load", "I_R04_Gripper_Pot", "I_R04_Gripper_Load", "I_R01_GripperLoad_lbf", "I_R02_GripperLoad_lbf", "I_R03_GripperLoad_lbf", "I_R04_GripperLoad_lbf", "M_R01_SJointAngle_Degree", "M_R01_LJointAngle_Degree", "M_R01_UJointAngle_Degree", "M_R01_RJointAngle_Degree", "M_R01_BJointAngle_Degree", "M_R01_TJointAngle_Degree", "M_R02_SJointAngle_Degree", "M_R02_LJointAngle_Degree", "M_R02_UJointAngle_Degree", "M_R02_RJointAngle_Degree", "M_R02_BJointAngle_Degree", "M_R02_TJointAngle_Degree", "M_R03_SJointAngle_Degree", "M_R03_LJointAngle_Degree", "M_R03_UJointAngle_Degree", "M_R03_RJointAngle_Degree", "M_R03_BJointAngle_Degree", "M_R03_TJointAngle_Degree", "M_R04_SJointAngle_Degree", "M_R04_LJointAngle_Degree", "M_R04_UJointAngle_Degree", "M_R04_RJointAngle_Degree", "M_R04_BJointAngle_Degree", "M_R04_TJointAngle_Degree", "Q_Cell_CycleCount", "CycleState" ]
ALL_SENSOR_KEYS = sorted(list(set(_BASE_KEYS + BOOLEAN_SENSOR_KEYS)))
CONTINUOUS_SENSOR_KEYS = [key for key in ALL_SENSOR_KEYS if key not in BOOLEAN_SENSOR_KEYS]
COMPONENT_COORDS_1 = { 'MHS': {'h': 251, 'w': 397, 'x1': 363, 'x2': 760, 'y1': 278, 'y2': 529}, 'R01': {'h': 439, 'w': 227, 'x1': 199, 'x2': 426, 'y1': 124, 'y2': 563}, 'R02': {'h': 187, 'w': 128, 'x1': 379, 'x2': 507, 'y1': 108, 'y2': 295}, 'R03': {'h': 148, 'w': 120, 'x1': 287, 'x2': 407, 'y1': 106, 'y2': 254}, 'R04': {'h': 264, 'w': 268, 'x1': 589, 'x2': 857, 'y1': 164, 'y2': 428}, 'Conv1': {'h': 39, 'w': 183, 'x1': 179, 'x2': 362, 'y1': 178, 'y2': 217}, 'Conv2': {'h': 182, 'w': 301, 'x1': 168, 'x2': 469, 'y1': 208, 'y2': 390}, 'Conv3': {'h': 81, 'w': 215, 'x1': 457, 'x2': 672, 'y1': 264, 'y2': 345}, 'Conv4': {'h': 106, 'w': 338, 'x1': 354, 'x2': 692, 'y1': 173, 'y2': 279} }
COMPONENT_COORDS_0 = { 'MHS': {'h': 53, 'w': 183, 'x1': 232, 'x2': 415, 'y1': 182, 'y2': 235}, 'R01': {'h': 160, 'w': 99, 'x1': 385, 'x2': 484, 'y1': 102, 'y2': 262}, 'R02': {'h': 148, 'w': 140, 'x1': 273, 'x2': 413, 'y1': 143, 'y2': 291}, 'R03': {'h': 187, 'w': 173, 'x1': 372, 'x2': 545, 'y1': 161, 'y2': 348}, 'R04': {'h': 139, 'w': 131, 'x1': 173, 'x2': 304, 'y1': 145, 'y2': 284}, 'Conv1': {'h': 158, 'w': 334, 'x1': 414, 'x2': 748, 'y1': 362, 'y2': 520}, 'Conv2': {'h': 139, 'w': 346, 'x1': 380, 'x2': 726, 'y1': 210, 'y2': 349}, 'Conv3': {'h': 43, 'w': 225, 'x1': 131, 'x2': 356, 'y1': 228, 'y2': 271}, 'Conv4': {'h': 294, 'w': 380, 'x1': 91, 'x2': 471, 'y1': 268, 'y2': 562} }

# --- 헬퍼 함수 (기존과 동일) ---
def load_and_label_excel_data(excel_files):
    master_filename_base = 'FFCell_CycleManagement'
    master_file = next((f for f in excel_files if os.path.basename(f).startswith(master_filename_base)), None)
    if not master_file: raise FileNotFoundError(f"Master file '{master_filename_base}' not found.")
    master_df = pd.read_excel(master_file)
    master_df['time'] = pd.to_datetime(master_df['time'])
    master_df['label'] = master_df['Description'].notna()
    master_df['block'] = (master_df['label'] != master_df['label'].shift()).cumsum()
    labeled_periods = master_df.groupby('block').agg(start_time=('time', 'min'), end_time=('time', 'max'), is_error=('label', 'first'))
    
    dfs=[]
    for f in excel_files:
        df = pd.read_excel(f)
        df['time'] = pd.to_datetime(df['time'])
        df = df.select_dtypes(include=[np.number, 'datetime64', 'bool'])
        dfs.append(df.set_index('time'))
        
    if not dfs: raise FileNotFoundError("No valid Excel files found.")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, axis=1).sort_index()

    # Drop the problematic column as identified by the user
    if 'I_MHS_GreenRocketTray' in merged_df.columns:
        print("Dropping 'I_MHS_GreenRocketTray' column from merged Excel data.")
        merged_df.drop(columns=['I_MHS_GreenRocketTray'], inplace=True)

    # Continue with the original processing
    merged_df = merged_df.ffill().bfill().dropna().reset_index()
    
    if merged_df.empty:
        print("[WARNING] Merged dataframe is empty even after removing the problematic column. Check data for other issues.")

    normal_df_list, error_df_list = [], []
    for _, period in labeled_periods.iterrows():
        mask = (merged_df['time'] >= period['start_time']) & (merged_df['time'] <= period['end_time'])
        data_chunk = merged_df[mask]
        if period['is_error']: error_df_list.append(data_chunk)
        else: normal_df_list.append(data_chunk)
        
    normal_df = pd.concat(normal_df_list, ignore_index=True) if normal_df_list else pd.DataFrame(columns=merged_df.columns)
    error_df = pd.concat(error_df_list, ignore_index=True) if error_df_list else pd.DataFrame(columns=merged_df.columns)
    
    return normal_df, error_df
def gaussian_nll(mu, sigma, target): # ... (기존 코드와 동일)
    return (0.5 * torch.log(2 * np.pi * sigma**2) + (target - mu)**2 / (2 * sigma**2)).sum()
def get_normalized_coords_tensor(coords_map, order, dims): # ... (기존 코드와 동일)
    w, h = dims
    all_coords = []
    for name in order:
        c = coords_map.get(name, {})
        all_coords.extend([(c.get('x1',0)+c.get('x2',0))/2/w, (c.get('y1',0)+c.get('y2',0))/2/h, c.get('w',0)/w, c.get('h',0)/h])
    return torch.tensor(all_coords, dtype=torch.float32)
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 'excel_context'는 가변 길이이므로 별도 처리
    excel_contexts = [item.pop('excel_context') for item in batch]
    
    # 나머지 데이터는 default_collate로 처리
    collated_batch = torch.utils.data.dataloader.default_collate(batch)

    # excel_context를 패딩
    # pad_sequence는 (seq_len, batch, feature) 형태를 기대하므로 permute 후, 다시 permute
    padded_excel_contexts = torch.nn.utils.rnn.pad_sequence(excel_contexts, batch_first=True, padding_value=0.0)

    # 어텐션 마스크 생성 (패딩된 부분은 True)
    # 각 시퀀스의 원본 길이를 기반으로 마스크 생성
    lengths = [len(seq) for seq in excel_contexts]
    excel_mask = torch.arange(padded_excel_contexts.size(1))[None, :] >= torch.tensor(lengths)[:, None]
    
    # 최종 배치 딕셔너리에 추가
    collated_batch['excel_context'] = padded_excel_contexts
    collated_batch['excel_mask'] = excel_mask

    return collated_batch

# --- 데이터셋 클래스 (기존과 동일) ---
class MultiModalManufacturingDataset(Dataset): # ... (기존 코드와 동일)
    def __init__(self, json_files, img_dirs, excel_df, transform, continuous_scaler, excel_scaler):
        self.img_dirs=img_dirs; self.transform=transform; self.excel_df=excel_df; self.continuous_scaler=continuous_scaler; self.excel_scaler=excel_scaler
        self.samples=[]
        for i, json_file in enumerate(json_files):
            img_dir = img_dirs[i]
            if not (os.path.exists(json_file) and os.path.exists(img_dir)): continue
            with open(json_file,'r') as f: json_data=json.load(f)
            for time_str, record in json_data.items():
                if not record.get('Images'): continue
                for img_filename in record['Images']: self.samples.append({'time_str':time_str,'record':record,'img_filename':img_filename,'img_dir':img_dir})
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        sample=self.samples[idx]; record,time_str=sample['record'],sample['time_str']; current_time=pd.to_datetime(time_str)
        if self.excel_df.empty: return None
        mask=(self.excel_df['time']>=current_time)&(self.excel_df['time']<(current_time+pd.Timedelta(milliseconds=500)))
        excel_window=self.excel_df.loc[mask].drop(columns=['time']).values
        if len(excel_window)==0:
            time_diff=(self.excel_df['time']-current_time).abs()
            if time_diff.empty: return None
            excel_window=self.excel_df.iloc[[time_diff.idxmin()]].drop(columns=['time']).values
        json_vals=record.get("Sensor_values",{})
        continuous_vals=np.array([float(json_vals.get(k,0)) for k in CONTINUOUS_SENSOR_KEYS])
        boolean_vals=np.array([float(json_vals.get(k,0)) for k in BOOLEAN_SENSOR_KEYS])
        img_path=os.path.join(sample['img_dir'],os.path.basename(sample['img_filename']))
        try: img=Image.open(img_path).convert("RGB")
        except FileNotFoundError: return None
        coords,c_type=(get_normalized_coords_tensor(COMPONENT_COORDS_1,COMPONENT_ORDER,ORIGINAL_IMAGE_DIMS),1) if os.path.splitext(sample['img_filename'])[0].endswith('_1') else (get_normalized_coords_tensor(COMPONENT_COORDS_0,COMPONENT_ORDER,ORIGINAL_IMAGE_DIMS),0)
        return {"img":self.transform(img),"excel_context":torch.tensor(self.excel_scaler.transform(excel_window),dtype=torch.float32),"cont_target":torch.tensor(self.continuous_scaler.transform(continuous_vals.reshape(1,-1))[0],dtype=torch.float32),"bool_target":torch.tensor(boolean_vals,dtype=torch.float32),"coords":coords,"c_type":torch.tensor(c_type,dtype=torch.long)}

# --- 새로운 Attentive CVAE-ANP 통합 모델 ---
class Attentive_CVAE_ANP(nn.Module): # ... (기존 코드와 동일)
    def __init__(self, cvae_model, excel_dim, num_continuous, num_boolean, latent_dim=128, coord_dim=36, hidden_dim=128, n_heads=8, attention_dropout=0.1):
        super(Attentive_CVAE_ANP, self).__init__()
        self.cvae = cvae_model
        
        # 1. Excel(Context) 시계열 데이터 인코더
        self.context_encoder = nn.Sequential(nn.Linear(excel_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        # 2. CVAE 잠재벡터(Query) 프로젝터
        self.query_projector = nn.Linear(latent_dim, hidden_dim)

        # 3. Cross-Attention 모듈
        self.cross_attention = Attention(
            hidden_dim=hidden_dim,
            attention_type="ptmultihead",
            x_dim=hidden_dim,
            n_heads=n_heads,
            dropout=attention_dropout
        )
        
        # 4. 최종 예측을 위한 디코더
        decoder_input_dim = hidden_dim + latent_dim + coord_dim
        self.continuous_decoder = nn.Sequential(nn.Linear(decoder_input_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, num_continuous * 2))
        self.boolean_decoder = nn.Sequential(nn.Linear(decoder_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_boolean))

    def forward(self, img, excel_context, coords, excel_mask=None):
        # 1. CVAE로 이미지 특징(z) 추출
        cvae_outputs = self.cvae(img, condition=excel_context)
        reconstruction, _, mu_cvae, log_var_cvae = cvae_outputs
        z = self.cvae.reparameterize(mu_cvae, log_var_cvae) # z: (batch, latent_dim)

        # 2. Excel 시계열(Context) 인코딩
        context_encoded = self.context_encoder(excel_context) # -> (batch, seq_len, hidden_dim)

        # 3. 이미지 특징(Query) 프로젝션
        query = self.query_projector(z).unsqueeze(1) # -> (batch, 1, hidden_dim) 
        
        # 4. Cross-Attention 수행 (Q: 이미지, K/V: Excel 시계열)
        attended_context = self.cross_attention(context_encoded, context_encoded, query, key_padding_mask=excel_mask).squeeze(1) # -> (batch, hidden_dim)

        # 5. 최종 예측을 위한 모든 정보 결합
        # 어텐션 결과 + CVAE 잠재벡터 + 좌표 정보
        combined_representation = torch.cat([attended_context, z, coords], dim=-1)
        
        # 6. 디코더로 최종 값 예측
        continuous_params = self.continuous_decoder(combined_representation)
        boolean_logits = self.boolean_decoder(combined_representation)
        
        mu_anp, log_sigma_anp = torch.chunk(continuous_params, 2, dim=-1)
        sigma_anp = 0.1 + 0.9 * F.softplus(log_sigma_anp)
        
        return reconstruction, mu_cvae, log_var_cvae, mu_anp, sigma_anp, boolean_logits

# --- 훈련/평가 함수 (기존과 동일) ---
def run_epoch(model, dataloader, optimizer, bce_loss_fn, kld_weight, device, is_train=True):
    # ... (기존 train_cvae_anp.py의 run_epoch 함수와 완전히 동일)
    if is_train: model.train()
    else: model.eval()
    total_loss, total_anp_loss, total_cvae_loss = 0,0,0
    pbar_desc = "Training" if is_train else "Evaluating"
    pbar = tqdm(dataloader, desc=pbar_desc, leave=False)
    context_manger = torch.enable_grad() if is_train else torch.no_grad()
    with context_manger:
        for batch in pbar:
            if batch is None: continue
            
            # 명시적으로 키를 사용하여 데이터를 device로 이동
            img = batch['img'].to(device)
            excel_context = batch['excel_context'].to(device)
            cont_target = batch['cont_target'].to(device)
            bool_target = batch['bool_target'].to(device)
            coords = batch['coords'].to(device)
            excel_mask = batch['excel_mask'].to(device)
            # c_type은 현재 모델에서 사용되지 않지만, 필요 시를 위해 주석 처리
            # c_type = batch['c_type'].to(device)

            if is_train: optimizer.zero_grad()
            
            # 모델 호출 시 excel_mask 전달
            recons, mu_cvae, log_var_cvae, mu_anp, sigma_anp, bool_logits = model(img, excel_context, coords, excel_mask=excel_mask)
            
            loss_anp_cont = gaussian_nll(mu_anp, sigma_anp, cont_target)
            loss_anp_bool = bce_loss_fn(bool_logits, bool_target)
            loss_anp = loss_anp_cont + loss_anp_bool
            cvae_loss_dict = model.cvae.loss_function(recons, img, mu_cvae, log_var_cvae, M_N=kld_weight)
            loss_cvae = cvae_loss_dict['loss']
            loss = loss_anp + BETA_CVAE_LOSS * loss_cvae
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_anp_loss += loss_anp.item()
            total_cvae_loss += loss_cvae.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}", anp=f"{loss_anp.item():.2f}", cvae=f"{loss_cvae.item():.2f}")
    num_batches = len(pbar)
    return (total_loss/num_batches, total_anp_loss/num_batches, total_cvae_loss/num_batches) if num_batches > 0 else (0,0,0)

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # 데이터 로드, 스케일러 피팅, 데이터로더 생성 (기존과 동일)
    print("--- Loading Data and Fitting Scalers ---")
    normal_excel_train, error_excel_train = load_and_label_excel_data(PATHS['train']['excel_files'])
    normal_excel_val,   error_excel_val   = load_and_label_excel_data(PATHS['val']['excel_files'])
    combined_excel_train = pd.concat([normal_excel_train, error_excel_train], ignore_index=True)

    if combined_excel_train.empty:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: The combined training dataframe is empty after loading and processing.")
        print("!!! This is likely due to one of the following issues:")
        print("!!! 1. The file paths pointing to the 'D:/archive/' drive are incorrect or inaccessible.")
        print("!!! 2. The timestamps in the various source Excel files do not overlap, leading to all rows being dropped.")
        print("!!! Please check the debugging output from the 'load_and_label_excel_data' function (printed above) to diagnose the issue.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit(1) # Exit with an error

    excel_scaler = StandardScaler().fit(combined_excel_train.drop(columns=['time']).values)
    def get_sensors_from_files(json_files):
        all_sensors=[]
        for file in json_files:
            if os.path.exists(file):
                with open(file, 'r') as f: data=json.load(f)
                all_sensors.extend([[float(v["Sensor_values"].get(k,0)) for k in CONTINUOUS_SENSOR_KEYS] for v in data.values()])
        return all_sensors
    train_sensor_data = get_sensors_from_files(PATHS['train']['normal_jsons'] + PATHS['train']['error_jsons'])
    continuous_scaler = StandardScaler().fit(train_sensor_data)
    transform = transforms.Compose([transforms.Resize((IMAGE_RESOLUTION, IMAGE_RESOLUTION)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    train_ds = ConcatDataset([MultiModalManufacturingDataset(PATHS['train']['normal_jsons'], PATHS['train']['normal_imgs'], normal_excel_train, transform, continuous_scaler, excel_scaler), MultiModalManufacturingDataset(PATHS['train']['error_jsons'], PATHS['train']['error_imgs'], error_excel_train, transform, continuous_scaler, excel_scaler)])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_ds = ConcatDataset([MultiModalManufacturingDataset(PATHS['val']['normal_jsons'], PATHS['val']['normal_imgs'], normal_excel_val, transform, continuous_scaler, excel_scaler), MultiModalManufacturingDataset(PATHS['val']['error_jsons'], PATHS['val']['error_imgs'], error_excel_val, transform, continuous_scaler, excel_scaler)])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    kld_weight = BATCH_SIZE / len(train_ds)

    # --- 모델 초기화 (Attentive 모델로 변경) ---
    cvae = ConditionalVAE_v2(in_channels=3, condition_dim=excel_scaler.n_features_in_, latent_dim=LATENT_DIM, img_size=IMAGE_RESOLUTION).to(DEVICE)
    
    model = Attentive_CVAE_ANP(cvae_model=cvae,
                               excel_dim=excel_scaler.n_features_in_,
                               num_continuous=len(CONTINUOUS_SENSOR_KEYS),
                               num_boolean=len(BOOLEAN_SENSOR_KEYS),
                               latent_dim=LATENT_DIM,
                               coord_dim=COORD_DIM
                               ).to(DEVICE)
    
    # --- 모델 파라미터 수 계산 및 출력 ---
    def count_parameters(model, model_name):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in {model_name}: {total_params:,}")
        return total_params

    print("\n--- Model Architecture Summary ---")
    cvae_params = count_parameters(model.cvae, "ConditionalVAE_v2")
    total_params = count_parameters(model, "Attentive_CVAE_ANP (Total)")
    anp_params = total_params - cvae_params
    print(f"Number of trainable parameters in ANP (Attention + Decoders): {anp_params:,}")
    print("------------------------------------")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # --- 학습 루프 (기존과 동일) ---
    best_val_loss = float('inf')
    best_model_wts = None
    print("\n--- Starting Attentive CVAE-ANP Model Training ---")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_anp, train_cvae = run_epoch(model, train_loader, optimizer, bce_loss_fn, kld_weight, DEVICE, is_train=True)
        val_loss, val_anp, val_cvae = run_epoch(model, val_loader, None, bce_loss_fn, kld_weight, DEVICE, is_train=False)
        print(f"Epoch {epoch+1} Summary:\n  Train -> Total: {train_loss:.4f} | ANP: {train_anp:.4f} | CVAE: {train_cvae:.4f}\n  Val   -> Total: {val_loss:.4f} | ANP: {val_anp:.4f} | CVAE: {val_cvae:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Validation loss improved to {val_loss:.4f}. Saving best model weights.")

    print("\n--- Training Finished ---")
    if best_model_wts is not None:
        print("\n--- Final Testing with Best Model ---")
        model.load_state_dict(best_model_wts)
        # 테스트 로더가 정의되지 않았으므로, 필요 시 val 로더로 대체하거나 test 로더를 정의해야 함.
        # test_loss, test_anp, test_cvae = run_epoch(model, test_loader, None, bce_loss_fn, kld_weight, DEVICE, is_train=False)
        # print(f"Final Test Loss -> Total: {test_loss:.4f} | ANP: {test_anp:.4f} | CVAE: {test_cvae:.4f}")
        torch.save(best_model_wts, BEST_MODEL_SAVE_PATH)
        print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
        
        # Save the scalers for use in the test script
        joblib.dump(continuous_scaler, 'continuous_scaler.joblib')
        joblib.dump(excel_scaler, 'excel_scaler.joblib')
        print("Scalers saved to continuous_scaler.joblib and excel_scaler.joblib")
    else:
        print("No model was saved as validation loss did not improve.")
