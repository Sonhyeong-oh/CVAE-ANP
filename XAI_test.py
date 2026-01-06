import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import cv2
from grad_CAM.utils.image import show_cam_on_image
from grad_CAM.eigen_cam import EigenCAM
from train import VQVAE_ANP_Predictor

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


# 2. 엑셀 로드 및 데이터 정제 함수
def load_and_merge_excel(excel_files):
    dfs = []
    for f in excel_files:
        if not os.path.exists(f): continue
        df = pd.read_excel(f)
        df['time'] = pd.to_datetime(df['time'])
        df = df.select_dtypes(include=[np.number, 'datetime64', 'bool'])
        dfs.append(df.set_index('time'))
    merged_df = pd.concat(dfs, axis=1).sort_index()
    merged_df = merged_df.dropna(axis=1, how='all') # 결측치 컬럼 제거
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill').dropna()
    return merged_df.reset_index()

# 3. JSON 센서 데이터 추출 보조 함수
def get_json_sensors(path):
    with open(path, 'r') as f: data = json.load(f)
    return [[float(v["Sensor_values"].get(k, 0)) for k in SENSOR_KEYS] for v in data.values()]

# 4. EigenCAM 및 유사 이미지 시각화 함수
def get_latent_representation(model, img_tensor, device):
    """ VQ-VAE 인코더를 통해 이미지의 잠재 표현(latent representation)을 추출합니다. """
    model.eval()
    with torch.no_grad():
        qt, qb, _, _, _ = model.vqvae.encode(img_tensor.to(device))
        z_img = torch.cat([qt.mean([2, 3]), qb.mean([2, 3])], dim=-1)
    return z_img.cpu()

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, excel_seq):
        super().__init__()
        self.model = model
        self.excel_seq = excel_seq
    
    def forward(self, img):
        y_pred_tensor, _ = self.model(img, self.excel_seq)
        return y_pred_tensor

def visualize_anomaly_with_comparison_v2(model, anomaly_img_tensor, anomaly_img_path, comparison_img_path, analysis_text, device, excel_seq, save_path):
    """
    1. 이상치 Eigen-CAM
    2. 이상치 이미지 + 바운딩 박스
    3. 정상(비교군) 이미지 + 동일 위치 바운딩 박스
    4. 상세 분석 문구 출력
    + 이미지를 화면에 띄우는 대신 파일로 저장
    """
    
    # --- 1. 기본 설정 및 이미지 준비 ---
    def denormalize(tensor):
        tensor = tensor * 0.5 + 0.5
        return np.clip(tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)

    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # 이미지 로드
    anomaly_rgb = denormalize(anomaly_img_tensor)
    comp_img = Image.open(comparison_img_path).convert("RGB")
    comp_img_tensor = transform(comp_img).unsqueeze(0).to(device)
    comp_rgb = denormalize(comp_img_tensor)

    # --- 2. Eigen-CAM 생성 및 바운딩 박스 좌표 추출 ---
    cam_model = ModelWrapper(model, excel_seq)
    target_layer = cam_model.model.vqvae.enc_b
    
    with EigenCAM(model=cam_model, target_layers=[target_layer]) as cam_extractor:
        grayscale_cam = cam_extractor(input_tensor=anomaly_img_tensor, targets=None)[0, :]

    thresh = np.percentile(grayscale_cam, 90)
    mask = np.uint8(grayscale_cam > thresh) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
    else:
        x, y, w, h = 10, 10, 100, 100

    # --- 3. 이미지 위에 그리기 ---
    cam_image_anomaly = show_cam_on_image(anomaly_rgb, grayscale_cam, use_rgb=True)
    anomaly_with_box = (anomaly_rgb * 255).astype(np.uint8).copy()
    cv2.rectangle(anomaly_with_box, (x, y), (x+w, y+h), (255, 0, 0), 2)
    comp_with_box = (comp_rgb * 255).astype(np.uint8).copy()
    cv2.rectangle(comp_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # --- 4. 1x3 시각화 및 저장 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    axes[0].imshow(cam_image_anomaly)
    axes[0].set_title("[1] Anomaly Eigen-CAM\n(Model's Focus Area)", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(anomaly_with_box)
    axes[1].set_title("[2] Anomaly Part\n(Detection Box)", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(comp_with_box)
    axes[2].set_title("[3] Normal Comparison\n(Same Part, Same Pose)", fontsize=12)
    axes[2].axis('off')

    plt.figtext(0.5, 0.02, analysis_text, ha="center", fontsize=11, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":10}, wrap=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


from VAE.vqvae import VQVAE

# --- 메인 실행 (상세 원인 분석 + 편차/이미지 시각화 기능 추가) ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "robust_sensor_model.pth"
    
    # --- 1. 경로 및 임계값 설정 ---
    JSON_FILE_NORMAL = r"D:/archive/Batch 1/data_1000.json"
    IMG_DIR_NORMAL = r"D:/archive/Batch 1/BATCH1000"
    JSON_FILE_ERROR = r"D:/archive/Batch 1/data_24000.json"
    IMG_DIR_ERROR = r"D:/archive/Batch 1/BATCH24000"
    EXCEL_FILES = [
        "D:/archive/Sensor_Data(Label)/EDT/sample/Conveyor_Signals_sample.xlsx", "D:/archive/Sensor_Data(Label)/EDT/sample/FFCell_CycleManagement_sample.xlsx",
        "D:/archive/Sensor_Data(Label)/EDT/sample/FFCellSafetyManagement_sample.xlsx", "D:/archive/Sensor_Data(Label)/EDT/sample/R01_Data_sample.xlsx",
        "D:/archive/Sensor_Data(Label)/EDT/sample/R02_Data_sample.xlsx", "D:/archive/Sensor_Data(Label)/EDT/sample/R03_Data_sample.xlsx",
        "D:/archive/Sensor_Data(Label)/EDT/sample/R04_Data_sample.xlsx"
    ]
    
    ANOMALY_THRESHOLD = 50.0
    PER_SENSOR_THRESHOLD = 20.0
    VIZ_DIR = "anomaly_visualizations"
    os.makedirs(VIZ_DIR, exist_ok=True)

    # --- 2. 스케일러 재구성 및 모델 로드 ---
    print("Re-fitting scalers and loading model...")
    merged_excel_df = load_and_merge_excel(EXCEL_FILES)
    excel_scaler = StandardScaler().fit(merged_excel_df.drop(columns=['time']).values)
    all_json_sensors = get_json_sensors(JSON_FILE_NORMAL) + get_json_sensors(JSON_FILE_ERROR)
    json_scaler = StandardScaler().fit(all_json_sensors)
    
    excel_dim = merged_excel_df.shape[1] - 1
    vqvae = VQVAE(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512).to(DEVICE)
    model = VQVAE_ANP_Predictor(vqvae, excel_dim=excel_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model and scalers ready.")

    # --- 3. [추가] 정상 이미지 잠재 벡터 미리 계산 ---
    print("Pre-calculating latent vectors for normal images...")
    transform_for_latent = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    normal_image_files = [os.path.join(IMG_DIR_NORMAL, f) for f in os.listdir(IMG_DIR_NORMAL) if f.endswith(('.png', '.jpg', '.jpeg'))]
    normal_latent_vectors = {}
    for img_path in tqdm(normal_image_files, desc="Calculating latents"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform_for_latent(img).unsqueeze(0).to(DEVICE)
            latent_vec = get_latent_representation(model, img_tensor, DEVICE)
            normal_latent_vectors[img_path] = latent_vec
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
    print(f"Calculated {len(normal_latent_vectors)} latent vectors.")

    # --- 4. 통합 테스트 실행 ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    all_results, y_true, y_pred = [], [], []
    test_targets = [
        {"json_path": JSON_FILE_NORMAL, "img_dir": IMG_DIR_NORMAL, "actual_label": "Normal"},
        {"json_path": JSON_FILE_ERROR, "img_dir": IMG_DIR_ERROR, "actual_label": "Anomaly"}
    ]

    for target in test_targets:
        actual_label_str = target["actual_label"]
        json_path, img_dir = target["json_path"], target["img_dir"]
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: test_data = json.load(f)
        time_keys = sorted(list(test_data.keys()))

        print(f"\nTesting {len(time_keys)} samples from {os.path.basename(json_path)} (Label: {actual_label_str})...")
        for idx, t_key in enumerate(tqdm(time_keys)):
            record = test_data[t_key]
            
            # 데이터 준비
            current_time = pd.to_datetime(t_key)
            next_time = pd.to_datetime(time_keys[idx+1]) if idx < len(time_keys) - 1 else current_time + pd.Timedelta(milliseconds=500)
            mask = (merged_excel_df['time'] >= current_time) & (merged_excel_df['time'] < next_time)
            excel_window = merged_excel_df.loc[mask].drop(columns=['time']).values
            if len(excel_window) == 0:
                closest_idx = (merged_excel_df['time'] - current_time).abs().idxmin()
                excel_window = merged_excel_df.iloc[[closest_idx]].drop(columns=['time']).values
            excel_seq_scaled = torch.tensor(excel_scaler.transform(excel_window), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            json_vals = record.get("Sensor_values", {})
            actual_sensors = np.array([float(json_vals.get(k, 0)) for k in SENSOR_KEYS])
            json_gt_scaled = torch.tensor(json_scaler.transform(actual_sensors.reshape(1, -1))[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            img_path = os.path.join(img_dir, os.path.basename(record['Images'][0]))
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            except FileNotFoundError: continue

            # [수정] 모델 예측, 점수 및 편차 계산
            with torch.no_grad():
                y_pred_tensor, metrics = model(img_tensor, excel_seq_scaled)
                y_pred_dist = metrics.get('y_dist')
                if y_pred_dist is None:
                    print(f"Warning: 'y_dist' not found in metrics for {t_key}. Skipping.")
                    continue
                nll_score = -y_pred_dist.log_prob(json_gt_scaled).sum().item()
                
                # 예측 편차(불확실성) 계산
                predicted_std_vec = y_pred_dist.stddev.squeeze().cpu().numpy() * json_scaler.scale_
                uncertainty = np.mean(predicted_std_vec)

            predicted_label_str = "Anomaly" if nll_score > ANOMALY_THRESHOLD else "Normal"
            
            description = "Normal"
            if predicted_label_str == "Anomaly":
                analysis_text = f"⚠️ [Anomaly Detected] Time: {t_key} | Score: {nll_score:.2f}\n"

                predicted_mean_orig = json_scaler.inverse_transform(y_pred_tensor.squeeze().cpu().numpy().reshape(1, -1))[0]
                actual_orig = json_scaler.inverse_transform(json_gt_scaled.squeeze().cpu().numpy().reshape(1, -1))[0]
                
                diffs = np.abs(predicted_mean_orig - actual_orig)
                max_diff_idx = np.argmax(diffs)
                
                if diffs[max_diff_idx] > PER_SENSOR_THRESHOLD:
                    sensor_name = SENSOR_KEYS[max_diff_idx]
                    description = f"Sensor-driven anomaly: High deviation in '{sensor_name}'."
                    detail = (f"Analysis: Anomaly likely caused by '{sensor_name}'.\n"
                              f"(Pred: {predicted_mean_orig[max_diff_idx]:.2f} ± {predicted_std_vec[max_diff_idx]:.2f}, "
                              f"Actual: {actual_orig[max_diff_idx]:.2f}, Diff: {diffs[max_diff_idx]:.2f})")
                    analysis_text += detail
                else:
                    description = "Cross-validated anomaly: Deviation in image and/or multiple sensors."
                    analysis_text += "Analysis: Anomaly from multiple sensor errors or image issue."

                anomaly_latent_vec = get_latent_representation(model, img_tensor, DEVICE)
                distances = {p: torch.linalg.norm(anomaly_latent_vec - v) for p, v in normal_latent_vectors.items()}
                
                if distances:
                    comparison_img_path = min(distances, key=distances.get)
                    
                    sanitized_t_key = t_key.replace(":", "-").replace(" ", "_")
                    save_path = os.path.join(VIZ_DIR, f"anomaly_{sanitized_t_key}.png")

                    visualize_anomaly_with_comparison_v2(
                        model=model,
                        anomaly_img_tensor=img_tensor,
                        anomaly_img_path=img_path,
                        comparison_img_path=comparison_img_path,
                        analysis_text=analysis_text,
                        device=DEVICE,
                        excel_seq=excel_seq_scaled,
                        save_path=save_path
                    )
                else:
                    # 유사 이미지를 못찾은 경우, 원본 이상치 이미지만 저장
                    sanitized_t_key = t_key.replace(":", "-").replace(" ", "_")
                    save_path = os.path.join(VIZ_DIR, f"anomaly_solo_{sanitized_t_key}.png")
                    
                    plt.figure(figsize=(8, 8))
                    img_display = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
                    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                    plt.imshow(img_display)
                    plt.title(f"[Anomaly Image] Time: {t_key}")
                    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":5})
                    plt.axis('off')
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()

                print(f"  -> Anomaly visualization saved to {save_path}")
                print("-" * 50)

            y_true.append(1 if actual_label_str == "Anomaly" else 0)
            y_pred.append(1 if predicted_label_str == "Anomaly" else 0)
            all_results.append({"time": t_key, "score": nll_score, "uncertainty": uncertainty, "Predicted_Label": predicted_label_str, "Actual_Label": actual_label_str, "description": description})

    print("\n--- Model Performance Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    df_results = pd.DataFrame(all_results)
    output_filename = "anomaly_detection_results_with_metrics.xlsx"
    df_results.to_excel(output_filename, index=False)
    print(f"\n--- Detection Summary ---")
    print("Predicted distribution:\n", df_results['Predicted_Label'].value_counts())
    print(f"\nResults saved to {output_filename}")