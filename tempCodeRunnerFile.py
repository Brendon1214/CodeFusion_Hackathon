import pandas as pd
import joblib

# ====== 配置 ======
MODEL_PATH = "model/flood_by_water_model.pkl"
TEST_CSV = "data/sarawak_waterlevel_all_pages.csv"  # 你的原始 CSV 路径
OUTPUT_CSV = "data/predicted_flood_results.csv"
# ===================

# 1. 加载模型
model = joblib.load(MODEL_PATH)

# 2. 读取水位数据
df = pd.read_csv(TEST_CSV)

# 3. 转换为数值型，处理空值
for col in ["Water Level (m)", "Alert Level (m)", "Danger Level (m)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. 创建模型输入 DataFrame（只提取模型需要的列）
input_df = df[["Water Level (m)", "Alert Level (m)", "Danger Level (m)"]].copy()

# 5. 模型预测
df["Flood_Prediction"] = model.predict(input_df)

# 6. 预测概率（可选）
if hasattr(model, "predict_proba"):
    df["Flood_Probability"] = model.predict_proba(input_df)[:, 1]

# 7. 显示前几条结果
print(df[["Station Name", "Water Level (m)", "Danger Level (m)", "Flood_Prediction", "Flood_Probability"]].head())

# 8. 保存结果
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存预测结果到：{OUTPUT_CSV}")
