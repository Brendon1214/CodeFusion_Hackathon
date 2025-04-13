# ✅ 整合版 app.py
# ✅ 功能包含：Submit Report, Heatmap, Statistics, Gallery, RAG Chat, Flood Risk Prediction with Weather + Telegram
# ✅ 深色主题兼容输出（RAG 结果）、Telegram 警报保留

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pydeck as pdk
import plotly.express as px
from PIL import Image
import requests
import joblib
from io import BytesIO

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ======================== CONFIG ========================
st.set_page_config(page_title="Disaster Flood System", page_icon="🌊", layout="wide")
DATA_FILE = "disaster_reports.csv"
IMAGE_FOLDER = "uploaded_images"
MODEL_PATH = "model/flood_predictor.pkl"
DATA_PATH = "data/sarawak_waterlevel_all_pages.csv"
API_KEY = "af23b9fb017f45eba3b72706251204"
TELEGRAM_TOKEN = "7575954980:AAECCMHpl_eQOSdav3fjSH6t2Wq8_vKJt-4"
TELEGRAM_CHAT_ID = "1975714507"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

LOCATION_COORDS = {
    "Kuching": (1.5533, 110.3592), "Sibu": (2.2870, 111.8326),
    "Miri": (4.3990, 113.9916), "Bintulu": (3.1700, 113.0300),
    "Limbang": (4.7500, 115.0000), "Sarikei": (2.1287, 111.5181),
    "Sri Aman": (1.2400, 111.4650)
}

# ======================== SIDEBAR ========================
page = st.sidebar.radio("📍 Navigation", [
    "📝 Submit Report", "📊 Statistics Dashboard", "📷 Community Gallery",
    "📽️ View Heatmap", "🔎 Ask Flood Data (RAG)", "🤖 Predict Flood Risk"
])

# ======================== PAGE: Submit Report ========================
if page == "📝 Submit Report":
    st.title("Submit Flood Report")
    location = st.selectbox("Select Location", list(LOCATION_COORDS.keys()))
    severity = st.radio("Select Severity", ["Mild", "Moderate", "Severe"])
    uploaded_file = st.file_uploader("Upload Photo (optional)", type=["jpg", "png", "jpeg"])
    description = st.text_area("Optional Description")

    if st.button("Submit"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lat, lon = LOCATION_COORDS[location]
        image_path = ""
        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.width > 1024:
                ratio = 1024 / float(image.width)
                image = image.resize((1024, int(image.height * ratio)))
            image_path = os.path.join(IMAGE_FOLDER, f"{timestamp.replace(':','-')}_{uploaded_file.name}")
            image.save(image_path)
        entry = pd.DataFrame([{
            "Location": location, "Latitude": lat, "Longitude": lon,
            "Severity": {"Mild": 1, "Moderate": 2, "Severe": 3}[severity],
            "Severity_Label": severity, "ImagePath": image_path,
            "Timestamp": timestamp, "Description": description
        }])
        entry.to_csv(DATA_FILE, mode="a", index=False, header=not os.path.exists(DATA_FILE), encoding="utf-8")
        st.success("✅ Report submitted successfully!")

# ======================== PAGE: Statistics ========================
elif page == "📊 Statistics Dashboard":
    st.title("Flood Statistics")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["Description"] = df.get("Description", "")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        st.plotly_chart(px.histogram(df, x="Severity_Label", color="Severity_Label"))
        loc_count = df["Location"].value_counts().reset_index()
        loc_count = df["Location"].value_counts().reset_index()
        loc_count.columns = ["Location", "count"]
        st.plotly_chart(px.bar(loc_count, x="Location", y="count", title="Reports by Location"))
        st.plotly_chart(px.histogram(df.dropna(), x="Timestamp", nbins=20))
    else:
        st.warning("No data yet.")

# ======================== PAGE: Gallery ========================
elif page == "📷 Community Gallery":
    st.title("Community Photo Gallery")

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["Description"] = df.get("Description", "")
        
        # ✅ 过滤出图片路径合法的行
        df = df[df["ImagePath"].apply(lambda x: isinstance(x, str) and os.path.exists(x))]

        if df.empty:
            st.info("No images found.")
        else:
            page_num = st.number_input("Page", 1, (len(df) // 4) + 1)
            start = (page_num - 1) * 4
            end = start + 4
            selected_rows = df.iloc[start:end]

            for idx, row in selected_rows.iterrows():
                st.markdown(f"### 📍 {row['Location']} - {row['Severity_Label']} ({row['Timestamp']})")
                if row['Description']:
                    st.markdown(f"_🆘 {row['Description']}_")

                try:
                    st.image(Image.open(row["ImagePath"]).resize((800, 800)))
                except:
                    st.warning("⚠️ Could not load image.")

                # ✅ 删除按钮（内部）
                if st.button(f"🗑️ Delete Photo", key=f"delete_{idx}"):
                    df_full = pd.read_csv(DATA_FILE)
                    df_full = df_full[df_full["ImagePath"] != row["ImagePath"]]
                    df_full.to_csv(DATA_FILE, index=False, encoding="utf-8")
                    if os.path.exists(row["ImagePath"]):
                        os.remove(row["ImagePath"])
                    st.success("🗑️ Photo deleted.")
                    st.rerun()

                st.markdown("---")
    else:
        st.info("⚠️ No report data found.")



# ======================== PAGE: Heatmap ========================
elif page == "📽️ View Heatmap":
    st.title("Flood Heatmap")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["Description"] = df.get("Description", "")
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(latitude=2.5, longitude=113, zoom=5.2),
            layers=[pdk.Layer("HeatmapLayer", data=df,
                get_position='[Longitude, Latitude]', get_weight='Severity', radiusPixels=60)]
        ))

# ======================== PAGE: RAG Ask ========================
elif page == "🔎 Ask Flood Data (RAG)":
    st.title("Ask Flood Data (RAG)")

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        docs = [
            Document(page_content=(
                f"Location: {r['Location']}\n"
                f"Severity: {r['Severity_Label']}\n"
                f"Timestamp: {r['Timestamp']}\n"
                f"Description: {r.get('Description', '')}"
            ))
            for _, r in df.iterrows()
        ]

        db = Chroma.from_documents(docs, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

        @st.cache_resource
        def load_llm():
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=0)  # 加速用 GPU
            return HuggingFacePipeline(pipeline=pipe)

        llm = load_llm()

        # ✅ 读取外部 prompt 文件
        with open("flood_prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = PromptTemplate.from_template(prompt_template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": prompt}
        )

        query = st.text_input("Ask anything about flood data (e.g., 'Which places had severe floods?')")
        if query:
            with st.spinner("🔍 Searching and generating answer..."):
                response = qa_chain.run(query)

            # ✨ 清洗回答内容
            lines = response.strip().split("\n")
            cleaned = lines[-1].strip()
            for prefix in ["Answer:", "Assistant:", "User:", "Rules:"]:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()
            if len(cleaned) < 3:
                cleaned = "Sorry, no relevant flood report found."

            # 💬 展示回答
            st.markdown(f"""
            <div style="padding: 15px; background-color:#111; color: #fff; border-radius: 10px; border-left: 5px solid #1e88e5;">
                <strong>📍 Answer:</strong><br>{cleaned}
            </div>
            """, unsafe_allow_html=True)

# ======================== PAGE: Flood Risk Prediction ========================
elif page == "🤖 Predict Flood Risk":
    st.title("🤖 Predict Flood Risk System")

    # 加载水位数据
    @st.cache_data
    def load_station_levels():
        df = pd.read_csv(DATA_PATH, encoding="latin1")
        df_clean = df[["Station Name", "Water Level (m)", "Danger Level (m)"]].dropna()
        df_clean["Station Name"] = df_clean["Station Name"].str.strip()
        grouped = df_clean.groupby("Station Name").first().reset_index()
        return {
            row["Station Name"]: {
                "Water Level (m)": row["Water Level (m)"],
                "Danger Level (m)": row["Danger Level (m)"]
            }
            for _, row in grouped.iterrows()
        }

    # Telegram 通知功能
    def send_telegram_alert(location, probability):
        message = f"🚨 Flood Risk Detected!\nLocation: {location}\nProbability: {probability:.2%}\nPlease take precautions."
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            res = requests.post(url, data=data)
            if res.status_code == 200:
                st.success("📱 Telegram alert sent.")
            else:
                st.warning(f"Telegram error: {res.text}")
        except Exception as e:
            st.warning(f"Telegram request failed: {e}")

    station_levels = load_station_levels()

    st.markdown("Select a location to retrieve real-time water level and forecast data. The model will predict flood risk accordingly.")

    location = st.selectbox("📍 Select a location", list(LOCATION_COORDS.keys()))

    if st.button("Predict Flood Risk"):
        with st.spinner("Retrieving weather forecast & analyzing flood risk..."):
            try:
                # 获取天气预报
                url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=3"
                response = requests.get(url)
                data = response.json()

                forecast_days = data["forecast"]["forecastday"]
                rainfall_list = [day["day"]["totalprecip_mm"] for day in forecast_days]
                dates = [day["date"] for day in forecast_days]

                rain_df = pd.DataFrame({"Date": dates, "Rainfall (mm)": rainfall_list})
                st.subheader("📅 3-Day Rainfall Forecast")
                st.table(rain_df)

                fig = px.bar(
                    rain_df,
                    x="Date",
                    y="Rainfall (mm)",
                    title="📊 Rainfall Trend",
                    color="Rainfall (mm)",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig)

                # 匹配水位站数据
                matched_station = next((s for s in station_levels if location.lower() in s.lower()), None)

                if matched_station:
                    base_water_level = station_levels[matched_station]["Water Level (m)"]
                    danger_level = station_levels[matched_station]["Danger Level (m)"]
                    alert_level = danger_level - 1.0
                else:
                    st.warning("⚠️ No water level data found for this location. Using default values.")
                    base_water_level = 3.0
                    danger_level = 4.2
                    alert_level = 3.5

                predicted_rainfall = rainfall_list[0]
                simulated_water_level = base_water_level + predicted_rainfall * 0.05

                # 展示模拟数据
                st.write(f"📈 Current Water Level: `{base_water_level:.2f} m`")
                st.write(f"📈 Predicted Water Level: `{simulated_water_level:.2f} m`")
                st.write(f"⚠️ Danger Level: `{danger_level:.2f} m`")

                # 预测
                input_df = pd.DataFrame([{
                    "Water Level (m)": simulated_water_level,
                    "Alert Level (m)": alert_level,
                    "Danger Level (m)": danger_level
                }])

                model = joblib.load(MODEL_PATH)
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                st.write(f"🧠 Model Prediction Probability: `{probability * 100:.2f}%`")

                if prediction == 1:
                    st.error("🚨 High Risk: Flood may occur!")
                    send_telegram_alert(location, probability)
                else:
                    st.success("✅ Low Risk: No flood predicted.")

                # 地图定位
                lat, lon = LOCATION_COORDS[location]
                st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=7)

            except Exception as e:
                st.error(f"❌ Error occurred: {e}")
