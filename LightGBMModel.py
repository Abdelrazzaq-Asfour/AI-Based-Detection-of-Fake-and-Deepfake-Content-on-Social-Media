import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    recall_score, mean_absolute_error, mean_squared_error
)
from transformers import AutoTokenizer, AutoModel
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load and clean data ---
df = pd.read_excel("/kaggle/input/fake-news-dataset/fake_News_Arbic_Content_V3_Moutasm_tamimi.xlsx", skiprows=1)
df = df.rename(columns={'date_Of_Puplish': 'Fake_News_Date'})
df = df[['Fake_News_Date', 'Platform_Name', 'Publisher', 'Fake_News_Content']].dropna()
df = df[df['Fake_News_Content'].str.strip() != ""]

# Filter for top 2 publishers for classification
top_publishers = df['Publisher'].value_counts().nlargest(2).index
df_classification = df[df['Publisher'].isin(top_publishers)].copy()

# --- Text Cleaning ---
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

df_classification['Fake_News_Content'] = df_classification['Fake_News_Content'].apply(clean_text)
df_classification = df_classification[df_classification['Fake_News_Content'].str.split().apply(len) >= 10]

# --- Date Features ---
df_classification['Fake_News_Date'] = pd.to_datetime(df_classification['Fake_News_Date'], errors='coerce')
df_classification = df_classification.dropna(subset=['Fake_News_Date'])
df_classification['year'] = df_classification['Fake_News_Date'].dt.year
df_classification['month'] = df_classification['Fake_News_Date'].dt.month
df_classification['year_norm'] = df_classification['year'] - df_classification['year'].min()
df_classification['month_norm'] = df_classification['month'] / 12

# --- Feature and Label Prep ---
X_text = df_classification['Fake_News_Content'].values
X_year = df_classification['year_norm'].values
X_month = df_classification['month_norm'].values
y = df_classification['Publisher'].values

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

X_train_text, X_test_text, X_train_year, X_test_year, X_train_month, X_test_month, y_train, y_test = train_test_split(
    X_text, X_year, X_month, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# --- Load AraBERT ---
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02").to(device)

def get_hybrid_embeddings(texts, batch_size=32, max_length=256):
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = list(texts[i:i+batch_size])
        encodings = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
            last_hidden = outputs.last_hidden_state
            attention_mask = encodings['attention_mask'].unsqueeze(-1)
            cls_token = last_hidden[:, 0, :]
            masked_hidden = last_hidden * attention_mask
            mean_pool = masked_hidden.sum(1) / attention_mask.sum(1)
            hybrid_emb = (cls_token + mean_pool) / 2
            embeddings.extend(hybrid_emb.cpu().numpy())
    return np.array(embeddings)

# --- Get Embeddings ---
X_train_embed = get_hybrid_embeddings(X_train_text)
X_test_embed = get_hybrid_embeddings(X_test_text)
X_train_final = np.hstack([X_train_embed, X_train_year.reshape(-1,1), X_train_month.reshape(-1,1)])
X_test_final = np.hstack([X_test_embed, X_test_year.reshape(-1,1), X_test_month.reshape(-1,1)])

# --- Train Multi-Class Classifier ---
clf = LGBMClassifier()
clf.fit(X_train_final, y_train)
y_pred = clf.predict(X_test_final)


df_time_series = df.copy()
df_time_series['Fake_News_Date'] = pd.to_datetime(df_time_series['Fake_News_Date'], errors='coerce')
df_time_series = df_time_series.dropna(subset=['Fake_News_Date'])


df_monthly = df_time_series.groupby(pd.Grouper(key='Fake_News_Date', freq='M')).size().reset_index(name='y')
df_monthly.rename(columns={'Fake_News_Date': 'ds'}, inplace=True)


prophet_model = Prophet()
prophet_model.fit(df_monthly)


future = prophet_model.make_future_dataframe(periods=3, freq='M')
forecast = prophet_model.predict(future)


y_true = df_monthly['y'].values
y_pred = forecast['yhat'].iloc[:len(y_true)].values


mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score,
    mean_absolute_error, mean_squared_error
)
from lightgbm import LGBMClassifier

# ============================
# ============================

start_time = time.time()
model = LGBMClassifier(verbose=-1)
model.fit(X_train_final, y_train)
end_time = time.time()
training_time = end_time - start_time

y_pred_bin = model.predict(X_test_final)

# --- Classification Report ---
accuracy_bin = accuracy_score(y_test, y_pred_bin)
report = classification_report(y_test, y_pred_bin, output_dict=True)
ber = 1 - recall_score(y_test, y_pred_bin, average='macro')

# --- Print Metrics ---
print("\n Classification Accuracy: LightGBM")
print(tabulate([["Accuracy", f"{accuracy_bin:.2%}", "Overall correctness of the model. Higher is better."]],
               headers=["Metric", "Value", "Explanation"], tablefmt="grid"))

summary_metrics = [
    ["Precision (weighted)", f"{precision:.4f}", "Weighted precision. Higher is better."],
    ["Recall (weighted)", f"{recall:.4f}", "Weighted recall. Higher is better."],
    
]
print("\n Summary Evaluation Metrics:")
print(tabulate(summary_metrics, headers=["Metric", "Value", "Explanation"], tablefmt="grid"))

print("=" * 60)
print(f" Training time: {training_time:.2f} seconds")
print("=" * 60)

# ============================
# ============================

df_all['Fake_News_Date'] = pd.to_datetime(df_all['Fake_News_Date'], errors='coerce')
forecast['ds'] = pd.to_datetime(forecast['ds'], errors='coerce')

df_grouped = df_all.groupby(df_all['Fake_News_Date'].dt.to_period('M')).size().reset_index(name='Fake_News_Count')
df_grouped['ds'] = df_grouped['Fake_News_Date'].dt.to_timestamp()

cutoff = df_grouped['ds'].max() - pd.DateOffset(months=8)
external_test_grouped = df_grouped[df_grouped['ds'] >= cutoff]

forecast['ds_period'] = forecast['ds'].dt.to_period('M')
external_test_grouped['ds_period'] = external_test_grouped['ds'].dt.to_period('M')

merged_external = pd.merge(external_test_grouped, forecast, on='ds_period', how='inner')

if merged_external.empty:
    print("⚠️ 'merged_external' is still empty. Please re-check periods or trimming.")
else:
    y_true_ext = merged_external['Fake_News_Count']
    y_pred_ext = merged_external['yhat']

    mae = mean_absolute_error(y_true_ext, y_pred_ext)
    mse = mean_squared_error(y_true_ext, y_pred_ext)
    rmse = np.sqrt(mse)

    prophet_metrics = [
        ["MAE", f"{mae:.2f}", "Mean Absolute Error. Lower is better."],
        ["MSE", f"{mse:.2f}", "Mean Squared Error. Lower is better."],
        ["RMSE", f"{rmse:.2f}", "Root Mean Squared Error. Lower is better."]
    ]

    print("\n📊 Prophet Forecasting Metrics (External Test Set):")
    print(f"📅Number of months : {external_test_grouped.shape[0]}")
    
    print(tabulate(prophet_metrics, headers=["Metric", "Value", "Explanation"], tablefmt="grid"))

# ============================
# ============================
pred_date = pd.to_datetime("2025-06")
pred_row = forecast[forecast['ds'].dt.to_period('M') == pred_date.to_period('M')]
future_month_df = df_all[df_all['Fake_News_Date'].dt.to_period('M') == pred_date.to_period('M')]

if not pred_row.empty:
    predicted_fake_news_count = int(pred_row['yhat'].values[0])
    forecast_msg = f" Forecast date used: {pred_date.date()}\n Predicted fake news count   : {predicted_fake_news_count}"
else:
    forecast_msg = " No forecast data available"

if not future_month_df.empty:
    most_common_platform = future_month_df['Platform_Name'].value_counts().idxmax() if 'Platform_Name' in future_month_df.columns else "Unknown"
    most_common_publisher = future_month_df['Publisher'].value_counts().idxmax() if 'Publisher' in future_month_df.columns else "Unknown"
    platform_msg = f" Predicted platform used in {pred_date.strftime('%Y-%m')}: {most_common_platform}"
    publisher_msg = f" Predicted publisher in {pred_date.strftime('%Y-%m')}: {most_common_publisher}"
else:
    platform_msg = f" No platform data available for {pred_date.strftime('%Y-%m')}"
    publisher_msg = f" No publisher data available for {pred_date.strftime('%Y-%m')}"

print("=" * 60)
print(f"{forecast_msg}\n{platform_msg}\n{publisher_msg}")
print("=" * 60)

# ============================
# ============================
plt.figure(figsize=(12, 6))
plt.plot(forecast['ds'], forecast['yhat'], marker='o')
plt.title('Predicted Fake News Count Over Time (Prophet Forecast)')
plt.xlabel('Date')
plt.ylabel('Predicted Fake News Count')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("This line chart shows the predicted number of fake news articles over time.")
print("It is generated using the Prophet forecasting model.")
print("Each point on the line represents the expected fake news count for a specific date.")
print("The overall trend helps identify peaks or declines in misinformation activity.")
y_pred_xgb = clf.predict(X_test_final)

y_test_labels = encoder.inverse_transform(y_test)
y_pred_labels = encoder.inverse_transform(y_pred_xgb)
class_names = encoder.classes_

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Publisher')
plt.ylabel('True Publisher')
plt.title('Confusion Matrix (Absolute)')
plt.tight_layout()
plt.savefig('confusion_matrix_absolute_named.png')
plt.show()

print("🔷 The absolute confusion matrix shows actual counts per publisher class.")
print("✅ Diagonal = correct predictions | ❌ Off-diagonal = misclassifications")

cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Publisher')
plt.ylabel('True Publisher')
plt.title('Confusion Matrix (Normalized)')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized_named.png')
plt.show()

print("📊 The normalized confusion matrix shows prediction accuracy per class as proportions.")
print("🔷 High diagonal values = strong model for that publisher.")


