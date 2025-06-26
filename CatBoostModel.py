
import pandas as pd
import numpy as np
import re
import torch
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score,
    mean_absolute_error, mean_squared_error
)
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModel
from prophet import Prophet
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cpu")  

df = pd.read_excel("/kaggle/input/fake-news-dataset/fake_News_Arbic_Content_V3_Moutasm_tamimi.xlsx", skiprows=1)
df = df.rename(columns={'date_Of_Puplish': 'Fake_News_Date'})
df = df[['Fake_News_Date', 'Platform_Name', 'Publisher', 'Fake_News_Content']].dropna()
df = df[df['Fake_News_Content'].str.strip() != ""]

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

df['Fake_News_Content'] = df['Fake_News_Content'].apply(clean_text)
df = df[df['Fake_News_Content'].str.split().apply(len) >= 10]

df['Fake_News_Date'] = pd.to_datetime(df['Fake_News_Date'], errors='coerce')
df = df.dropna(subset=['Fake_News_Date'])
df['year'] = df['Fake_News_Date'].dt.year
df['month'] = df['Fake_News_Date'].dt.month
df['year_norm'] = df['year'] - df['year'].min()
df['month_norm'] = df['month'] / 12

top_publishers = df['Publisher'].value_counts().nlargest(2).index
df_top = df[df['Publisher'].isin(top_publishers)].copy()

X_text = df_top['Fake_News_Content'].values
X_year = df_top['year_norm'].values
X_month = df_top['month_norm'].values
y_pub = df_top['Publisher'].values
encoder = LabelEncoder()
y_encoded_pub = encoder.fit_transform(y_pub)

X_train_text, X_test_text, X_train_year, X_test_year, X_train_month, X_test_month, y_train_pub, y_test_pub = train_test_split(
    X_text, X_year, X_month, y_encoded_pub, test_size=0.2, stratify=y_encoded_pub, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02").to(device)

def get_hybrid_embeddings(texts, batch_size=32, max_length=256):
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = list(texts[i:i+batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state
            attn_mask = enc['attention_mask'].unsqueeze(-1)
            cls_token = out[:, 0, :]
            mean_pool = (out * attn_mask).sum(1) / attn_mask.sum(1)
            hybrid = (cls_token + mean_pool) / 2
            embeddings.extend(hybrid.cpu().numpy())
    return np.array(embeddings)

X_train_embed = get_hybrid_embeddings(X_train_text)
X_test_embed = get_hybrid_embeddings(X_test_text)

X_train_final = np.hstack([X_train_embed, X_train_year.reshape(-1, 1), X_train_month.reshape(-1, 1)])
X_test_final = np.hstack([X_test_embed, X_test_year.reshape(-1, 1), X_test_month.reshape(-1, 1)])


import pandas as pd
import numpy as np
import time
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, recall_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostClassifier
from prophet import Prophet
from tabulate import tabulate

df['Fake_News_Date'] = pd.to_datetime(df['Fake_News_Date'], errors='coerce')
df_all = df.copy()  

df['Binary_Label'] = df['Publisher'].apply(
    lambda p: 'Digital' if p in ['Social Media Group/Page', 'Website', 'Youtube'] else 'Non-Digital'
)
X_bin = df['Fake_News_Content']
y_bin = LabelEncoder().fit_transform(df['Binary_Label'])

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_bin)
X_test_tfidf = vectorizer.transform(X_test_bin)

catboost = CatBoostClassifier(random_seed=42, verbose=0)
param_dist = {
    'iterations': [100, 200, 300],
    'depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5]
}

random_search = RandomizedSearchCV(catboost, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

start_time = time.time()
random_search.fit(X_train_tfidf, y_train_bin)
end_time = time.time()
training_time = end_time - start_time

y_pred_bin = random_search.best_estimator_.predict(X_test_tfidf)
report = classification_report(y_test_bin, y_pred_bin, output_dict=True)
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
ber = 1 - recall_score(y_test_bin, y_pred_bin, average='macro')

print(f"\n🎯 accuracy CatBoost : {accuracy_score(y_test_bin, y_pred_bin) * 100:.2f}%")
print("=" * 60)
print(f" Training time: {training_time:.2f} seconds")
print("=" * 60)
summary_metrics = [
    ["Precision (weighted)", f"{precision:.4f}", "Weighted precision. Higher is better."],
    ["Recall (weighted)", f"{recall:.4f}", "Weighted recall. Higher is better."],
   
]
print("\n Summary Evaluation Metrics:")
print(tabulate(summary_metrics, headers=["Metric", "Value", "Explanation"], tablefmt="grid"))

# ✅ Prophet Forecast Evaluation
df_grouped = df.groupby(df['Fake_News_Date'].dt.to_period("M")).size().reset_index(name='Fake_News_Count')
df_grouped['ds'] = df_grouped['Fake_News_Date'].dt.to_timestamp()
df_grouped = df_grouped[['ds', 'Fake_News_Count']]

model_prophet = Prophet()
model_prophet.fit(df_grouped.rename(columns={'Fake_News_Count': 'y'}))
future = model_prophet.make_future_dataframe(periods=6, freq='M')
forecast = model_prophet.predict(future)

cutoff_period = df_grouped['ds'].max() - pd.DateOffset(months=5)
external_test_grouped = df_grouped[df_grouped['ds'] >= cutoff_period]

forecast['ds'] = pd.to_datetime(forecast['ds']).dt.to_period('M').dt.to_timestamp()
merged_external = pd.merge(forecast, external_test_grouped, on='ds', how='inner')

y_true_external = merged_external['Fake_News_Count']
y_pred_external = merged_external['yhat']

mae_external = mean_absolute_error(y_true_external, y_pred_external)
mse_external = mean_squared_error(y_true_external, y_pred_external)
rmse_external = np.sqrt(mse_external)

external_metrics = [
    ["MAE", f"{mae_external:.2f}", "Mean Absolute Error. Lower is better."],
    ["MSE", f"{mse_external:.2f}", "Mean Squared Error. Lower is better."],
    ["RMSE", f"{rmse_external:.2f}", "Root Mean Squared Error. Lower is better."]
]

print("\n📊 Prophet Forecasting Metrics - External Test Set:")
print(f"📅Number of months : {external_test_grouped.shape[0]}")
print(tabulate(external_metrics, headers=["Metric", "Value", "Explanation"], tablefmt="grid"))

# ✅ تنبؤات مخصصة لشهر معين
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

from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

y_train_publisher = y_train_pub
y_test_publisher = y_test_pub

clf = CatBoostClassifier(verbose=0, iterations=300, depth=6, learning_rate=0.03, random_seed=42)
clf.fit(X_train_final, y_train_publisher)

y_pred = clf.predict(X_test_final)

y_test_labels = encoder.inverse_transform(y_test_publisher)
y_pred_labels = encoder.inverse_transform(y_pred)
class_names = encoder.classes_

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Publisher')
plt.ylabel('True Publisher')
plt.title('Confusion Matrix (Absolute)')
plt.tight_layout()
plt.savefig('confusion_matrix_absolute_arabert_catboost.png')
plt.show()

print("🔷 The absolute confusion matrix shows actual counts per publisher class.")
print("✅ Diagonal = correct predictions | ❌ Off-diagonal = misclassifications")

cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Publisher')
plt.ylabel('True Publisher')
plt.title('Confusion Matrix (Normalized)')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized_arabert_catboost.png')
plt.show()

print("📊 The normalized confusion matrix shows prediction accuracy per publisher as proportions.")
print("🔁 Publishers used:", list(class_names))

