!pip install requests pandas matplotlib seaborn scikit-learn statsmodels

import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import drive
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import bootstrap

drive.mount('/content/drive')
DATA_PATH = '/content/drive/MyDrive/data/'
OUTPUT_PATH = '/content/drive/MyDrive/output/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

sheet_url = "https://docs.google.com/spreadsheets/d/1ph_tzoDnLRvzXJNon5fAULSUtBH_u11aNLg2uw0Riqc/export?format=csv"
data = pd.read_csv(sheet_url)

API_KEY = "sua_chave_api_qwen2_5_max"
url = "https://api.qwen.com/v1/analyze"

def analyze_text(text):
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, json={"text": text}, timeout=10)
        response.raise_for_status()
        return response.json()["analysis"]
    except requests.exceptions.RequestException as e:
        print(f"Erro: {e}")
        return "Análise indisponível"

data['analysis'] = data.apply(lambda row: analyze_text(f"Participante {row['participant_id']} do grupo {row['grupo']} com ansiedade {row['ansiedade']}, alpha wave {row['alpha_wave']}, HRV {row['hrv']} e EDA {row['eda']}"), axis=1)

scaler = MinMaxScaler()
data[['ansiedade', 'alpha_wave', 'hrv', 'eda']] = scaler.fit_transform(data[['ansiedade', 'alpha_wave', 'hrv', 'eda']])

with open(os.path.join(OUTPUT_PATH, "sumario.txt"), 'w') as f:
    f.write("Sumário Estatístico:\n")
    f.write(data.describe().to_string())
    f.write("\n\nCorrelação:\n")
    f.write(str(data[['ansiedade', 'alpha_wave', 'hrv', 'eda']].corr()))

plt.style.use('dark_background')
neon_colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FFA500', '#00FF00', '#FF1493']

plt.figure(figsize=(10, 6))
sns.kdeplot(data['ansiedade'], color=neon_colors[0], label='Ansiedade')
sns.kdeplot(data['alpha_wave'], color=neon_colors[1], label='Alpha Wave')
plt.title('KDE de Ansiedade e Alpha Wave (Normalizados)', color='white')
plt.xlabel('Valores Normalizados', color='white')
plt.ylabel('Densidade', color='white')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, "kde_ansiedade_alpha.png"), bbox_inches='tight', facecolor='black')
plt.close()

plt.figure(figsize=(10, 6))
sns.violinplot(x="grupo", y="ansiedade", data=data, palette=neon_colors[:2])
plt.title("Violin Plot de Ansiedade por Grupo", color='white')
plt.ylabel("Ansiedade (Normalizada)", color='white')
plt.xlabel("Grupo", color='white')
plt.savefig(os.path.join(OUTPUT_PATH, "violin_ansiedade_grupo.png"), bbox_inches='tight', facecolor='black')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x="grupo", y="hrv", data=data, palette=neon_colors[:2])
plt.title("Boxplot de HRV por Grupo", color='white')
plt.ylabel("HRV (Normalizado)", color='white')
plt.xlabel("Grupo", color='white')
plt.savefig(os.path.join(OUTPUT_PATH, "boxplot_hrv_grupo.png"), bbox_inches='tight', facecolor='black')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="alpha_wave", y="hrv", hue="grupo", size="ansiedade", data=data, palette=neon_colors[:2])
plt.title("Scatterplot: Alpha Wave vs HRV", color='white')
plt.xlabel("Alpha Wave (Normalizado)", color='white')
plt.ylabel("HRV (Normalizado)", color='white')
plt.savefig(os.path.join(OUTPUT_PATH, "scatterplot_alpha_hrv.png"), bbox_inches='tight', facecolor='black')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x="grupo", y="eda", data=data, palette=neon_colors[:2])
plt.title("Barplot: EDA por Grupo", color='white')
plt.xlabel("Grupo", color='white')
plt.ylabel("EDA (Normalizado)", color='white')
plt.savefig(os.path.join(OUTPUT_PATH, "barplot_eda_grupo.png"), bbox_inches='tight', facecolor='black')
plt.close()

# Regressão Linear Múltipla
X = data[['alpha_wave', 'hrv', 'eda']]
y = data['ansiedade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (Regressão Linear): {mse}")
print(f"R-squared (Regressão Linear): {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color=neon_colors[4], label=f'Predito (MSE={mse:.2f}, R²={r2:.2f})')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal')
plt.xlabel("Valores Reais de Ansiedade", color='white')
plt.ylabel("Valores Preditos de Ansiedade", color='white')
plt.title("Regressão Linear Múltipla: Valores Reais vs. Preditos", color='white')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, "linear_regression.png"), bbox_inches='tight', facecolor='black')
plt.close()

# Bootstrap para Diferença de Médias de Ansiedade entre Grupos
def bootstrap_ci(data, statistic, alpha=0.05, n_resamples=500):
    resamples = np.random.choice(data, size=(n_resamples, len(data)), replace=True)
    stats = np.apply_along_axis(statistic, 1, resamples)
    stats.sort()
    lower_idx = int(n_resamples * alpha / 2)
    upper_idx = int(n_resamples * (1 - alpha / 2))
    return stats[lower_idx], stats[upper_idx]

def mean_diff(data1, data2):
    return np.mean(data1) - np.mean(data2)

grupo_1_ansiedade = data[data['grupo'] == 'Grupo 1 (Tratamento)']['ansiedade']
grupo_2_ansiedade = data[data['grupo'] == 'Grupo 2 (Controle)']['ansiedade']

if not grupo_1_ansiedade.empty and not grupo_2_ansiedade.empty:
    ci_diff = bootstrap_ci(np.concatenate([grupo_1_ansiedade.values, grupo_2_ansiedade.values]), lambda x: mean_diff(x[:len(grupo_1_ansiedade)], x[len(grupo_1_ansiedade):]), n_resamples=500)
    print(f"Intervalo de Confiança Bootstrap (Diferença de Médias Ansiedade): {ci_diff}")
else:
    print("Não foi possível calcular o intervalo de confiança bootstrap devido a dados insuficientes.")

# Teste t independente (comparação de ansiedade entre grupos)
if not grupo_1_ansiedade.empty and not grupo_2_ansiedade.empty:
    t_statistic, p_value = ttest_ind(grupo_1_ansiedade, grupo_2_ansiedade)
    print(f"Teste t independente: Estatística={t_statistic:.2f}, p-valor={p_value:.3f}")
else:
    print("Não foi possível realizar o teste t devido a dados insuficientes.")

data.to_csv(os.path.join(OUTPUT_PATH, "processed_data.csv"), index=False)
print("Notebook concluído. Resultados salvos em processed_data.csv, sumario.txt e gráficos PNG.")
