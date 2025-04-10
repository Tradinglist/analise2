import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import logging
import os

# --- Configuração de Arquivo ---
PREDICTION_LOG_FILE = "prediction_log.csv"
GRAPH_FILE = "grafico.png"

# Configuração de Logs
logging.basicConfig(filename="prediction.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_yfinance_data(symbol="BTC-EUR", period="6mo", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            logging.error("Erro: Dados vazios obtidos de Yahoo Finance.")
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Erro ao obter dados do Yahoo Finance: {e}")
        return None

def load_prediction_log():
    if os.path.exists(PREDICTION_LOG_FILE):
        return pd.read_csv(PREDICTION_LOG_FILE, parse_dates=["prediction_time"])
    else:
        return pd.DataFrame(columns=["prediction_time", "predicted_price", "actual_price", "error_percent"])

def save_prediction_log(log_df):
    log_df.to_csv(PREDICTION_LOG_FILE, index=False)

def check_past_predictions(df, log_df):
    now = df.index[-1]
    updated_log = log_df.copy()
    for i, row in log_df.iterrows():
        if pd.isna(row["actual_price"]) and now >= row["prediction_time"]:
            try:
                actual = df.loc[row["prediction_time"]]["Close"]
                error = abs(actual - row["predicted_price"]) / row["predicted_price"] * 100
                updated_log.at[i, "actual_price"] = actual
                updated_log.at[i, "error_percent"] = error
                logging.info(f"✔️ Atualizado: previsão para {row['prediction_time']} -> erro: {error:.2f}%")
            except KeyError:
                continue
    return updated_log

def retrain_model(df, features, target):
    try:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"📊 Acurácia do modelo: {accuracy:.2%}")

        return model
    except Exception as e:
        logging.error(f"Erro ao treinar o modelo: {e}")
        return None

def main():
    # Obtenção de dados históricos
    df = get_yfinance_data()
    if df is None:
        return

    # Verificação do tipo de dados
    print(f"Tipo de df: {type(df)}")
    print(f"Primeiras linhas do df:\n{df.head()}")

    # Verificar o tipo da coluna 'Close'
    print(f"Tipo de df['Close']: {type(df['Close'])}")

    # Convertendo a coluna 'Close' para valores numéricos
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
    except Exception as e:
        logging.error(f"Erro ao converter 'Close' para numérico: {e}")
        return

    # Iniciar e calcular o MACD com parâmetros padrão
    try:
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
    except Exception as e:
        logging.error(f"Erro ao calcular MACD: {e}")
        return

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['rsi'] = rsi.rsi()

    # Definir alvo para o modelo (se o próximo preço de fechamento será maior)
    df['future_close'] = df['Close'].shift(-1)
    df['target'] = (df['future_close'] > df['Close']).astype(int)
    df.dropna(inplace=True)

    # Seleção de características
    features = ['macd', 'macd_signal', 'macd_diff', 'rsi', 'Volume']
    target = 'target'

    # Re-treinamento do modelo
    model = retrain_model(df, features, target)
    if model is None:
        return

    # Predição para o próximo período
    current_features = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(current_features)[0]
    current_price = df['Close'].iloc[-1]
    predicted_price = current_price * (1.01 if prediction == 1 else 0.99)
    prediction_time = df.index[-1] + pd.Timedelta(hours=1)

    direction = "📈 SUBIR" if prediction == 1 else "📉 CAIR"
    logging.info(f"Previsão: {direction}, Estimado: €{predicted_price:.2f}, Hora: {prediction_time}")

    # Registo da previsão
    log_df = load_prediction_log()
    log_df = check_past_predictions(df, log_df)
    new_row = pd.DataFrame([{
        "prediction_time": prediction_time,
        "predicted_price": predicted_price,
        "actual_price": None,
        "error_percent": None
    }])
    log_df = pd.concat([log_df, new_row], ignore_index=True)
    save_prediction_log(log_df)

    # Gráfico
    plt.figure(figsize=(14, 8))
    plt.subplot(2,1,1)
    plt.plot(df['Close'][-200:], label='Preço BTC/EUR', color='blue')
    plt.title("Preço BTC/EUR (Últimas 200h)")
    plt.ylabel("€")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(df['macd_signal'][-200:], label='MACD Signal', color='orange')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    buy_signals = df[(df['macd_signal'].shift(1) < 0) & (df['macd_signal'] > 0)]
    sell_signals = df[(df['macd_signal'].shift(1) > 0) & (df['macd_signal'] < 0)]
    plt.scatter(buy_signals.index, buy_signals['macd_signal'], marker='^', color='green', label='📈 Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['macd_signal'], marker='v', color='red', label='📉 Sell Signal')
    plt.title("MACD Signal com Sinais de Compra/Venda")
    plt.ylabel("MACD Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE)
    plt.close()

if __name__ == "__main__":
    while True:
        main()
        time.sleep(3600)  # Espera de 1 hora entre execuções
