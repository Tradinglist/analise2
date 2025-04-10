
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import requests
import os

# --- Configuração Telegram ---
TELEGRAM_BOT_TOKEN = "TEU_BOT_TOKEN_AQUI"
TELEGRAM_CHAT_ID = "TEU_CHAT_ID_AQUI"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Erro ao enviar mensagem:", e)

def get_yfinance_data(symbol="BTC-EUR", period="6mo", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def load_prediction_log():
    if os.path.exists("prediction_log.csv"):
        return pd.read_csv("prediction_log.csv", parse_dates=["prediction_time"])
    else:
        return pd.DataFrame(columns=["prediction_time", "predicted_price", "actual_price", "error_percent"])

def save_prediction_log(log_df):
    log_df.to_csv("prediction_log.csv", index=False)

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
                print(f"✔️ Atualizado: previsão para {row['prediction_time']} -> erro: {error:.2f}%")
            except KeyError:
                continue
    return updated_log

def main():
    df = get_yfinance_data()
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['rsi'] = rsi.rsi()

    df['future_close'] = df['Close'].shift(-1)
    df['target'] = (df['future_close'] > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['macd', 'macd_signal', 'macd_diff', 'rsi', 'Volume']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"📊 Acurácia: {accuracy:.2%}")

    current_features = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(current_features)[0]
    current_price = df['Close'].iloc[-1]
    predicted_price = current_price * (1.01 if prediction == 1 else 0.99)
    prediction_time = df.index[-1] + pd.Timedelta(hours=1)

    direction = "📈 SUBIR" if prediction == 1 else "📉 CAIR"
    print(f"Previsão: {direction}, Estimado: €{predicted_price:.2f}, Hora: {prediction_time}")

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

    # Mensagem Telegram
    telegram_msg = f"""
📊 Previsão BTC/EUR (próxima hora):
➡️ Direção: {direction}
💶 Preço atual: €{current_price:.2f}
📅 Preço estimado: €{predicted_price:.2f}
🕒 Hora: {prediction_time.strftime('%Y-%m-%d %H:%M')}
"""
    send_telegram_message(telegram_msg.strip())

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
    plt.savefig("grafico.png")
    plt.close()

if __name__ == "__main__":
    while True:
        main()
        time.sleep(3600)
