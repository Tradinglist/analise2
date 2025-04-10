import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import streamlit as st

# --- Configuração de Arquivo ---
PREDICTION_LOG_FILE = "prediction_log.csv"
GRAPH_FILE = "grafico.png"

# Configuração de Logs
st.set_page_config(page_title="BTC/EUR Previsão com Machine Learning")

def get_yfinance_data(symbol="BTC-EUR", period="6mo", interval="1h"):
    try:
        st.write(f"Obtendo dados para o símbolo {symbol}...")
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            st.write("Erro: Dados vazios obtidos de Yahoo Finance.")
            return None
        df.dropna(inplace=True)
        st.write(f"Dados obtidos com sucesso. Número de linhas: {len(df)}")
        return df
    except Exception as e:
        st.write(f"Erro ao obter dados do Yahoo Finance: {e}")
        return None

def load_prediction_log():
    if os.path.exists(PREDICTION_LOG_FILE):
        return pd.read_csv(PREDICTION_LOG_FILE, parse_dates=["prediction_time"])
    else:
        return pd.DataFrame(columns=["prediction_time", "predicted_price", "actual_price", "error_percent"])

def save_prediction_log(log_df):
    log_df.to_csv(PREDICTION_LOG_FILE, index=False)

def retrain_model(df, features, target):
    try:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.write(f"📊 Acurácia do modelo: {accuracy:.2%}")
        return model
    except Exception as e:
        st.write(f"Erro ao treinar o modelo: {e}")
        return None

def main():
    st.title("Previsão de Preço BTC/EUR com Machine Learning")
    
    # Obtenção de dados históricos
    df = get_yfinance_data()
    if df is None:
        st.write("Erro: Não foi possível obter dados.")
        return

    # Verificação do tipo de dados
    st.write(f"Tipo de df: {type(df)}")
    st.write(f"Primeiras linhas do df:")
    st.write(df.head())

    # Convertendo a coluna 'Close' para valores numéricos
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        st.write(f"Dados da coluna 'Close' após conversão:")
        st.write(df['Close'].head())
    except Exception as e:
        st.write(f"Erro ao converter 'Close' para numérico: {e}")
        return

    # Iniciar e calcular o MACD com parâmetros padrão
    try:
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        st.write(f"MACD calculado com sucesso!")
    except Exception as e:
        st.write(f"Erro ao calcular MACD: {e}")
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
        st.write("Erro: Não foi possível treinar o modelo.")
        return

    # Predição para o próximo período
    current_features = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(current_features)[0]
    current_price = df['Close'].iloc[-1]
    predicted_price = current_price * (1.01 if prediction == 1 else 0.99)
    prediction_time = df.index[-1] + pd.Timedelta(hours=1)

    direction = "📈 SUBIR" if prediction == 1 else "📉 CAIR"
    st.write(f"Previsão: {direction}, Estimado: €{predicted_price:.2f}, Hora: {prediction_time}")

    # Registo da previsão
    log_df = load_prediction_log()
    new_row = pd.DataFrame([{
        "prediction_time": prediction_time,
        "predicted_price": predicted_price,
        "actual_price": None,
        "error_percent": None
    }])
    log_df = pd.concat([log_df, new_row], ignore_index=True)
    save_prediction_log(log_df)

    # Gráfico
    try:
        fig, ax = plt.subplots(2, 1, figsize=(14, 8))

        ax[0].plot(df['Close'][-200:], label='Preço BTC/EUR', color='blue')
        ax[0].set_title("Preço BTC/EUR (Últimas 200h)")
        ax[0].set_ylabel("€")
        ax[0].legend()

        ax[1].plot(df['macd_signal'][-200:], label='MACD Signal', color='orange')
        ax[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        buy_signals = df[(df['macd_signal'].shift(1) < 0) & (df['macd_signal'] > 0)]
        sell_signals = df[(df['macd_signal'].shift(1) > 0) & (df['macd_signal'] < 0)]
        ax[1].scatter(buy_signals.index, buy_signals['macd_signal'], marker='^', color='green', label='📈 Buy Signal')
        ax[1].scatter(sell_signals.index, sell_signals['macd_signal'], marker='v', color='red', label='📉 Sell Signal')
        ax[1].set_title("MACD Signal com Sinais de Compra/Venda")
        ax[1].set_ylabel("MACD Signal")
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Erro ao gerar gráfico: {e}")

if __name__ == "__main__":
    main()
