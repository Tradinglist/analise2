import yfinance as yf
import pandas as pd
import streamlit as st
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

st.set_page_config(page_title="BTC/EUR Previsão com Machine Learning")

def get_yfinance_data(symbol="BTC-EUR", period="6mo", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

def main():
    st.title("Previsão de Preço BTC/EUR com Machine Learning")
    
    # Step 1: Load data
    df = get_yfinance_data()
    if df is not None:
        st.write(f"Dados carregados: {df.head()}")
    else:
        st.write("Erro ao carregar os dados do Yahoo Finance.")
        return

    # Step 2: Convert 'Close' to numeric and check
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        st.write(f"Coluna 'Close' convertida com sucesso!")
        st.write(df['Close'].head())
    except Exception as e:
        st.write(f"Erro ao converter 'Close' para numérico: {e}")
        return

    # Step 3: Calculate MACD
    st.write("Calculando MACD...")
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    st.write(f"MACD calculado. Primeiras linhas do dataframe:")
    st.write(df[['Close', 'macd']].head())

    # Step 4: Calculate RSI
    st.write("Calculando RSI...")
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['rsi'] = rsi.rsi()

    # Step 5: Feature engineering
    df['future_close'] = df['Close'].shift(-1)
    df['target'] = (df['future_close'] > df['Close']).astype(int)
    df.dropna(inplace=True)

    # Step 6: Train the model
    st.write("Treinando o modelo Random Forest...")
    features = ['macd', 'rsi']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.write(f"Acurácia do modelo: {accuracy:.2%}")

    # Step 7: Predict next value
    current_features = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(current_features)[0]
    direction = "📈 SUBIR" if prediction == 1 else "📉 CAIR"
    st.write(f"Previsão para a próxima hora: {direction}")

if __name__ == "__main__":
    main()
