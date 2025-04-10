# Previsor de Criptomoeda com yFinance + Telegram

Este script prevê se o preço do BTC/EUR irá subir ou descer na próxima hora com base nos indicadores MACD e RSI. Ele:
- Treina um modelo com dados de 6 meses (1h de intervalo)
- Usa MACD, RSI e Random Forest para prever
- Envia alertas via Telegram com preço estimado e direção
- Registra as previsões e compara com o valor real para calcular erro

## Requisitos

```bash
pip install -r requirements.txt
```

## Execução

```bash
python predictor.py
```

## Automatização

O script está configurado para rodar automaticamente a cada hora (loop com `time.sleep(3600)`).

### Variáveis a configurar:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Essas variáveis estão no topo do ficheiro `predictor.py`.

## Saída

- Envio de mensagem no Telegram com:
  - Direção prevista (subida ou queda)
  - Preço atual
  - Preço previsto para a próxima hora
- Gráfico com preço + sinais de MACD
- Log em `prediction_log.csv` com erro da previsão