import os
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import json
import pickle
import asyncio

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ====== AI (XGBoost fallback) ======
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    from sklearn.linear_model import LogisticRegression
    USE_XGB = False

TOKEN = os.getenv("TOKEN")

TOP_STOCKS = [
    "CIB","TALAAT","FWRY","EFG","SWDY",
    "ETEL","HRHO","ABUK","ORAS","EAST",
    "JUFO","AMOC","PHDC","SODIC","CCAP",
    "OLFI","KABO","EGTS","ISPH","DSCW"
]

MODEL_FILE = "model.pkl"
LAST_TRAIN_FILE = "last_train.txt"
SIGNALS_FILE = "signals.json"

# ===== DATA =====
def get_data(symbol):
    try:
        df = yf.Ticker(symbol + ".CA").history(period="6mo")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1]), df
    except:
        pass
    return None, None

# ===== FEATURES =====
def prepare(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Vol_Avg"] = df["Volume"].rolling(10).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_Avg"]
    df["Range"] = df["High"] - df["Low"]
    df["Breakout"] = (df["Close"] > df["High"].shift(5)).astype(int)

    return df.dropna()

# ===== TRAIN =====
def train(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","Momentum","MA20","MA50","Vol_Ratio","Range","Breakout"]].dropna()
    y = df["Target"].loc[X.index]

    if len(X) < 50:
        return None, None, None

    if USE_XGB:
        model = XGBClassifier(n_estimators=100, max_depth=4)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X, y)
    return model, X, y

# ===== BACKTEST =====
def backtest(model, X, y):
    preds = model.predict(X)
    return round((preds == y).mean() * 100, 2)

# ===== LOAD MODEL =====
def load_model(df):
    today = str(datetime.date.today())

    if os.path.exists(LAST_TRAIN_FILE):
        last = open(LAST_TRAIN_FILE).read()
    else:
        last = ""

    if last != today:
        model, X, y = train(df)
        if model is None:
            return None, None

        acc = backtest(model, X, y)

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        with open(LAST_TRAIN_FILE, "w") as f:
            f.write(today)

        return model, acc
    else:
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f), None
        except:
            return None, None

# ===== PREDICT =====
def predict(model, row):
    X = pd.DataFrame([[
        row["RSI"], row["Momentum"], row["MA20"], row["MA50"],
        row["Vol_Ratio"], row["Range"], row["Breakout"]
    ]], columns=[
        "RSI","Momentum","MA20","MA50",
        "Vol_Ratio","Range","Breakout"
    ])
    return model.predict_proba(X)[0][1]

# ===== ANALYZE =====
def analyze(symbol):
    price, df = get_data(symbol)
    if df is None:
        return "❌ السهم غير متاح"

    df = prepare(df)
    if df.empty:
        return "❌ بيانات غير كافية"

    model, acc = load_model(df)
    if model is None:
        return "❌ الموديل مش جاهز"

    last = df.iloc[-1]
    prob = predict(model, last)

    score = int(prob * 100)

    if prob > 0.65:
        summary = "اتجاه صاعد قوي"
    elif prob > 0.5:
        summary = "اتجاه حيادي"
    else:
        summary = "اتجاه ضعيف"

    return f"""📊 {symbol}
💰 السعر: {round(price,2)}

🤖 AI Score: {score}/100
📈 احتمال الصعود: {prob:.0%}

✍️ {summary}

⚠️ تحليل فني فقط وليس نصيحة استثمارية
"""

# ===== HANDLER =====
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()

    await update.message.reply_text("⏳ جاري التحليل...")

    result = analyze(text)

    await update.message.reply_text(result)

# ===== MAIN =====
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🚀 BOT RUNNING WORKER MODE")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()  # يخليه شغال للأبد

if __name__ == "__main__":
    asyncio.run(main())
