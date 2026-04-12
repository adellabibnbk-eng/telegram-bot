import os
import pandas as pd
import yfinance as yf
import numpy as np

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.linear_model import LogisticRegression

TOKEN = os.getenv("TOKEN")

def get_data(symbol):
    try:
        stock = yf.Ticker(symbol + ".CA")
        df = stock.history(period="6mo")
        if not df.empty:
            return float(df["Close"].iloc[-1]), df
    except:
        pass
    return None, None

def calculate(df):
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df.dropna()

def pivot_levels(df):
    last = df.iloc[-1]
    h, l, c = last["High"], last["Low"], last["Close"]

    pivot = (h + l + c) / 3

    r1 = (2 * pivot) - l
    s1 = (2 * pivot) - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)
    r3 = h + 2 * (pivot - l)
    s3 = l - 2 * (h - pivot)

    return round(s1,2), round(s2,2), round(s3,2), round(r1,2), round(r2,2), round(r3,2)

def train_ai(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    X = df[["RSI","MACD","EMA50","EMA200"]].dropna()
    y = df["Target"].loc[X.index]

    model = LogisticRegression()
    model.fit(X,y)
    return model

def predict_ai(model, last):
    X = np.array([[last["RSI"], last["MACD"], last["EMA50"], last["EMA200"]]])
    return model.predict_proba(X)[0][1]

def analyze(df):
    last = df.iloc[-1]

    trend = "صاعد" if last["EMA50"] > last["EMA200"] else "هابط"

    score = 0
    if trend == "صاعد": score += 30
    if last["MACD"] > last["Signal"]: score += 30
    if 40 < last["RSI"] < 60: score += 20

    if score >= 60:
        signal = "شراء 🔥"
    elif score <= 30:
        signal = "بيع ❌"
    else:
        signal = "انتظار ⏸"

    return signal, trend, score, last

# ================= BOT =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 ابعت سهم زي CIB")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.upper()

    price, df = get_data(symbol)

    if df is None:
        await update.message.reply_text("❌ مش لاقي السهم")
        return

    df = calculate(df)
    model = train_ai(df)
    prob = predict_ai(model, df.iloc[-1])

    signal, trend, score, last = analyze(df)
    s1,s2,s3,r1,r2,r3 = pivot_levels(df)

    msg = f"""📊 {symbol}
💰 {round(price,2)}

RSI: {last['RSI']:.2f}
MACD: {last['MACD']:.2f}

📉 {trend}
📊 Score: {score}

🤖 {prob:.2%}
🔥 {signal}

🟢 {s1}/{s2}/{s3}
🔴 {r1}/{r2}/{r3}
"""

    await update.message.reply_text(msg)

# ================= RUN =================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

print("🚀 BOT RUNNING")

app.run_polling(drop_pending_updates=True)
