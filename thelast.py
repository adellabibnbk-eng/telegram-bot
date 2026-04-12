import os
import pandas as pd
import yfinance as yf
import numpy as np
import pytz

from datetime import time
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from sklearn.linear_model import LogisticRegression

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TOP_STOCKS = [
    "CIB","TALAAT","FWRY","EFG","SWDY",
    "ETEL","HRHO","ABUK","ORAS","EAST",
    "JUFO","AMOC","PHDC","SODIC","CCAP",
    "OLFI","KABO","EGTS","ISPH","DSCW"
]

# ================= DATA =================
def get_data(symbol):
    try:
        stock = yf.Ticker(symbol + ".CA")
        df = stock.history(period="6mo")
        if not df.empty:
            return float(df["Close"].iloc[-1]), df
    except:
        pass
    return None, None

# ================= INDICATORS =================
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

# ================= SCORE =================
def score_stock(df):
    last = df.iloc[-1]
    score = 0

    if last["EMA50"] > last["EMA200"]:
        score += 30

    if 30 < last["RSI"] < 70:
        score += 20
    elif last["RSI"] < 30:
        score += 15

    if last["MACD"] > last["Signal"]:
        score += 25

    if df["Close"].iloc[-1] > df["Close"].iloc[-5]:
        score += 10

    return score

# ================= AI =================
def train_ai(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","MACD","EMA50","EMA200"]].dropna()
    y = df["Target"].loc[X.index]

    model = LogisticRegression()
    model.fit(X,y)
    return model

def predict_ai(model, last):
    X = pd.DataFrame([[last["RSI"], last["MACD"], last["EMA50"], last["EMA200"]]],
                     columns=["RSI","MACD","EMA50","EMA200"])
    return model.predict_proba(X)[0][1]

# ================= ANALYSIS =================
def analyze(symbol):
    price, df = get_data(symbol)
    if df is None:
        return None

    df = calculate(df)
    model = train_ai(df)
    prob = predict_ai(model, df.iloc[-1])
    score = score_stock(df)

    return f"""📊 {symbol}
💰 {round(price,2)}
🎯 {score}/100
🤖 {prob:.0%}
"""

# ================= BOT =================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.upper().strip()

    if not symbol.isalpha():
        await update.message.reply_text("❌ ابعت رمز سهم بالإنجليزي")
        return

    await update.message.reply_text("⏳ تحليل...")

    result = analyze(symbol)

    if result is None:
        await update.message.reply_text("❌ السهم مش موجود")
        return

    await update.message.reply_text(result)

# ================= DAILY =================
async def daily_report(context: ContextTypes.DEFAULT_TYPE):
    for symbol in TOP_STOCKS:
        msg = analyze(symbol)
        if msg:
            await context.bot.send_message(chat_id=CHAT_ID, text=msg)

# ================= RUN =================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

# ⏰ توقيت القاهرة
cairo = pytz.timezone("Africa/Cairo")

# لو JobQueue موجود
if app.job_queue:
    app.job_queue.run_daily(
        daily_report,
        time=time(hour=9, minute=0, tzinfo=cairo)
    )

print("🚀 BOT RUNNING")

app.run_polling(drop_pending_updates=True)
