import os
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import json
import pickle

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# 🔥 fallback لو xgboost مش متسطب
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

# ================= FEATURES =================
def prepare_features(df):
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

# ================= TRAIN =================
def train_model(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","Momentum","MA20","MA50","Vol_Ratio","Range","Breakout"]].dropna()
    y = df["Target"].loc[X.index]

    if USE_XGB:
        model = XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X, y)
    return model, X, y

# ================= BACKTEST =================
def backtest(model, X, y):
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    return round(accuracy * 100, 2)

# ================= DAILY TRAIN =================
def load_or_train(df):
    today = str(datetime.date.today())

    if os.path.exists(LAST_TRAIN_FILE):
        last = open(LAST_TRAIN_FILE).read()
    else:
        last = ""

    if last != today:
        model, X, y = train_model(df)
        acc = backtest(model, X, y)

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        with open(LAST_TRAIN_FILE, "w") as f:
            f.write(today)

        return model, acc
    else:
        try:
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
            return model, None
        except:
            model, X, y = train_model(df)
            return model, None

# ================= PREDICT =================
def predict(model, row):
    X = pd.DataFrame([[
        row["RSI"], row["Momentum"], row["MA20"], row["MA50"],
        row["Vol_Ratio"], row["Range"], row["Breakout"]
    ]], columns=[
        "RSI","Momentum","MA20","MA50",
        "Vol_Ratio","Range","Breakout"
    ])

    return model.predict_proba(X)[0][1]

# ================= SUMMARY =================
def summary_line(prob):
    if prob > 0.65:
        return "السهم في اتجاه إيجابي مع فرص صعود واضحة مدعومة بالزخم."
    elif prob > 0.5:
        return "السهم في منطقة حيادية مع احتمالية صعود محدودة."
    else:
        return "السهم يظهر ضعف في الاتجاه وقد يواجه ضغط بيعي."

# ================= SAVE SIGNAL =================
def save_signal(symbol, price, prob):
    signal = {
        "symbol": symbol,
        "price": price,
        "prob": prob,
        "date": str(datetime.date.today())
    }

    try:
        with open(SIGNALS_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(signal)

    with open(SIGNALS_FILE, "w") as f:
        json.dump(data, f)

# ================= PERFORMANCE =================
def evaluate_signals():
    try:
        with open(SIGNALS_FILE, "r") as f:
            data = json.load(f)
    except:
        return "لا توجد بيانات"

    success = 0
    total = 0

    for s in data:
        price, df = get_data(s["symbol"])
        if df is None:
            continue

        current = df["Close"].iloc[-1]

        if current > s["price"]:
            success += 1

        total += 1

    if total == 0:
        return "لا توجد نتائج كافية"

    rate = (success / total) * 100
    return f"📊 نسبة النجاح: {round(rate,2)}%"

# ================= ANALYSIS =================
def analyze(symbol):
    price, df = get_data(symbol)

    if df is None:
        return None

    df = prepare_features(df)

    if df is None or df.empty:
        return None

    last = df.iloc[-1]

    model, acc = load_or_train(df)
    prob = predict(model, last)

    score = int(prob * 100)
    summary = summary_line(prob)

    save_signal(symbol, round(price,2), prob)

    msg = f"""📊 {symbol}

💰 السعر: {round(price,2)}

🤖 AI Score: {score}/100
📈 احتمال الصعود: {prob:.0%}

━━━━━━━━━━━━━━━
✍️ الخلاصة:
{summary}

━━━━━━━━━━━━━━━
📊 دقة النموذج:
{acc if acc else "محدث مسبقًا"} %

━━━━━━━━━━━━━━━
⚠️ إخلاء مسؤولية:
هذا التحليل مبني على نماذج إحصائية وتحليل فني
ولا يُعد نصيحة استثمارية مباشرة.
"""

    return msg, prob

# ================= RANKING =================
def get_ranking():
    results = []

    for symbol in TOP_STOCKS:
        try:
            msg, prob = analyze(symbol)
            if msg:
                results.append((symbol, prob))
        except:
            continue

    results.sort(key=lambda x: x[1], reverse=True)

    text = "🔥 أفضل فرص اليوم:\n\n"

    for i, (symbol, prob) in enumerate(results[:5], 1):
        text += f"{i}. {symbol} → {int(prob*100)}/100\n"

    return text

# ================= BOT =================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if text == "ranking":
        await update.message.reply_text("⏳ تحليل السوق...")
        await update.message.reply_text(get_ranking())
        return

    if text == "performance":
        await update.message.reply_text(evaluate_signals())
        return

    symbol = text.upper()

    await update.message.reply_text("⏳ جاري التحليل...")

    result = analyze(symbol)

    if result is None:
        await update.message.reply_text("❌ السهم غير متاح")
        return

    msg, _ = result
    await update.message.reply_text(msg)

# ================= RUN =================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

print("🚀 BOT RUNNING - SAFE MODE")

app.run_polling(drop_pending_updates=True)
