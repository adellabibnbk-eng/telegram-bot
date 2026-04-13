import os
import pandas as pd
import yfinance as yf
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler

# ===== CONFIG =====
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TOP_STOCKS = [
    "CIB","TALAAT","FWRY","EFG","SWDY",
    "ETEL","HRHO","ABUK","ORAS","EAST"
]

# ===== SYMBOL =====
def fix_symbol(symbol):
    symbol = symbol.upper().strip()
    if "." not in symbol:
        symbol += ".CA"
    return symbol

# ===== DATA =====
def get_data(symbol):
    try:
        symbol = fix_symbol(symbol)
        df = yf.Ticker(symbol).history(period="6mo")
        if not df.empty:
            return float(df["Close"].iloc[-1]), df
    except:
        pass
    return None, None

# ===== SUPPORT / RESIST =====
def sr(df):
    s = df["Low"].rolling(20).min().tail(3).values
    r = df["High"].rolling(20).max().tail(3).values
    s = [round(x,2) for x in s if not np.isnan(x)][::-1]
    r = [round(x,2) for x in r if not np.isnan(x)][::-1]
    return s, r

# ===== FEATURES =====
def prep(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    return df.dropna()

# ===== AI =====
def train(df):
    from sklearn.linear_model import LogisticRegression

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","Momentum","MA20","MA50"]].dropna()
    y = df["Target"].loc[X.index]

    if len(X) < 50:
        return None

    model = LogisticRegression()
    model.fit(X, y)

    return model

# ===== ANALYZE =====
def analyze(symbol):
    price, df = get_data(symbol)

    if df is None:
        return None

    df = prep(df)
    if df.empty:
        return None

    model = train(df)
    if model is None:
        return None

    last = df.iloc[-1]

    prob = model.predict_proba([[last["RSI"], last["Momentum"], last["MA20"], last["MA50"]]])[0][1]
    score = int(prob * 100)

    supports, resistances = sr(df)

    trend = "صاعد 🔼" if last["MA20"] > last["MA50"] else "هابط 🔽"

    entry = supports[0] if supports else price
    stop = round(entry * 0.97, 2)
    target = resistances[0] if resistances else price

    rr = round((target - entry) / (entry - stop), 2) if (entry - stop) else 0

    if prob > 0.6 and rr > 1.5:
        decision = "🟢 فرصة قوية"
    elif prob > 0.5:
        decision = "🟡 فرصة متوسطة"
    else:
        decision = "🔴 ضعيف"

    return {
        "symbol": symbol,
        "price": price,
        "score": score,
        "prob": prob,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
        "decision": decision,
        "trend": trend
    }

# ===== DAILY REPORT =====
def daily(bot):
    results = []

    for s in TOP_STOCKS:
        r = analyze(s)
        if r:
            results.append(r)

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    msg = "📊 TOP فرص اليوم:\n\n"

    for i, r in enumerate(results,1):
        msg += f"""{i}) {r['symbol']}
Score: {r['score']}/100
Entry: {r['entry']}
Target: {r['target']}
RR: {r['rr']}
{r['decision']}

"""

    bot.send_message(chat_id=CHAT_ID, text=msg)

# ===== HANDLER =====
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip()

    await update.message.reply_text("⏳ جاري التحليل...")

    r = analyze(symbol)

    if not r:
        await update.message.reply_text("❌ السهم غير متاح")
        return

    msg = f"""📊 {r['symbol']}

💰 السعر: {round(r['price'],2)}

📈 الاتجاه: {r['trend']}

🤖 AI Score: {r['score']}/100
📈 احتمال الصعود: {r['prob']:.0%}

━━━━━━━━━━━━━━━
💼 الصفقة:

🎯 دخول: {r['entry']}
🛑 وقف: {r['stop']}
🎯 هدف: {r['target']}

📊 RR: {r['rr']}

━━━━━━━━━━━━━━━
🔥 التقييم:

{r['decision']}

⚠️ ليس توصية استثمارية
"""

    await update.message.reply_text(msg)

# ===== MAIN =====
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT, handle))

    scheduler = BackgroundScheduler(timezone="Africa/Cairo")
    scheduler.add_job(lambda: daily(app.bot), 'cron', hour=9)
    scheduler.start()

    print("🚀 PRO BOT RUNNING")

    app.run_polling()

if __name__ == "__main__":
    main()
