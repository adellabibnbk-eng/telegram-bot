import os
import pandas as pd
import yfinance as yf
import numpy as np
import asyncio

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ===== SYMBOL =====
def fix_symbol(symbol):
    symbol = symbol.upper().strip()
    if "." not in symbol:
        symbol += ".CA"
    return symbol

# ===== DATA (15 min) =====
def get_data(symbol):
    try:
        symbol = fix_symbol(symbol)
        df = yf.Ticker(symbol).history(period="5d", interval="15m")

        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1]), df
    except:
        pass
    return None, None

# ===== SUPPORT / RESISTANCE =====
def sr(df):
    supports = sorted(df["Low"].tail(50))[:3]
    resistances = sorted(df["High"].tail(50))[-3:]

    supports = [round(x,2) for x in supports]
    resistances = [round(x,2) for x in resistances[::-1]]

    return supports, resistances

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

    X = pd.DataFrame([[last["RSI"], last["Momentum"], last["MA20"], last["MA50"]]],
                     columns=["RSI","Momentum","MA20","MA50"])

    prob = model.predict_proba(X)[0][1]
    score = int(prob * 100)

    supports, resistances = sr(df)

    trend = "صاعد 🔼" if last["MA20"] > last["MA50"] else "هابط 🔽"

    # ===== ENTRY SYSTEM =====
    entry = supports[0]
    stop = round(entry * 0.97, 2)
    target = resistances[0]

    rr = round((target - entry) / (entry - stop), 2) if (entry - stop) else 0

    # ===== DECISION =====
    if prob > 0.65 and rr > 1.5:
        decision = "🟢 صفقة قوية"
    elif prob > 0.5:
        decision = "🟡 صفقة متوسطة"
    else:
        decision = "🔴 تجنب"

    return f"""📊 تحليل سهم: {symbol}

💰 السعر الحالي: {round(price,2)}

━━━━━━━━━━━━━━━
📈 الاتجاه:
{trend}
👉 (بناءً على المتوسطات المتحركة)

📊 RSI:
{round(last['RSI'],1)}
👉 أقل من 30 = شراء محتمل
👉 أعلى من 70 = بيع محتمل

━━━━━━━━━━━━━━━
🟢 مستويات الدعم:
{supports}

🔴 مستويات المقاومة:
{resistances}

━━━━━━━━━━━━━━━
🤖 تحليل الذكاء الاصطناعي:

📊 التقييم: {score}/100
📈 احتمال الصعود: {prob:.0%}

━━━━━━━━━━━━━━━
💼 خطة التداول:

🎯 دخول: {entry}
🛑 وقف خسارة: {stop}
🎯 هدف: {target}

📊 نسبة المخاطرة/الربح: {rr}

━━━━━━━━━━━━━━━
🔥 التقييم النهائي:
{decision}

━━━━━━━━━━━━━━━
⚠️ إخلاء المسؤولية:
هذا التحليل مبني على أدوات التحليل الفني ونماذج الذكاء الاصطناعي،
ولا يُعد توصية مباشرة بالشراء أو البيع.
القرار الاستثماري مسؤوليتك بالكامل.
"""

# ===== DAILY =====
def daily(bot):
    msg = "📊 تقرير يومي جاهز 🚀"
    asyncio.run(bot.send_message(chat_id=CHAT_ID, text=msg))

# ===== HANDLER =====
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip()

    await update.message.reply_text("⏳ جاري التحليل...")

    result = analyze(symbol)

    if not result:
        await update.message.reply_text("❌ السهم غير متاح أو البيانات ضعيفة")
        return

    await update.message.reply_text(result)

# ===== MAIN =====
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT, handle))

    scheduler = BackgroundScheduler(timezone="Africa/Cairo")
    scheduler.add_job(lambda: daily(app.bot), 'cron', hour=9)
    scheduler.start()

    print("🚀 BOT PRO MAX RUNNING")

    app.run_polling()

if __name__ == "__main__":
    main()
