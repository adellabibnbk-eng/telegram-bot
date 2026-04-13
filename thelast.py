import os
import pandas as pd
import yfinance as yf
import numpy as np

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

TOKEN = os.getenv("TOKEN")

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
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()

# ================= SUPPORT / RESIST =================
def get_levels(df):
    recent = df.tail(20)
    support = round(recent["Low"].min(), 2)
    resistance = round(recent["High"].max(), 2)
    return support, resistance

# ================= TREND =================
def get_trend(df):
    last = df.iloc[-1]

    if last["MA50"] > last["MA200"]:
        return "📈 اتجاه صاعد"
    elif last["MA50"] < last["MA200"]:
        return "📉 اتجاه هابط"
    else:
        return "⚖️ اتجاه عرضي"

# ================= ANALYSIS =================
def analyze(symbol):
    price, df = get_data(symbol)

    if df is None:
        return None

    df = calculate(df)
    last = df.iloc[-1]

    support, resistance = get_levels(df)
    trend = get_trend(df)

    # RSI
    rsi = round(last["RSI"], 2)
    if rsi > 70:
        rsi_text = "تشبع شراء (احتمال تصحيح)"
    elif rsi < 30:
        rsi_text = "تشبع بيع (احتمال ارتداد)"
    else:
        rsi_text = "منطقة طبيعية"

    # Volume
    vol_now = df["Volume"].iloc[-1]
    vol_avg = df["Volume"].rolling(10).mean().iloc[-1]

    if vol_now > vol_avg:
        vol_text = "حجم تداول مرتفع (تأكيد للحركة)"
    else:
        vol_text = "حجم ضعيف (حذر)"

    # Decision logic (بدون افتراضات)
    if price <= support * 1.02 and rsi < 40:
        decision = "📈 فرصة شراء من الدعم"
    elif price >= resistance * 0.98 and rsi > 60:
        decision = "📉 منطقة بيع / جني أرباح"
    else:
        decision = "⚖️ انتظار ومراقبة"

    # Stop / Target
    stop_loss = round(support * 0.97, 2)
    target = resistance

    return f"""📊 تحليل فني لسهم {symbol}

💰 السعر الحالي: {round(price,2)}

━━━━━━━━━━━━━━━
📈 الاتجاه:
{trend}

━━━━━━━━━━━━━━━
📊 الدعم والمقاومة:
🟢 الدعم: {support}
🔴 المقاومة: {resistance}

━━━━━━━━━━━━━━━
📉 مؤشر RSI: {rsi}
📌 {rsi_text}

━━━━━━━━━━━━━━━
📊 حجم التداول:
📌 {vol_text}

━━━━━━━━━━━━━━━
🎯 خطة التداول:

📍 نقطة الدخول:
قرب {support} أو بعد اختراق {resistance}

🛑 وقف الخسارة:
{stop_loss}

🎯 الهدف:
{target}

━━━━━━━━━━━━━━━
🔥 التوصية:
{decision}

━━━━━━━━━━━━━━━
⚠️ إخلاء مسؤولية:
هذا التحليل مبني على أدوات التحليل الفني فقط
ولا يُعد توصية استثمارية مباشرة.
القرار النهائي مسؤولية المستخدم بالكامل،
وقد تتحمل الأسواق مخاطر غير متوقعة.
"""

# ================= BOT =================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.upper().strip()

    if not symbol.isalpha():
        await update.message.reply_text("❌ ابعت رمز سهم بالإنجليزي")
        return

    await update.message.reply_text("⏳ جاري التحليل...")

    result = analyze(symbol)

    if result is None:
        await update.message.reply_text("❌ السهم غير متاح")
        return

    await update.message.reply_text(result)

# ================= RUN =================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

print("🚀 BOT RUNNING")

app.run_polling(drop_pending_updates=True)
