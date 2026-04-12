import os
import pandas as pd
import yfinance as yf
import numpy as np

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from sklearn.linear_model import LogisticRegression

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

# ================= ANALYSIS TEXT =================
def build_analysis(symbol, price, df, prob):
    last = df.iloc[-1]

    # RSI
    rsi = last["RSI"]
    if rsi > 70:
        rsi_text = "🔴 متشبع شراء (احتمال هبوط)"
    elif rsi < 30:
        rsi_text = "🟢 متشبع بيع (فرصة صعود)"
    else:
        rsi_text = "⚖️ طبيعي"

    # MACD
    if last["MACD"] > last["Signal"]:
        macd_text = "📈 زخم صاعد"
    else:
        macd_text = "📉 زخم هابط"

    # Trend
    if last["EMA50"] > last["EMA200"]:
        trend = "📈 اتجاه صاعد"
        trend_score = 1
    else:
        trend = "📉 اتجاه هابط"
        trend_score = -1

    # AI
    if prob > 0.6:
        ai_text = "🟢 فرصة صعود قوية"
    elif prob > 0.5:
        ai_text = "⚖️ فرصة متوسطة"
    else:
        ai_text = "🔴 احتمال هبوط"

    # FINAL DECISION
    if trend_score == 1 and prob > 0.6:
        decision = "🔥 شراء قوي"
    elif prob > 0.5:
        decision = "📈 شراء بحذر"
    elif prob < 0.4:
        decision = "❌ بيع"
    else:
        decision = "⚖️ حيادي"

    return f"""📊 تحليل سهم {symbol}

💰 السعر: {round(price,2)}

📉 RSI: {round(rsi,2)}
{rsi_text}

📊 MACD: {round(last["MACD"],2)}
{macd_text}

📈 الاتجاه:
{trend}

🤖 الذكاء الاصطناعي:
احتمال الصعود: {prob:.0%}
{ai_text}

━━━━━━━━━━━━━━━
🎯 التقييم النهائي:
{decision}
"""

# ================= BOT =================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.upper().strip()

    if not symbol.isalpha():
        await update.message.reply_text("❌ ابعت رمز سهم بالإنجليزي")
        return

    await update.message.reply_text("⏳ جاري التحليل...")

    price, df = get_data(symbol)

    if df is None:
        await update.message.reply_text("❌ السهم مش موجود")
        return

    df = calculate(df)
    model = train_ai(df)
    prob = predict_ai(model, df.iloc[-1])

    msg = build_analysis(symbol, price, df, prob)

    # 🔥 إضافة التسويق
    msg += "\n\n💰 للاشتراك في التوصيات اليومية ابعت (اشتراك)"

    await update.message.reply_text(msg)

# ================= RUN =================
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

print("🚀 BOT RUNNING")

app.run_polling(drop_pending_updates=True)
