import os
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import asyncio

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ===== AI =====
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    from sklearn.linear_model import LogisticRegression
    USE_XGB = False

TOKEN = os.getenv("TOKEN")

# ===== DATA =====
def get_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period="6mo")
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

    return df.dropna()

# ===== TRAIN =====
def train(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    X = df[["RSI","Momentum","MA20","MA50","Vol_Ratio"]].dropna()
    y = df["Target"].loc[X.index]

    if len(X) < 50:
        return None, None, None

    if USE_XGB:
        model = XGBClassifier(n_estimators=100)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X, y)
    return model, X, y

# ===== BACKTEST =====
def backtest(model, X, y):
    preds = model.predict(X)
    return round((preds == y).mean() * 100, 2)

# ===== MODEL =====
MODEL_FILE = "model.pkl"
LAST_TRAIN = "last_train.txt"

def load_model(df):
    today = str(datetime.date.today())

    if os.path.exists(LAST_TRAIN):
        last = open(LAST_TRAIN).read()
    else:
        last = ""

    if last != today:
        model, X, y = train(df)
        if model is None:
            return None, None

        acc = backtest(model, X, y)

        import pickle
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        with open(LAST_TRAIN, "w") as f:
            f.write(today)

        return model, acc
    else:
        try:
            import pickle
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f), None
        except:
            return None, None

# ===== PREDICT =====
def predict(model, row):
    X = pd.DataFrame([[
        row["RSI"], row["Momentum"], row["MA20"], row["MA50"], row["Vol_Ratio"]
    ]], columns=["RSI","Momentum","MA20","MA50","Vol_Ratio"])
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

    # ===== تحليل فني =====
    trend = "صاعد 🔼" if last["MA20"] > last["MA50"] else "هابط 🔽"

    if last["RSI"] > 70:
        rsi_text = "تشبع شراء (احتمال تصحيح)"
    elif last["RSI"] < 30:
        rsi_text = "تشبع بيع (فرصة ارتداد)"
    else:
        rsi_text = "منطقة طبيعية"

    volume_text = "سيولة قوية" if last["Vol_Ratio"] > 1 else "سيولة ضعيفة"

    # ===== AI =====
    if prob > 0.65:
        ai_text = "توقع صعود قوي"
    elif prob > 0.5:
        ai_text = "توقع محايد"
    else:
        ai_text = "توقع هبوط"

    # ===== الخلاصة =====
    if prob > 0.65 and trend == "صاعد 🔼":
        final = "السهم في وضع إيجابي (اتجاه + AI داعم)"
    elif prob < 0.5 and trend == "هابط 🔽":
        final = "السهم ضعيف (اتجاه + AI سلبي)"
    else:
        final = "وضع مختلط يحتاج تأكيد"

    return f"""📊 {symbol}

💰 السعر الحالي: {round(price,2)}

━━━━━━━━━━━━━━━
📊 التحليل الفني:

📈 الاتجاه: {trend}
➡️ (حسب متوسطات الحركة)

📉 RSI: {round(last["RSI"],1)}
➡️ {rsi_text}

📊 السيولة:
➡️ {volume_text}

━━━━━━━━━━━━━━━
🤖 تحليل الذكاء الاصطناعي:

🎯 AI Score: {score}/100
📈 احتمال الصعود: {prob:.0%}

🧠 التقييم:
➡️ {ai_text}

📊 دقة النموذج تاريخيًا:
➡️ {acc if acc else "مستقر"} %

━━━━━━━━━━━━━━━
✍️ التقييم النهائي:

{final}

━━━━━━━━━━━━━━━
⚠️ إخلاء مسؤولية:
هذا التحليل مبني على بيانات تاريخية ونماذج إحصائية
ولا يُعد توصية استثمارية مباشرة.
"""

# ===== HANDLER =====
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()

    await update.message.reply_text("⏳ جاري التحليل...")

    result = analyze(symbol)

    await update.message.reply_text(result)

# ===== MAIN =====
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🚀 BOT RUNNING PRO VERSION")

    await app.bot.delete_webhook(drop_pending_updates=True)

    await app.initialize()
    await app.start()

    await app.bot.get_updates(offset=-1)

    await app.updater.start_polling()

    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
