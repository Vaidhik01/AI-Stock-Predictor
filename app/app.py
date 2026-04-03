import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import yfinance as yf
import datetime
import plotly.express as px

from utils.data_loader import load_data
from utils.features import add_features
from utils.sentiment import get_sentiment


def get_metal_prices_inr():
    try:
        gold_df = yf.download("GC=F", period="5d")
        silver_df = yf.download("SI=F", period="5d")
        usd_df = yf.download("USDINR=X", period="5d")

        gold_usd = float(gold_df['Close'].dropna().iloc[-1])
        silver_usd = float(silver_df['Close'].dropna().iloc[-1])
        usd_inr = float(usd_df['Close'].dropna().iloc[-1])

        gold_inr = (gold_usd * usd_inr) / 31.1035
        silver_inr = (silver_usd * usd_inr) / 31.1035

        return round(gold_inr, 2), round(silver_inr, 2)

    except:
        return 6200, 75  # fallback
def get_stock_link(stock):
    base_url = "https://finance.yahoo.com/quote/"
    return f"{base_url}{stock}"

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Stock Predictor", layout="centered")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<h1 style='text-align: center; color: #00ADB5;'>
🚀 INVESTMENT GUIDE
</h1>
""", unsafe_allow_html=True)


# -------------------------------
# 🪙 GOLD & SILVER RATES (₹)
# -------------------------------
gold_price, silver_price = get_metal_prices_inr()

col1, col2 = st.columns(2)

col1.metric("🟡 Gold (₹/gram)", gold_price)
col2.metric("⚪ Silver (₹/gram)", silver_price)

st.divider()

# -------------------------------
# STOCK LIST
# -------------------------------
stock_list = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS",
    "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "TECHM.NS"
]
stock_categories = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS"],
    "Energy": ["RELIANCE.NS"]
}
category = st.selectbox(
    "Select Category",
    ["All", "IT", "Banking", "Energy"]
)
# -------------------------------
# INPUT
# -------------------------------
if category == "All":
    filtered_stocks = stock_list
else:
    filtered_stocks = stock_categories[category]

stock = st.selectbox("Select Stock", filtered_stocks)
st.info(f"Showing stocks in: {category}")

# -------------------------------
# MAIN LOGIC
# -------------------------------
with st.spinner("Analyzing stock data..."):
    st.markdown(f"🔗 [View {stock} Details]({get_stock_link(stock)})")

    try:
        data = load_data(stock)
        data = add_features(data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if data is None or data.empty:
        st.error("No data available")
        st.stop()

    # Fix MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # -------------------------------
    # GRAPH (PLOTLY)
    # -------------------------------
    st.subheader("📈 Stock Price Trend")

    data = data.reset_index()
    data.set_index('Date', inplace=True)

    fig = px.line(data, x=data.index, y='Close', title="Stock Price Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------------
    # ML MODEL (FIXED - NO LEAKAGE)
    # -------------------------------
    X = data[['MA10', 'MA50']]
    y = data['Target']

    

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    latest_data = X.iloc[-1:]
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model.fit(X_train, y_train)

    latest_data = X.iloc[-1:]
    prediction = model.predict(latest_data)[0]
    confidence = model.predict_proba(latest_data)[0][prediction]
    accuracy = model.score(X_test, y_test)

    # Sentiment (can be upgraded to stock-specific)
    sentiment_score = get_sentiment()

# -------------------------------
# METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Price", round(data['Close'].iloc[-1], 2))
col2.metric("📊 Confidence", f"{round(confidence*100, 2)}%")
col3.metric("🧠 Sentiment", round(sentiment_score, 2))

st.divider()

# -------------------------------
# PREDICTION LOGIC (IMPROVED)
# -------------------------------
st.markdown("### 🎯 Prediction")

if prediction == 1 and confidence > 0.6:
    result = "BUY"
elif prediction == 0 and confidence > 0.6:
    result = "SELL"
else:
    result = "HOLD"

if result == "BUY":
    st.success("📈 BUY")
elif result == "SELL":
    st.error("📉 SELL")
else:
    st.warning("🤝 HOLD")

# -------------------------------
# CONFIDENCE LEVEL
# -------------------------------
if confidence > 0.75:
    st.success("High confidence")
elif confidence > 0.5:
    st.warning("Moderate confidence")
else:
    st.error("Low confidence")

# -------------------------------
# ANALYSIS
# -------------------------------
st.markdown("## 🧠 Analysis")

if prediction == 1:
    st.write("Short-term average (MA10) > Long-term (MA50) → Bullish")
else:
    st.write("Short-term average < Long-term → Bearish")

# -------------------------------
# RISK (IMPROVED)
# -------------------------------
st.markdown("## ⚠️ Risk Level")

volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]

if volatility > 0.02:
    st.error("High Risk")
elif volatility > 0.01:
    st.warning("Medium Risk")
else:
    st.success("Low Risk")

# -------------------------------
# RECOMMENDATION
# -------------------------------
st.markdown("## 📌 Recommendation")

if result == "BUY":
    st.write("Consider short-term buying")
elif result == "SELL":
    st.write("Consider exiting position")
else:
    st.write("Wait and observe")

st.write(f"Model Accuracy: {round(accuracy*100,2)}%")

st.divider()

# -------------------------------
# TOP STOCKS (OPTIMIZED)
# -------------------------------
st.markdown("## 🏆 Top Stock Recommendations")

if st.button("🔍 Show Top Stocks"):

    results = []
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)

    for s in stock_list:
        try:
            d = load_data(s)
            d = add_features(d)

            if d is None or d.empty:
                continue

            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

            X_temp = d[['MA10', 'MA50']]
            y_temp = d['Target']

            base_model.fit(X_temp, y_temp)

            pred = base_model.predict(X_temp.tail(1))[0]
            conf = base_model.predict_proba(X_temp.tail(1))[0][pred]

            results.append((s, pred, conf))

        except:
            continue

    results = sorted(results, key=lambda x: x[2], reverse=True)

    for stock_name, pred, conf in results[:3]:
        label = "📈 BUY" if pred == 1 else "📉 SELL"
        link = get_stock_link(stock_name)
        st.markdown(f"{stock_name} → {label} ({round(conf*100,2)}%) 🔗 [Open]({link})")

st.divider()

# -------------------------------
# COMPARE STOCKS
# -------------------------------
st.markdown("## 📊 Compare Multiple Stocks")

selected = st.multiselect(
    "Select stocks",
    stock_list,
    default=["TCS.NS", "INFY.NS"]
)

if selected:
    multi_data = yf.download(selected, period="6mo")['Close']

    if isinstance(multi_data.columns, pd.MultiIndex):
        multi_data.columns = multi_data.columns.get_level_values(0)

    st.line_chart(multi_data)

st.divider()

# -------------------------------
# 💰 INVESTMENT ADVISOR
# -------------------------------
st.markdown("## 💰 Investment Advisor")

investment = st.number_input("Enter Investment Amount ₹", min_value=1000, step=1000)
risk_level = st.selectbox(
    "Select Risk Level",
    ["Low", "Medium", "High"]
)
if st.button("💡 Get Investment Plan"):

    if investment <= 0:
        st.error("Enter a valid amount")
    else:
        results = []
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        for s in stock_list:
            try:
                d = load_data(s)
                d = add_features(d)

                if d is None or d.empty:
                    continue

                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)

                X_temp = d[['MA10', 'MA50']]
                y_temp = d['Target']

                split = int(len(X_temp) * 0.8)
                X_tr, X_te = X_temp[:split], X_temp[split:]
                y_tr, y_te = y_temp[:split], y_temp[split:]

                model.fit(X_tr, y_tr)

                pred = model.predict(X_temp.tail(1))[0]
                conf = model.predict_proba(X_temp.tail(1))[0][pred]

                # Risk-based filtering
                if risk_level == "Low":
                    if pred == 1 and conf > 0.75:
                        results.append((s, conf))

                elif risk_level == "Medium":
                    if pred == 1 and conf > 0.6:
                        results.append((s, conf))

                elif risk_level == "High":
                    if pred == 1:
                        results.append((s, conf))

            except:
                continue

        if not results:
            st.warning("No strong BUY signals found")
        else:
            results = sorted(results, key=lambda x: x[1], reverse=True)[:3]

            total_conf = sum(conf for _, conf in results)

            st.markdown("### 📊 Suggested Investment Allocation")

            for stock_name, conf in results:
                allocation = (conf / total_conf) * investment
                link = get_stock_link(stock_name)
                st.markdown(f"{stock_name} → ₹{round(allocation, 2)} 🔗 [View]({link})")
            if risk_level == "Low":
                st.success("🟢 Low Risk Portfolio (Stable stocks)")
            elif risk_level == "Medium":
                st.warning("🟡 Medium Risk Portfolio (Balanced)")
            else:
                st.error("🔴 High Risk Portfolio (Aggressive)")
            st.success("Diversified AI-based portfolio generated 🚀")


st.divider()

st.divider()

st.divider()

# -------------------------------
# 📊 SIP CALCULATOR
# -------------------------------
st.markdown("## 📊 SIP Investment Calculator")

sip_amount = st.number_input("Monthly Investment (₹)", min_value=500, step=500)
duration = st.slider("Investment Duration (Months)", 6, 60, 12)

selected_stocks = st.multiselect(
    "Select Stocks for SIP",
    stock_list,
    default=["TCS.NS", "INFY.NS"]
)

if st.button("📈 Calculate SIP Growth"):

    if sip_amount <= 0 or not selected_stocks:
        st.error("Enter valid inputs")
    else:
        # 📈 Assumed return (12%)
        annual_return = 0.12
        monthly_rate = annual_return / 12

        future_value = 0

        # SIP Calculation
        for i in range(duration):
            future_value = (future_value + sip_amount) * (1 + monthly_rate)

        # 💰 Results
        total_invested = sip_amount * duration
        profit = future_value - total_invested

        st.subheader("💰 Investment Summary")
        st.info(f"Total Invested: ₹{total_invested}")
        st.success(f"Future Value: ₹{round(future_value, 2)}")
        st.success(f"Profit Earned: ₹{round(profit, 2)}")

        # 📊 Allocation per stock
        st.markdown("### 📊 SIP Allocation per Stock")
        per_stock = sip_amount / len(selected_stocks)

        for s in selected_stocks:
            link = get_stock_link(s)
            st.markdown(f"{s} → ₹{round(per_stock, 2)} 🔗 [View]({link})")

        # 🥧 PIE CHART (Invested vs Profit)
        import plotly.express as px

        labels = ["Invested Amount", "Profit"]
        values = [total_invested, profit]

        fig = px.pie(
            names=labels,
            values=values,
            title="💰 Investment vs Profit"
        )

        fig.update_traces(textinfo='percent+label')

        st.plotly_chart(fig, use_container_width=True)
# -------------------------------
# BONUS: CREDIT SCORE / FINANCE HEALTH
# -------------------------------
st.markdown("## 💳 Financial Health Checker")

income = st.number_input("Monthly Income ₹", min_value=0)
expenses = st.number_input("Monthly Expenses ₹", min_value=0)

if st.button("Check Financial Health"):
    if income == 0:
        st.error("Enter valid income")
    else:
        savings = income - expenses

        if savings > income * 0.3:
            st.success("Excellent Financial Health 💚")
        elif savings > income * 0.1:
            st.warning("Moderate Financial Health ⚠️")
        else:
            st.error("Poor Financial Health 🚨")

# -------------------------------
# 🏦 FD INTEREST RATES
# -------------------------------
st.markdown("## 🏦 Fixed Deposit (FD) Interest Rates")

st.success("FD is low risk compared to stocks")

fd_rates = {
    "SBI": "6.50% - 7.50%",
    "HDFC Bank": "6.60% - 7.75%",
    "ICICI Bank": "6.70% - 7.80%",
    "Axis Bank": "6.75% - 7.85%"
}

col1, col2 = st.columns(2)

banks = list(fd_rates.keys())

col1.metric(banks[0], fd_rates[banks[0]])
col2.metric(banks[1], fd_rates[banks[1]])

col3, col4 = st.columns(2)

col3.metric(banks[2], fd_rates[banks[2]])
col4.metric(banks[3], fd_rates[banks[3]])

st.info("Rates are indicative and may vary based on tenure.")
st.divider()




# -------------------------------
# 🏡 REAL ESTATE ANALYZER
# -------------------------------
st.markdown("## 🏡 Smart Real Estate Investment")

city = st.selectbox(
    "Select City",
    ["Hyderabad", "Bangalore", "Chennai"],
    key="city_select"
)

budget = st.number_input(
    "Enter Budget (₹ per sq.yd)",
    min_value=10000,
    step=5000,
    key="land_budget"
)

if st.button("🔍 Analyze Land Investment"):

    if city == "Hyderabad":
        if budget >= 30000:
            area = "Kokapet"
            price = 30000
            growth = "High"
            risk = "Medium"
        else:
            area = "Shamshabad"
            price = 20000
            growth = "Medium"
            risk = "Low"

    elif city == "Bangalore":
        if budget >= 35000:
            area = "Whitefield"
            price = 35000
            growth = "High"
            risk = "Medium"
        else:
            area = "Devanahalli"
            price = 25000
            growth = "Medium"
            risk = "Low"

    elif city == "Chennai":
        if budget >= 25000:
            area = "OMR"
            price = 25000
            growth = "High"
            risk = "Medium"
        else:
            area = "Tambaram"
            price = 18000
            growth = "Medium"
            risk = "Low"

    # -------------------------------
    # DISPLAY OUTPUT
    # -------------------------------
    st.subheader("📍 Recommended Investment Area")

    col1, col2 = st.columns(2)
    col1.metric("City", city)
    col2.metric("Area", area)

    col3, col4 = st.columns(2)
    col3.metric("Price (₹/sq.yd)", price)
    col4.metric("Growth Potential", growth)

    if risk == "Low":
        st.success("🟢 Low Risk Investment")
    elif risk == "Medium":
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

    # -------------------------------
    # INVESTMENT SUGGESTION
    # -------------------------------
    if growth == "High":
        st.success("🔥 Strong investment opportunity")
    else:
        st.info("Stable long-term investment option")

# -------------------------------
# TIMESTAMP
# -------------------------------
st.write("🕒 Last Updated:", datetime.datetime.now())
