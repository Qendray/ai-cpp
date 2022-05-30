import streamlit as st
from datetime import date
 
import yfinance as yf
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
#from plotly import graph_objs as go

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("AI CPP")

commodities = ("CL=F","^N225","PL=F","HG=F","GC=F","BTI")
selected_commodities = st.selectbox("Select dataset for prediction",commodities)

n_years = st.slider("Years of prediction",1,4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.downloader(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_commodities)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.writer(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.scatter(x=data['Date'], y=data['Open'], name='commodities_open'))
    fig.add_trace(go.scatter(x=data['Date'], y=data['Close'], name='commodities_close'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(colums={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast =m.predict(future)

st.subheader('Forecast data')
st.writer(forecast.tail())

st.writer('forecast data')
fig1 = plot_plotly(m, forecast)
st.plot_chart(fig1)

st.writer('forecast components')
fig2 = m.plot_components(forecast)
st.writer(fig2)

