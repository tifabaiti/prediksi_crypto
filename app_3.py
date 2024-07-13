import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer

# Fungsi untuk mengonversi nilai dengan 'K', 'M', dan '%' menjadi angka
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '').replace('%', '')
        if 'K' in value:
            return float(value.replace('K', '')) * 1000
        elif 'M' in value:
            return float(value.replace('M', '')) * 1000000
    try:
        return float(value)
    except ValueError:
        return np.nan

# Fungsi untuk memuat dan memproses dataset
def load_and_process_data(file_path, selected_features=None):
    df = pd.read_csv(file_path)

    # Memilih fitur yang sesuai jika dipilih
    if selected_features:
        df = df[selected_features]

    # Mengatasi nilai non-numerik
    for column in df.columns:
        df[column] = df[column].apply(convert_to_numeric)

    # Memastikan semua kolom numerik memiliki tipe float
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Imputasi nilai hilang dengan mean
    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)

    # Membuat DataFrame kembali setelah imputasi
    df = pd.DataFrame(df, columns=df.columns)

    # Mengatasi Outliers (opsional)
    # numeric_df = df.select_dtypes(include=[np.number])
    # Q1 = numeric_df.quantile(0.25)
    # Q3 = numeric_df.quantile(0.75)
    # IQR = Q3 - Q1
    # df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df

# Fungsi untuk melatih model SVR
def train_svr_model(X_train, y_train):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVR(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Fungsi untuk melakukan prediksi
def predict_price(model, scaler, input_data, columns):
    input_df = pd.DataFrame(input_data, columns=columns, index=[0])
    input_scaled = scaler.transform(input_df)
    prediksi = model.predict(input_scaled)
    return prediksi[0]

# Membaca dataset untuk Bitcoin, Dogecoin, dan Ethereum
bitcoin_df = load_and_process_data('bitcoin.csv', ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%'])
dogecoin_df = load_and_process_data('dogecoin_10_parameter.csv', ['SMA_10', 'WMA_10', 'Momentum_10', 'Stoch_K%_10', 'Stoch_D%_10', 'RSI_10', 'MACD', 'Williams_R%_10', 'AD_Oscillator', 'CCI_10'])
ethereum_df = load_and_process_data('ethereum_10_parameter.csv', ['SMA_10', 'WMA_10', 'Momentum_10', 'Stoch_K%_10', 'Stoch_D%_10', 'RSI_10', 'MACD', 'Williams_R%_10', 'AD_Oscillator', 'CCI_10'])

# Membuat aplikasi Streamlit
st.title('Prediksi Harga Cryptocurrency')

# Menu navigasi untuk memilih cryptocurrency
menu = st.sidebar.selectbox('Pilih Cryptocurrency', ('Bitcoin', 'Dogecoin', 'Ethereum'))

# Menampilkan form input untuk prediksi
if menu == 'Bitcoin':
    st.header('Prediksi Harga Bitcoin')
    selected_df = bitcoin_df
elif menu == 'Dogecoin':
    st.header('Prediksi Harga Dogecoin')
    selected_df = dogecoin_df
elif menu == 'Ethereum':
    st.header('Prediksi Harga Ethereum')
    selected_df = ethereum_df

st.sidebar.header('Input Data')
input_data = {}
for column in selected_df.columns:
    input_data[column] = st.sidebar.number_input(column, value=float(selected_df[column].mean()))

# Melakukan pembagian data menjadi data pelatihan dan pengujian
X = selected_df.drop(columns=['Terakhir'])
y = selected_df['Terakhir']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model dan melakukan prediksi jika tombol ditekan
if st.sidebar.button('Mulai Prediksi', key='predict_button'):
    model, scaler = train_svr_model(X_train, y_train)
    prediksi_harga = predict_price(model, scaler, input_data, X.columns)

    # Menampilkan hasil prediksi
    st.write('### Data Input Pengguna')
    input_df = pd.DataFrame(input_data, index=[0])
    st.write(input_df)

    st.write('### Prediksi Harga Berikutnya')
    st.write(prediksi_harga)

# Mengatur tata letak warna tombol
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #0072b1;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
