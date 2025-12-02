import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Loan Approval Prediction')
st.markdown('Isi data di bawah ini untuk memprediksi apakah pinjaman akan **Disetujui** atau **Ditolak**.')

st.divider()

st.header("Data Pemohon")

# Bagi tampilan jadi 2 kolom biar rapi
col1, col2 = st.columns(2)

with col1:
    # Input Dependents
    no_of_dependents = st.number_input('Jumlah Tanggungan (Dependents)', min_value=0, max_value=20, value=0)
    
    # Input Education (Dropdown)
    edu_options = ['Graduate', 'Not Graduate']
    education_input = st.selectbox('Pendidikan Terakhir', edu_options)
    # Konversi input user jadi angka (1 atau 0) sesuai logic model.py
    education = 1 if education_input == 'Graduate' else 0

    # Input Self Employed (Dropdown)
    emp_options = ['Yes', 'No']
    self_emp_input = st.selectbox('Wiraswasta (Self Employed)?', emp_options)
    # Konversi input user jadi angka (1 atau 0)
    self_employed = 1 if self_emp_input == 'Yes' else 0
    
    # Input Income
    income_annum = st.number_input('Pendapatan Tahunan (Annual Income)', min_value=0, value=1000000)
    
    # Input Loan Amount
    loan_amount = st.number_input('Jumlah Pinjaman (Loan Amount)', min_value=0, value=1000000)
    
    # Input Loan Term
    loan_term = st.number_input('Jangka Waktu Pinjaman (Tahun)', min_value=1, max_value=30, value=1)

with col2:
    # Input CIBIL Score (Skor Kredit)
    cibil_score = st.slider('CIBIL Score (Credit Score)', min_value=300, max_value=900, value=600)
    st.caption("*Skor di atas 700 biasanya lebih baik.")
    
    # Input Aset-aset
    residential_assets = st.number_input('Nilai Aset Rumah', min_value=0, value=0)
    commercial_assets = st.number_input('Nilai Aset Komersial', min_value=0, value=0)
    luxury_assets = st.number_input('Nilai Aset Mewah', min_value=0, value=0)
    bank_asset = st.number_input('Nilai Aset di Bank', min_value=0, value=0)

st.divider()

# Tombol Prediksi
if st.button("Prediksi Status Pinjaman"):
    # Susun data input menjadi array 
    # (Urutan HARUS SAMA PERSIS dengan kolom X di model.py)
    input_data = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_asset
    ]])
    
    # Panggil fungsi prediksi
    hasil = predict(input_data)
    
    # Tampilkan Hasil
    if hasil[0] == 'Approved':
        st.success(f"Selamat! Status Pinjaman: **{hasil[0]}**")
        st.balloons()
    else:
        st.error(f"Maaf, Status Pinjaman: **{hasil[0]}**")