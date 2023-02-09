import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


model = joblib.load('model.h11')
scaler = joblib.load('scaler.h11')

st.title("Profit Prediction App")


Categories = ['Automobile' ,'Phones&Tablets', 'Electronics' ,'Health&Beauty', 'Home&Office',
 'Supermarket' ,'Computing', 'Gaming' ,'Baby Products', 'Sporting Goods','Fashion']
Categories_encode = {'Automobile':[0,0,0,0,0,0,0,0,0,0],
                        'Baby Products':[1,0,0,0,0,0,0,0,0,0],
                        'Computing':[0,1,0,0,0,0,0,0,0,0],
                        'Electronics':[0,0,1,0,0,0,0,0,0,0],
                        'Fashion':[0,0,0,1,0,0,0,0,0,0],
                        'Gaming':[0,0,0,0,1,0,0,0,0,0],
                        'Health&Beauty':[0,0,0,0,0,1,0,0,0,0],
                        'Home&Office':[0,0,0,0,0,0,1,0,0,0],
                        'Phones&Tablets':[0,0,0,0,0,0,0,1,0,0],
                        'Sporting Goods':[0,0,0,0,0,0,0,0,1,0],
                        'Supermarket':[0,0,0,0,0,0,0,0,0,1]}
category = st.selectbox("choose the category:",Categories)
var1 = Categories_encode[category]
var2 = st.number_input(" Enter the month")
var3 = st.number_input("Enter the year")
options = ['Damanhur', 'Hurghada', 'Qena', 'Cairo', 'Luxor' ,'Marsa Matruh',
 'Shibin El Kom' ,'Port Said', 'Kafr El Sheikh' ,'Ismailia' ,'Giza' ,'Arish',
 'Beni Suef', 'Alexandria' ,'Minya', 'Banha', 'Damietta' ,'Asyut', 'Zagazig',
 'Mansoura' ,'Aswan', 'Kharga', 'Tanta', 'Faiyum']
cities = {'Alexandria':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Arish':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Aswan':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Asyut':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Banha':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Beni Suef':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Cairo':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Damanhur':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Damietta':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Faiyum':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Giza':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            'Hurghada':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            'Ismailia':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            'Kafr El Sheikh':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            'Kharga':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            'Luxor':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            'Mansoura':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            'Marsa Matruh':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            'Minya':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            'Port Said':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            'Qena':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
            'Shibin El Kom':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
            'Tanta':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
            'Zagazig':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
city = st.selectbox("choose the city:",options)
var4 = cities[city]
result1 = [int(x) for x in var4]
#result1 = ', '.join(str(item) for item in result1)

suppliers = ['supplier_3' ,'supplier_1' ,'supplier_4' ,'supplier_2']
suppliers_encode = {'supplier_1':[0,0,0],
                    'supplier_2':[1,0,0],
                    'supplier_3':[0,1,0], 
                    'supplier_4':[0,0,1]}
supplier = st.selectbox("choose the supplier:",suppliers)
var5= suppliers_encode[supplier]
result2 = [int(x) for x in var5]

lst = [var1,var2,var3,result1,result2]
flattened = []
for elem in lst:
    if type(elem) == list:
        flattened.extend(elem)
    else:
        flattened.append(elem)

pred = model.predict(scaler.transform([flattened]))[0]



if st.button('Predict'):
	st.success(f'Predicted Profit is :{pred}')