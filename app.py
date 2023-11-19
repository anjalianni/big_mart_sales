

import streamlit as st
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import inspect
import pickle
from xgboost import XGBRegressor



pickle_in=open("classifier1.pkl","rb")
model=pickle.load(pickle_in)


def welcome():
        return "Welcome All"
def predict_sale(Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Location_Type,Outlet_Type):
        lst=np.array([Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Location_Type,Outlet_Type],dtype=object)  
        lst_reshape=np.array(lst).reshape((1,-1))

        prediction=model.predict(lst_reshape)
        print(prediction)
        return prediction
def main():
       st.title("BIG MART SALES ")
       html_temp = """
       <div style="background-color:teal;padding:10px">
       <h2 style="color:white;text-align:center;">BIG MART SALES </h2>
       </div>
       """
       st.markdown(html_temp,unsafe_allow_html=True)
       Item_Identifier = st.text_input("Item_Identifier")
       Item_Weight	 = st.text_input("Item_Weight")
       Item_Fat_Content = st.text_input("Item_Fat_Content")
       Item_Visibility = st.text_input("Item_Visibility")
       Item_Type = st.text_input("Item_Type")
       Item_MRP	 = st.text_input("Item_MRP")
       Outlet_Identifier = st.text_input("Outlet_Identifier")
       Outlet_Establishment_Year = st.text_input("Outlet_Establishment_Year")
       Outlet_Location_Type = st.text_input("Outlet_Location_Type")
       Outlet_Type	 = st.text_input("Outlet_Type")
       result=""
       if st.button("Predict"):
           
           result=predict_sale(Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Location_Type,Outlet_Type)
           #result = google.searchGoogle(param).encode("utf-8")
       st.success('The output is {}'.format(result))
if __name__=='__main__':
       main()


    

