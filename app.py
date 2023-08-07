# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:47:11 2023

@author: michalk
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d
from keras.models import load_model

st.set_page_config(layout='wide')

def warnings(value, minimum, maximum, label):
    if value < minimum or value > maximum:
        st.warning('%s is out of range. Please change input data' %label, icon="⚠️")

class features:
    def __init__ (self, Ta, Tf, Lwl, B, Disp, V):
        self.Ta = Ta
        self.Tf = Tf
        self.Lwl = Lwl
        self.B = B
        self.Disp = Disp
        self.V = V
        self.g = 9.81
    
    def calc(self):
        trim = (self.Ta-self.Tf)+3
        T = (self.Ta+self.Tf)/2
        CB = self.Disp/(self.Lwl*self.B*T)
        BT = self.B/T
        Fr = self.V*0.51444/(self.Lwl*self.g)**0.5
        return (trim, T, CB, BT, Fr)
    
    def form_factor(self, CB, T):
        k = 0.017 + 20*CB/((self.Lwl/self.B)**2*np.sqrt(self.B/T))
        return k


class ITTC_resistance:
    def __init__(self, V, Lwl, k, CR, AT, S):
        self.V = V
        self.Lwl = Lwl
        self.k = k
        self.CR = CR
        self.AT = AT
        self.S = S
        self.g = 9.81
        self.vi = 1.188*10**-6
        self.ros = 1025.9
        self.ks = 150*10**-6

    def calc_RTS(self):
        re = self.V*0.51444*self.Lwl/self.vi
        Fr = self.V*0.51444/(np.sqrt(self.Lwl*self.g))

        dCF = (105*(self.ks/self.Lwl)**(1/3)-0.64)*10**-3
        CFS = (0.075/(np.log10(re - 2)**2)) + dCF
        CAA = 0.001*self.AT/self.S
        CTS = CFS*(1+self.k) + dCF + self.CR + CAA   
        RTS = CTS*0.5*self.ros*(self.V*0.51444)**2*self.S*10**-3
        PE = RTS*self.V*0.51444
        columns = ['V [knots]', 'Fr [-]', 'RT [kN]', 'PE [kW]', 'CT [10-3]', 'Re [10+8]', 'CF [10-3]', 'CR [10-3]']
        df = pd.DataFrame(np.array([self.V, Fr, RTS, PE, CTS*10**3, re*10**-8, CFS*10**3, self.CR*10**3]).T, columns=columns).round(2)
        return RTS, df

class resistance:
    def __init__(self, X, columns):
        Cr = []
        self.X = X
        self.Cr = Cr
        self.columns = columns
        
    def predict_DNN(self, model):
        for i in Fr:
            X = self.X
            X = np.insert(X,6,i)
            df = pd.DataFrame([X], columns=self.columns)
            print(df)
            arr = np.zeros((2,len(df.columns)))
            arr[0] = df.values
            arr[1] = df.values
            self.Cr.append(model.predict(df)[0])
            warnings(i, 0.11, 0.37, 'V')
        
        self.Cr = np.array(self.Cr)[:,0]
        return self.Cr

    def predict_ML(self, model):
      for i in Fr:
          X = self.X
          X = np.insert(X,6,i)
          df = pd.DataFrame([X], columns=self.columns)
          self.Cr.append(model.predict(df)[0])
          warnings(i, 0.11, 0.37, 'V')
      return self.Cr

class math:
    def __init__(self, V_des, V, RTS):
        self.V_des = V_des
        self.RTS = RTS  
        self.V = V

    def interp(self):
        interp = interp1d(self.V, self.RTS)
        RTSi = float(interp(V_des))
        print(RTSi)
        return(RTSi)
              
            
class plot:
    def __init__(self, Fr, RTS, V, title, V_des, RTSi):
        self.RTS = RTS
        self.Fr = Fr
        self.V = V
        self.title = title
    
    def chart(self):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(self.V, self.RTS, c='crimson', marker='o', zorder=1)
        ax.vlines(V_des, 0, RTSi, color='black', linestyle='--', linewidth=1)
        ax.hlines(RTSi, 0, V_des, color='black', linestyle='--', linewidth=1)
        ax.scatter(V_des, RTSi, c='black', zorder=2)
        ax.set_xlabel('V [knots]')
        ax.set_ylabel('$R_{TS}$ [kN]')
        ax.set_xlim(min(V)-1, max(V)+1)
        ax.set_ylim(min(RTS)-20, max(RTS)+20)
        ax.text(V_des-1.5,RTSi+0.01*RTSi,'RTS = %.1f kN' %RTSi)
        ax.set_title('Resistance Prediction - %s' %self.title)
        plt.grid()
        return fig

class save:
    def __init__(self, df, model, fig):
        self.df = df
        self.model = model
        self.model = self.model.replace(" ", "_")
        self.model = self.model.lower()
        self.fig = fig
    
    def save_data(self):
        self.df.to_csv('./output/%s_pred.csv' %self.model, sep=';')
        
    def save_figure(self):
        self.fig.savefig('./output/%s_fig.png' %self.model, dpi=300)
        
        

    
        
if __name__ == "__main__":    
    
    model = joblib.load(open('./models/ml_model.pkl', 'rb'))
    dnn_model = load_model('./models/dnn_model.h5py')
    st.sidebar.header('Ship Hydrostatic Data')
    st.header('Fishing Vessel Resistance Prediction')
    col1, col2 = st.columns(2)
    Lwl = st.sidebar.number_input('Length of Waterline: $LWL [m]$', value=70)
    B = st.sidebar.number_input('Breadth at Watelrine: $B [m]$', value=13)
    Disp = st.sidebar.number_input(r'Displacement Volume: $\nabla [m^3]$', value=3200)
    S = st.sidebar.number_input('Area of wetted surface: $S [m^2]$', value=1500)
    Ta = st.sidebar.number_input('Draught Aft: $T_A [m]$', value=5)
    Tf = st.sidebar.number_input('Draught Fore: $T_F [m]$', value=5)
    LCB = st.sidebar.number_input('Longitudinal Centre of Buoyancy: $LCB [m]$', value=26)
    AT = st.sidebar.number_input('Transverse projected area of ship above waterline: $A_T[m^2]$', value=200)
    CP = st.sidebar.number_input('Longitudinal Prismatic Coefficient: $C_P [-]$', value=0.79)
    CM = st.sidebar.number_input('Midship Section Coefficient: $C_M [-]$', value=0.92)
    
    st.sidebar.text('Ship Speed Range: V [knots]')
    side_col = st.sidebar.columns(3)
    with side_col[0]:
        vmin = float(st.text_input('Vmin', 10))
    with side_col[1]:
        step = float(st.text_input('step', 1))
    with side_col[2]:
        vmax = float(st.text_input('Vmax', 16))
    
    V = np.arange(vmin,vmax+1,step)
    
    feat = features(Ta, Tf, Lwl, B, Disp, V)
    trim, T, CB, BT, Fr = feat.calc()
    
    k_radio = st.sidebar.radio('Form Factor: $k [-]$', ('Input value', 'Calculate'), horizontal=True)
    if k_radio == 'Input value':
        k = st.sidebar.number_input(' ', value=0.37)
    else:
        k = feat.form_factor(CB, T)
    
    warnings(k, 0.2, 0.66, 'k')
    warnings(CB, 0.48, 0.88, 'CB')
    warnings(CM, 0.87, 0.99, 'CM')
    warnings(LCB, 20.0, 50.0, 'LCB')
    warnings(CP, 0.59, 0.86, 'CP')
    warnings(BT, 2.1, 4.8, 'B/T')
    warnings(trim, 1.8, 6.2, 'Trim')


    
    X = np.array([k, CB, LCB, CM, CP, BT, trim]) 
    columns = ['k', 'CB', 'LCB', 'CM', 'CP', 'BT','Fr', 'Trim']
    with col1:
        model_sel = st.radio('Prediction Model Selection', ('Gradient Boosting Regressor', 'Deep Neural Network'), horizontal=True, index=0)     
        res = resistance(X, columns)
        if model_sel == 'Gradient Boosting Regressor':                 
            Cr = np.array(res.predict_ML(model))*10**-3
        else:
            Cr = np.array(res.predict_DNN(dnn_model))*10**-3

    res_pred = ITTC_resistance(V, Lwl,k, Cr, AT, S)
    [RTS, df] = res_pred.calc_RTS()
    
    with col2:
        V_des = float(st.number_input('Design Speed [knots]', value=14.0))
        st.dataframe(df, hide_index=True, use_container_width=True)
            
        mm = math(V_des, V, RTS)
        RTSi = mm.interp()
        plot = plot(Fr, RTS, V, model_sel, V_des, RTSi)
        fig = plot.chart()
        
        sv = save(df, model_sel, fig)
        if st.button('Save Data'):
            sv.save_data()
        if st.button('Save Figure'):
            sv.save_figure()
    
    with col1:
        st.pyplot(fig, use_container_width=True)
            

        

