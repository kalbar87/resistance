# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:47:11 2023

@author: michalk
"""
import pandas as pd
import pickle
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(layout='wide')


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
        
    def calc_RTS(self):
        vi = 1.188*10**-6
        ros = 1025.9
        re = self.V*0.51444*self.Lwl/vi
        ks = 150*10**-6
        dCF = (105*(ks/self.Lwl)**(1/3)-0.64)*10**-3
        CFS = (0.075/(np.log10(re - 2)**2)) + dCF
        CAA = 0.001*self.AT/self.S
        CTS = CFS*(1+self.k) + dCF + self.CR + CAA   
        RTS = CTS*0.5*ros*(self.V*0.51444)**2*self.S*10**-3
        columns = ['V [knots]', 'RTS$ [kN]', 'CTS [10-3]', 'Re [10+8]', 'CFS [10-3]']
        df = pd.DataFrame(np.array([self.V, RTS, CTS*10**3, re*10**-8, CFS*10**3]).T, columns=columns)
        return RTS, df

class resistance:
    def __init__(self, X):
        Cr_ML = []
        self.X = X
        self.Cr_ML = Cr_ML
        
        
    def predict_ML(self, model, columns):
        for i in Fr:
            X = self.X
            X = np.insert(X,6,i)
            df = pd.DataFrame([X], columns=columns)
            self.Cr_ML.append(model.predict(df)[0])
        return self.Cr_ML


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
    def __init__(self, Fr, RTS, V):
        self.RTS = RTS
        self.Fr = Fr
        self.V = V
    
    def chart(self, V_des, RTSi):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(self.V, self.RTS, c='crimson')
        ax.vlines(V_des, 0, RTSi, color='black', linestyle='--', linewidth=1)
        ax.hlines(RTSi, 0, V_des, color='black', linestyle='--', linewidth=1)
        ax.scatter(V_des, RTSi, c='black')
        ax.set_xlabel('V [knots]')
        ax.set_ylabel('$R_{TS}$ [kN]')
        ax.set_xlim(min(V)-1, max(V)+1)
        ax.set_ylim(min(RTS)-20, max(RTS)+20)
        ax.text(V_des-1.5,RTSi+0.01*RTSi,'RTS = %.1f kN' %RTSi)
        plt.grid()
        return fig
        

            
        
        
if __name__ == "__main__":    
    
    model = joblib.load(open('ml_model.pkl', 'rb'))
    st.sidebar.header('Ship Hydrostatic Data')
    st.header('Machine Learning and Deep Neural Network Ship Resistance Prediction')
    col1, col2 = st.columns(2)
    Lwl = st.sidebar.number_input('Length of Waterline: $LWL [m]$', value=66.18)
    B = st.sidebar.number_input('Breadth at Watelrine: $B [m]$', value=15)
    Disp = st.sidebar.number_input(r'Displacement Volume: $\nabla [m^3]$', value=3661)
    S = st.sidebar.number_input('Area of wetted surface: $S [m^2]$', value=1449)
    Ta = st.sidebar.number_input('Draught Aft: $T_A [m]$', value=5.4)
    Tf = st.sidebar.number_input('Draught Fore: $T_F [m]$', value=5.4)
    LCB = st.sidebar.number_input('Longitudinal Centre of Buoyancy: $LCB [m]$', value=28.548)
    AT = st.sidebar.number_input('Transverse projected area of ship above waterline: $A_T[m^2]$', value=265)
    CP = st.sidebar.number_input('Longitudinal Prismatic Coefficient: $C_P [-]$', value=0.7519)
    CM = st.sidebar.number_input('Midship Section Coefficient: $C_M [-]$', value=0.959)
    
    st.sidebar.text('Ship Speed Range: V [knots]')
    side_col = st.sidebar.columns(3)
    with side_col[0]:
        vmin = float(st.text_input('Vmin', 8))
    with side_col[1]:
        step = float(st.text_input('step', 1))
    with side_col[2]:
        vmax = float(st.text_input('Vmax', 14))
    
    V = np.arange(vmin,vmax,step)
    
    feat = features(Ta, Tf, Lwl, B, Disp, V)
    trim, T, CB, BT, Fr = feat.calc()
    
    k_radio = st.sidebar.radio('Form Factor: $k [-]$', ('Input value', 'Calculate'), horizontal=True)
    if k_radio == 'Input value':
        k = st.sidebar.number_input(' ', value=0.37)
    else:
        k = feat.form_factor(CB, T)
    
    
    X = np.array([k, CB, LCB, CM, CP, BT, trim]) 
    columns = ['k', 'CB', 'LCB', 'CM', 'CP', 'BT','Fr', 'Trim']
    
    res = resistance(X)
    Cr_ML = np.array(res.predict_ML(model, columns))*10**-3
    
    res_pred = ITTC_resistance(V, Lwl,k, Cr_ML, AT, S)
    [RTS, df] = res_pred.calc_RTS()
    
    with col2:
        V_des = float(st.number_input('Design Speed [knots]', value=12.5))
        st.dataframe(df, hide_index=True, use_container_width=True)
    mm = math(V_des, V, RTS)
    RTSi = mm.interp()
    plot = plot(Fr, RTS, V)
    fig = plot.chart(V_des, RTSi)
    with col1:
        st.pyplot(fig, use_container_width=True)


        

