# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:30:25 2023

@author: michalk
"""

class residual:
    
    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
    
    def calc(self):
        residual = self.y_pred-self.y_test
        residual = residual.reset_index()
        max_residual = max(abs(residual))
        return(residual, max_residual)
    
    
  
    #ind = np.where(abs(res) == max_res)
    #ax0.plot(y_pred, res,'o', markeredgecolor=color, label=lbl)
    ##ax0.vlines(x = y_pred[ind], ymin = 0, ymax = res.iloc[ind].values, color='dark%s' %color)
    #ax0.set_ylim(-1.5,1.5)
    #ax0.set_xlim(-1,8)
    #ax0.set_title('Target Residual plot', fontsize=22)
    #ax0.set_ylabel('Residuals')
    #ax0.set_xlabel('Predictied values')
    #ax0.legend()
    #sns.histplot(y=res, bins=30,ax=ax1, color='light%s'%color, stat='density')
    #ns.kdeplot(y=res, color=color, ax=ax1)
    #ax1.axhline(y=0, xmin=0, xmax=5, c='black', zorder=1)
    #x1.set_ylim(-1.5, 1.5)
    #ax1.text(0.3,1-0.05*i, '%s_mean = %.2f' %(lbl,np.mean(res)), transform=ax1.transAxes, fontsize=13)
    #ax1.axis('off')
        
        
        