# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 00:49:12 2017

@author: Rodrigo
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import glob

#-- Mudar a variavel DIRS de acordo com o diretorio
DIRS = r"C:\Users\rodrigo.goncalves\Documents\Base de dados\alertario\anchieta\anchieta"
posto = 'anchieta'

#-- le a serie csv com data e prec. diaria
lst1 = glob.glob(os.path.normpath(DIRS)+os.sep+'*.txt')
lst1 = [l for l in lst1 if posto in l]
lstAnos = list(set([os.path.split(l)[1].replace(posto,'').split('_')[1][:4] for l in lst1]))

lstDFAnos = []
#d15, d60, d240, d1440, d5760 = ({} for i in range(5))
dfMaxs = pd.DataFrame()
for ano in lstAnos:
    lstDFAno = []
    lstAnoAtual = [n for n in lst1 if ano in n]
    for arqA in lstAnoAtual:
        ds2 = pd.read_fwf(arqA, widths=[12,10,6,9,7,7,7,4], skiprows=4, parse_dates=True)
        cols = ['15 min','01 h','04 h','24 h','96 h']
        ds2[cols] = ds2[cols].apply(pd.to_numeric,errors='coerce')
        # Fazer mais acumulações que nao tem nos dados originais
        ds2['30 min'] = ds2['15 min'].rolling(window=2,center=False).sum().round(2)
        ds2['02 h'] = ds2['15 min'].rolling(window=8,center=False).sum().round(2)
        ds2['06 h'] = ds2['15 min'].rolling(window=24,center=False).sum().round(2)
        ds2['12 h'] = ds2['15 min'].rolling(window=48,center=False).sum().round(2)
        ds2['16 h'] = ds2['15 min'].rolling(window=64,center=False).sum().round(2)
        lstDFAno.append(ds2)
    dfAno = pd.concat(lstDFAno)
    
    dfMaxs = pd.concat([dfMaxs,pd.DataFrame({15:dfAno['15 min'].max(),30:dfAno['30 min'].max(),60:dfAno['01 h'].max(),
                                             120:dfAno['02 h'].max(),240:dfAno['04 h'].max(),360:dfAno['06 h'].max(),
                                             720:dfAno['12 h'].max(),960:dfAno['16 h'].max(),1440:dfAno['24 h'].max()},
                                                index=[int(ano)])])
    '''
    d15[ano] = dfAno['15 min'].max()
    d60[ano] = dfAno['01 h'].max()
    d240[ano] = dfAno['04 h'].max() 
    d1440[ano] = dfAno['24 h'].max() 
    d5760[ano] =  dfAno['96 h'].max()
    '''
    lstDFAnos.append(dfAno)

dfMaxs = pd.DataFrame.sort_index(dfMaxs)

# --------------------

#lista de duracoes
dfPdf = pd.DataFrame(index=[5,10,20,25,50,100,500])
for col in dfMaxs:
    #dfAc = pd.DataFrame.from_dict(dur,orient='index')
    dfAc = dfMaxs[col]
    serY = dfAc.copy()
    y = dfAc.copy()
    
    # descobre melhores parametros das distribuicoes segundo metodo dos momentos-L
    from lmoments3 import distr
    #import lmoments3 as lm
    
    paras = distr.gev.lmom_fit(y)
    parGB = distr.gum.lmom_fit(y)
    
    #seta os parametros das distribuicoes e gera distribuicoes segundo o dominio especificado
    
    #-- dominio para plotar as dist. estatisticas
    dom1 = np.arange(0,400,1)
    k1= paras['c']
    l1= paras['loc']
    s1= paras['scale']
    d1GEV = stats.genextreme.pdf(dom1,k1,loc=l1, scale=s1)
    vals = stats.gumbel_r.pdf(dom1,loc=parGB['loc'], scale=parGB['scale'])
    
    #-- Tabela TRS x Pmax
    lstTRS = np.array([5,10,20,25,50,100,500])
    pGEV = stats.genextreme.ppf([1-(1/TR) for TR in lstTRS], k1,loc=l1,scale=s1)
    pGB = stats.gumbel_r.ppf([1-(1/TR) for TR in lstTRS],loc=parGB['loc'], scale=parGB['scale'])
    tabTRST = pd.DataFrame.from_items([('TR (anos)',lstTRS),(str(col)+' (mm) - GEV ',pGEV),(str(col)+' (mm) - GUMBEL',pGB)])
    tabTRST = tabTRST.set_index(['TR (anos)'])
    # escolhe GEV como distribuicao a ser utilizada
    dfPdf[col]=tabTRST[str(col)+' (mm) - GEV ']
    print(tabTRST)
    
dfPdf.transpose().plot(xlim=(15,960))

dfIDF  = dfPdf.div(dfPdf.columns.to_series()/60.0, axis=1)

#so plota as duracoes de 15 a 960 minutos
dfIDF.transpose().plot(xlim=(15,960),xticks=np.arange(0,960,60),grid=True)
dfIDF.to_clipboard()
