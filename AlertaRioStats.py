# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 00:49:12 2017

@author: Rodrigo
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
from lmoments3 import distr
import time
start_time = time.time()

#-- mudar DIRS colocando o diretorio onde encontram-se os txt's do alertario
DIRS = r'G:\Python_Projs\python3\alertario\sepetiba'
#-- mudar a variavel posto com o prefixo do arquivo txt do alertario
posto = 'sepetiba'

#-- le a serie csv com data e prec. diaria
lst1 = glob.glob(os.path.normpath(DIRS)+os.sep+'*.txt')
lst1 = [l for l in lst1 if posto in l]
lstAnos = list(set([os.path.split(l)[1].replace(posto,'').split('_')[1][:4] for l in lst1]))

lstDFAnos = pd.DataFrame()
#d15, d60, d240, d1440, d5760 = ({} for i in range(5))

for ano in lstAnos:
    lstDFAno = []
    lstAnoAtual = [n for n in lst1 if ano in n]
    for arqA in lstAnoAtual:
        ds2 = pd.read_fwf(arqA, widths=[22,6,9,7,7,7,4], skiprows=4, parse_dates=[0],dayfirst=True)
        cols = ['15 min']
        ds2[cols] = ds2[cols].apply(pd.to_numeric,errors='coerce')
        ds2 = pd.DataFrame({'Data':ds2['Dia         Hora'],'15 min':ds2['15 min']})
        ds2 = ds2.sort_values('Data')
        lstDFAno.append(ds2)
    dfAno = pd.concat(lstDFAno)

    lstDFAnos =pd.concat([lstDFAnos, dfAno])

lstDFAnos = lstDFAnos.sort_values('Data')
lstDFAnos = lstDFAnos.reset_index(drop=True)
lstDFAnos = lstDFAnos.dropna()  #-- filtra valores NaN - linhas sem medicoes de 15 minutos


#-- Fazer as acumulações. Nao acreditei nos dados originais... Fiz a acumulação por mim mesmo.
lstDFAnos['30 min'] = lstDFAnos['15 min'].rolling(window=2,center=False).sum().round(2)
lstDFAnos['01 h'] = lstDFAnos['15 min'].rolling(window=4,center=False).sum().round(2)
lstDFAnos['02 h'] = lstDFAnos['15 min'].rolling(window=8,center=False).sum().round(2)
lstDFAnos['04 h'] = lstDFAnos['15 min'].rolling(window=16,center=False).sum().round(2)
lstDFAnos['06 h'] = lstDFAnos['15 min'].rolling(window=24,center=False).sum().round(2)
lstDFAnos['12 h'] = lstDFAnos['15 min'].rolling(window=48,center=False).sum().round(2)
lstDFAnos['16 h'] = lstDFAnos['15 min'].rolling(window=64,center=False).sum().round(2)
lstDFAnos['24 h'] = lstDFAnos['15 min'].rolling(window=96,center=False).sum().round(2)

print("--- %.3f seconds --- pra processar os arquivos." % (time.time() - start_time))


dfMaxs = pd.DataFrame()
for ano in lstAnos:
    dfAno= lstDFAnos.loc[lstDFAnos['Data'].dt.year==int(ano)]
    dfMaxs = pd.concat([dfMaxs,pd.DataFrame({15:dfAno['15 min'].max(),30:dfAno['30 min'].max(),60:dfAno['01 h'].max(),
                                             120:dfAno['02 h'].max(),240:dfAno['04 h'].max(),360:dfAno['06 h'].max(),
                                             720:dfAno['12 h'].max(),960:dfAno['16 h'].max(),1440:dfAno['24 h'].max()},
                                                index=[int(ano)])])

dfMaxs = pd.DataFrame.sort_index(dfMaxs)

    

# --------------------

#lista de TEMPOS DE RECORRENCIA
dfPdf = pd.DataFrame(index=[5,10,20,25,50,100,500])

for col in dfMaxs:
    #dfAc = pd.DataFrame.from_dict(dur,orient='index')
    dfAc = dfMaxs[col]
    y = dfAc.copy()
    
    # descobre melhores parametros das distribuicoes segundo metodo dos momentos-L
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
    #escolhe GEV para 
    dfPdf[col]=tabTRST[str(col)+' (mm) - GUMBEL']
    print(tabTRST)
    
dfPdf.transpose().plot(xlim=(15,960))

dfIDF  = dfPdf.div(dfPdf.columns.to_series()/60.0, axis=1)

dfIDF.transpose().plot(xlim=(15,960),xticks=np.arange(0,960,60),grid=True)
dfIDF.to_clipboard()

