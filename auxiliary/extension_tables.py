# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:17:41 2021

@author: renuka
"""

import pandas as pd
import statsmodels.api as sma
import statsmodels.formula.api as sm
import statsmodels.stats.sandwich_covariance as sw
import numpy as np
import statsmodels as statsmodels
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col
from linearmodels.panel import PanelOLS

#For NLSY79

def extension_table1(df1):
    
    df2= df1.loc[(df1['age']>=23) & (df1['sample']==0), :]
    com_covs= ['female', 'hisp_male', 'hisp_female', 'black_male' , 'black_female']
    com_covs2= ['age', 'year' ,'div' , 'metro' ,'urban', 'educ']
    base= ['afqt_std', 'soc_nlsy2_std', 'afqt_socnlsy2']
    
    df2= df2[base+ com_covs + com_covs2 + ['business', 'pubid']].dropna()
    df2= pd.get_dummies(df2, columns= com_covs2, drop_first=True)
    
    com_cats= df2.columns[df2.columns.str.startswith('age') | df2.columns.str.startswith('year')| df2.columns.str.startswith('div')| df2.columns.str.startswith('metro')| 
                          df2.columns.str.startswith('urban') | df2.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()
    
    regs2a= df2[com_covs+ com_cats+ base]
    table2a= statsmodels.regression.linear_model.OLS(df2['business'], regs2a, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df2['pubid']}, use_t=True)

    #For NLSY97
    df3= df1.loc[(df1['age']>=23) & (df1['sample']==1), :]
    
    df3= df3[base+ com_covs + com_covs2 + ['business', 'pubid']].dropna()
    df3= pd.get_dummies(df3, columns= com_covs2, drop_first=True)
    
    com_cats= df3.columns[df3.columns.str.startswith('age') | df3.columns.str.startswith('year')| df3.columns.str.startswith('div')| df3.columns.str.startswith('metro')| 
                          df3.columns.str.startswith('urban') | df3.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()
    
    regs3a= df3[com_covs+ com_cats+ ['afqt_std', 'soc_nlsy2_std', 'afqt_socnlsy2']]
    table3a= statsmodels.regression.linear_model.OLS(df3['business'], regs3a, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df3['pubid']}, use_t=True)

    
    business = pd.DataFrame({'Outcome: Business Ownership': ['Cognitive skills (AQT, standardized)', '', 'Social skills (standardized)', '',
                            'Cognitive * Social','', 'Observations'],
                          'NLSY79': [format(table2a.params['afqt_std'], ".3f"), format(table2a.pvalues['afqt_std'], ".3f"), format(table2a.params['soc_nlsy2_std'],".3f"),
                                 format(table2a.pvalues['soc_nlsy2_std'], ".3f"), format(table2a.params['afqt_socnlsy2'],".3f"), format(table2a.pvalues['afqt_socnlsy2'], ".3f"),
                                 table2a.nobs],
                          'NLSY97': [format(table3a.params['afqt_std'], ".3f"), format(table3a.pvalues['afqt_std'], ".3f"), format(table3a.params['soc_nlsy2_std'], ".3f"), 
                                 format(table3a.pvalues['soc_nlsy2_std'], ".3f"), format(table3a.params['afqt_socnlsy2'], ".3f"), format(table3a.pvalues['afqt_socnlsy2'], ".3f") 
                                , table3a.nobs]})
    
    return business


def extension_table2(df1):
    
    #NLSY79 Employees hired
    
    df5= df1.loc[(df1['age']>=23) & (df1['sample']==0), :]
    com_covs= ['female', 'hisp_male', 'hisp_female', 'black_male' , 'black_female']
    com_covs2= ['age', 'year' ,'div' , 'metro' ,'urban', 'educ']
    base= ['afqt_std', 'soc_nlsy2_std', 'afqt_socnlsy2']
    
    df5= df5[base+ com_covs + com_covs2 + ['employees_1', 'pubid']].dropna()
    df5= pd.get_dummies(df5, columns= com_covs2, drop_first=True)
    
    com_cats= df5.columns[df5.columns.str.startswith('age') | df5.columns.str.startswith('year')| df5.columns.str.startswith('div')| df5.columns.str.startswith('metro')| 
                          df5.columns.str.startswith('urban') | df5.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()
    
    regs5a= df5[com_covs+ com_cats+ base]
    table5a= statsmodels.regression.linear_model.OLS(df5['employees_1'], regs5a, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df5['pubid']}, use_t=True)
    
    
    #NLSY97 Employees hired
    
    df4= df1.loc[(df1['age']>=23) & (df1['sample']==1), :]
    
    df4= df4[base+ com_covs + com_covs2 + ['employees_1', 'pubid']].dropna()
    df4= pd.get_dummies(df4, columns= com_covs2, drop_first=True)
    
    com_cats= df4.columns[df4.columns.str.startswith('age') | df4.columns.str.startswith('year')| df4.columns.str.startswith('div')| df4.columns.str.startswith('metro')| 
                          df4.columns.str.startswith('urban') | df4.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()
    
    regs4a= df4[com_covs+ com_cats+ base]
    table4a= statsmodels.regression.linear_model.OLS(df4['employees_1'], regs4a, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df4['pubid']}, use_t=True)
    
    
    no_employees= pd.DataFrame({'Outcome: No. of employees': ['Cognitive skills (AQT, standardized)', '', 'Social skills (standardized)', '',
                            'Cognitive * Social','', 'Observations'],
                          'NLSY79': [format(table5a.params['afqt_std'], ".3f"), format(table5a.pvalues['afqt_std'], ".3f"), format(table5a.params['soc_nlsy2_std'],".3f"),
                                 format(table5a.pvalues['soc_nlsy2_std'], ".3f"), format(table5a.params['afqt_socnlsy2'],".3f"), format(table5a.pvalues['afqt_socnlsy2'], ".3f"),
                                 table5a.nobs],
                          'NLSY97': [format(table4a.params['afqt_std'], ".3f"), format(table4a.pvalues['afqt_std'], ".3f"), format(table4a.params['soc_nlsy2_std'], ".3f"), 
                                 format(table4a.pvalues['soc_nlsy2_std'], ".3f"), format(table4a.params['afqt_socnlsy2'], ".3f"), format(table4a.pvalues['afqt_socnlsy2'], ".3f") 
                                , table4a.nobs]})
    
    return no_employees

def summary_stats3(sum_stats3):
    """
      Creates Table 3 for summary stats.
    """
    variables = sum_stats3[['educ', 'business', 'employees_1', 'soc_nlsy2_std', 'afqt_std', 'female', 
                      'hisp_female','hisp_male', 'black_male', 'black_female']]

    table1 = pd.DataFrame()
    table1['Mean'] = variables.mean()
    table1['Standard Deviation'] = variables.std()
    table1 = table1.astype(float).round(2)
    table1['Description'] = [
                             "Years of completed education", 
                             "Business", 
                             "Number of employees hired",
                             "Social Skills measure",
                             "AFQT (Cognitive skills measure)", 
                             "Female", 
                             "Hispanic Female",
                             "Hispanc Male", 
                             "Black Male", 
                             "Black Female"]


    return table1