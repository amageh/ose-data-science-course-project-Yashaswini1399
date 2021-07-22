# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 10:44:07 2021

@author: renuka
"""

import pandas as pd
import statsmodels.api as sma
import statsmodels.formula.api as sm
import statsmodels.stats.sandwich_covariance as sw
import numpy as np
import statsmodels as statsmodels
from statsmodels.iolib.summary2 import summary_col
from linearmodels.panel import PanelOLS

def replication_table1(df):
    #Getting the vars ready
    df1= df.loc[(df['complete79']==1) & (df['age']>=23) & (df['sample']==0), :]
    df1= pd.get_dummies(df1, columns= ['age', 'year' ,'div' , 'metro' ,'urban', 'educ']) 

    com_covs= ['female', 'hisp_male', 'hisp_female', 'black_male' , 'black_female']
    com_cats= df1.columns[df1.columns.str.startswith('age') | df1.columns.str.startswith('year')| df1.columns.str.startswith('div')| df1.columns.str.startswith('metro')| df1.columns.str.startswith('urban')]
    com_cats= com_cats.tolist()
    com_cats.remove('age_test')
    com_cats2= df1.columns[df1.columns.str.startswith('educ')]
    com_cats2= com_cats2.tolist()
    needed_vars= ['pubid', 'weight']


    regs= [['soc_nlsy_std'], ['afqt_std', 'soc_nlsy_std'], ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy'], ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy', 'noncog_std'], 
          ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy', 'noncog_std', 'afqt_noncog'], ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy', 'noncog_std'], ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy', 'noncog_std', 'afqt_noncog']]

    dfs= []
    regressors= []
    table1= []
    reg1= []

    #The regressions
    for i in range(5):
        dfs.append(df1[com_covs + com_cats + needed_vars + ['ln_wage'] + regs[i]].dropna())

        regressors.append(dfs[i][dfs[i].columns[~dfs[i].columns.isin(['ln_wage','pubid', 'weight'])]])

        table1.append(statsmodels.regression.linear_model.WLS(dfs[i]['ln_wage'], regressors[i], weights= dfs[i]['weight']).fit(cov_type='cluster', cov_kwds={'groups': dfs[i]['pubid']}, use_t=True))

    for i in range(5,7):
        dfs.append(df1[com_covs + com_cats2 + com_cats + needed_vars + ['ln_wage'] + regs[i]].dropna())

        regressors.append(dfs[i][dfs[i].columns[~dfs[i].columns.isin(['ln_wage','pubid', 'weight'])]])

        table1.append(statsmodels.regression.linear_model.WLS(dfs[i]['ln_wage'], regressors[i], weights= dfs[i]['weight']).fit(cov_type='cluster', cov_kwds={'groups': dfs[i]['pubid']}, use_t=True))


    #Forming the table    
    info_dict={'No. of observations' : lambda x: f"{int(x.nobs):d}"}

    final1= summary_col(results=[table1[0], table1[1], table1[2], table1[3], table1[5], table1[4], table1[6]],
                                float_format='%0.3f',
                                stars = True,
                                model_names=['(1)',
                                             '(2)',
                                             '(3)',
                                             '(4)',
                                             '(5)',
                                             '(6)',
                                             '(7)'],
                                info_dict= info_dict,
                                regressor_order=['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy', 'noncog_std', 'afqt_noncog'],
                                drop_omitted= True)
    final1.add_title("Table 1: Labor Market Returns to Cognitive Skills and Social Skills in the NLSY79")

    return final1

def replication_table2(df):
    #Getting the vars ready
    df2= df.loc[(df['complete79']==1) & (df['age']>=23) & (df['sample']==0) & (df['wage'].notnull()), :]
    df2= pd.get_dummies(df2, columns= ['age', 'year' ,'div' , 'metro' ,'urban', 'educ','ind6090'])

    com_covs= ['female', 'hisp_male', 'hisp_female', 'black_male' , 'black_female']
    com_cats= df2.columns[df2.columns.str.startswith('age') | df2.columns.str.startswith('year')| df2.columns.str.startswith('div')| df2.columns.str.startswith('metro')| 
                          df2.columns.str.startswith('urban') | df2.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()
    com_cats.remove('age_test')
    com_cats2= df2.columns[df2.columns.str.startswith('ind6090')]
    com_cats2= com_cats2.tolist()
    needed_vars= ['pubid', 'weight']

    #2a
    regs2a= df2[com_covs+ com_cats+ com_cats2+ ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy']]
    table2a= statsmodels.regression.linear_model.WLS(df2['routine'], regs2a, weights= df2['weight'], hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df2['pubid']}, use_t=True)

    #2b
    regs2b= df2[com_covs+ com_cats+ ['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy', 'math', 'number_facility', 'reason', 'info_use']]
    table2b= statsmodels.regression.linear_model.WLS(df2['routine'], regs2b, weights= df2['weight'], hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df2['pubid']}, use_t=True)

    #2c
    table2c= statsmodels.regression.linear_model.WLS(df2['socskills'], regs2a, weights= df2['weight'], hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df2['pubid']}, use_t=True)

    #2d
    table2d= statsmodels.regression.linear_model.WLS(df2['socskills'], regs2b, weights= df2['weight'], hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df2['pubid']}, use_t=True)

    #Forming the table
    info_dict={'No. of observations' : lambda x: f"{int(x.nobs):d}"}

    final2= summary_col(results=[table2a, table2b, table2c, table2d],
                                float_format='%0.3f',
                                stars = True,
                                model_names=['Routine\n (1)',
                                             'Routine \n(2)',
                                             'Social Skills \n(3)',
                                             'Social Skills \n(4)'],
                                info_dict= info_dict,
                                regressor_order=['afqt_std', 'soc_nlsy_std', 'afqt_socnlsy'],
                                drop_omitted= True)
    final2.add_title("Table 2: Occupational Sorting on Skills in the NLSY79")

    return final2

def replication_table3(df):
    df3= df.loc[(df['complete79']==1) & (df['age']>=23) & (df['sample']==0)]
    year = pd.Categorical(df3.year)
    df3= df3.set_index(['pubid', 'year'])
    df3['year']=year
    df3= pd.get_dummies(df3, columns= ['age' ,'div' , 'metro' ,'urban', 'year']) 


    com_cats= df3.columns[df3.columns.str.startswith('age') | df3.columns.str.startswith('div')| df3.columns.str.startswith('metro')| 
                          df3.columns.str.startswith('urban') | df3.columns.str.startswith('year')]
    com_cats= com_cats.tolist()
    com_cats.remove('age_test')

    #3a
    df3a= df3[['ln_wage','routine', 'afqt_routine', 'socnlsy_routine', 'afqt_socnlsy_routine', 'weight']+ com_cats].dropna()
    exog_a= ['routine', 'afqt_routine', 'socnlsy_routine', 'afqt_socnlsy_routine']+ com_cats
    exog_a= df3a[exog_a].drop(columns= ['year_2010', 'div_4.0', 'year_2012', 'metro_4.0', 'urban_2.0'])
    mod_a= PanelOLS(df3a['ln_wage'], exog_a, weights= df3a['weight'], entity_effects=True)
    fit_a= mod_a.fit(cov_type='clustered', cluster_entity=True)

    #3b
    df3b= df3[['ln_wage', 'socskills', 'afqt_socskills', 'socnlsy_socskills', 'afqt_socnlsy_socskills', 'weight']+com_cats].dropna()
    exog_b= ['socskills', 'afqt_socskills', 'socnlsy_socskills', 'afqt_socnlsy_socskills']+com_cats
    exog_b= df3b[exog_b].drop(columns= ['year_2010', 'div_4.0', 'year_2012', 'metro_4.0', 'urban_2.0'])
    mod_b= PanelOLS(df3b['ln_wage'], exog_b, weights= df3b['weight'], entity_effects=True)
    fit_b= mod_b.fit(cov_type='clustered', cluster_entity=True)

    #3c
    df3c= df3[['ln_wage', 'routine', 'afqt_routine', 'socnlsy_routine', 'afqt_socnlsy_routine', 'socskills', 'afqt_socskills', 'socnlsy_socskills', 'afqt_socnlsy_socskills',  'weight']+com_cats].dropna()
    exog_c= ['routine', 'afqt_routine', 'socnlsy_routine', 'afqt_socnlsy_routine','socskills', 'afqt_socskills', 'socnlsy_socskills', 'afqt_socnlsy_socskills']+com_cats
    exog_c= df3c[exog_c].drop(columns= ['year_2010', 'div_4.0', 'year_2012', 'metro_4.0', 'urban_2.0'])
    mod_c= PanelOLS(df3c['ln_wage'], exog_c, weights= df3c['weight'], entity_effects=True)
    fit_c= mod_c.fit(cov_type='clustered', cluster_entity=True)


    #Forming the table
    final3 = pd.DataFrame({'Outcome: log hourly wages': ['Routine task intensity', '', 'Cognitive * Routine task intensity', '',
                            'Social skills * Routine task intensity','', 'Cognitive * Social * Routine task intensity', '',
                            'Social skill task intensity', '', 'Cognitive * Social skill task intensity', '', 'Social skills * Social skill task intensity',
                            '','Cognitive * Social * Social skill task intensity', '', 'Observations'],
                          '(1)': [format(fit_a.params['routine'], ".4f"), format(fit_a.pvalues['routine'], ".4f"), format(fit_a.params['afqt_routine'],".4f"),
                                 format(fit_a.pvalues['afqt_routine'], ".4f"), format(fit_a.params['socnlsy_routine'],".4f"), format(fit_a.pvalues['socnlsy_routine'], ".4f"),
                                 format(fit_a.params['afqt_socnlsy_routine'], ".4f"), format(fit_a.pvalues['afqt_socnlsy_routine'], ".4f"), '', '', '',
                                 '', '', '', '', '', fit_a.nobs],
                          '(2)': ['', '', '', '', '', '', '', '', format(fit_b.params['socskills'], ".4f"), format(fit_b.pvalues['socskills'], ".4f"), format(fit_b.params['afqt_socskills'], ".4f"), 
                                 format(fit_b.pvalues['afqt_socskills'], ".4f"), format(fit_b.params['socnlsy_socskills'], ".4f"), format(fit_b.pvalues['socnlsy_socskills'], ".4f"), 
                                 format(fit_b.params['afqt_socnlsy_socskills'], ".4f"), format(fit_b.pvalues['afqt_socnlsy_socskills'], ".4f"), fit_b.nobs],
                          '(3)': [format(fit_c.params[0], ".4f"), format(fit_c.pvalues[0], ".4f"), format(fit_c.params[1], ".4f"), format(fit_c.pvalues[1], ".4f"), 
                                 format(fit_c.params[2], ".4f"), format(fit_c.pvalues[2], ".4f"), format(fit_c.params[3], ".4f"), format(fit_c.pvalues[3], ".4f"), 
                                 format(fit_c.params[4], ".4f"), format(fit_c.pvalues[4], ".4f"), format(fit_c.params[5], ".4f"), format(fit_c.pvalues[5], ".4f"),
                                 format(fit_c.params[6], ".4f"), format(fit_c.pvalues[6], ".4f"), format(fit_c.params[7], ".4f"), format(fit_c.pvalues[7], ".4f"),
                                 fit_c.nobs]})
    return final3

def replication_table4(df):
    
    #Getting the vars ready
    df4= df.loc[(df['age'] >=25) & (df['age'] <=33), :]
    com_covs= ['female', 'hisp_male', 'hisp_female', 'black_male' , 'black_female']
    com_covs2= ['age', 'year' ,'div' , 'metro' ,'urban']
    com_covs3= ['age', 'year' ,'div' , 'metro' ,'urban', 'educ']
    base= ['afqt_std', 'afqt_sample', 'soc_nlsy2_std', 'soc_nlsy2_sample', 'afqt_socnlsy2', 'afqt_socnlsy2_sample', 'sample']
    base2= ['afqt_std', 'afqt_sample', 'soc_nlsy2_std', 'soc_nlsy2_sample', 'afqt_socnlsy2', 'afqt_socnlsy2_sample', 'noncog_std', 'noncog_sample', 'sample']

    df5= df.loc[(df['age'] >=25) & (df['age'] <=33) & (df['complete97']==1), :]

    #4.1
    df4a= df4[base + com_covs + com_covs2 + ['pubid', 'emp']].dropna()
    df4a= pd.get_dummies(df4a, columns= com_covs2, drop_first=True)

    com_cats= df4a.columns[df4a.columns.str.startswith('age') | df4a.columns.str.startswith('year')| df4a.columns.str.startswith('div')| df4a.columns.str.startswith('metro')| 
                          df4a.columns.str.startswith('urban')]
    com_cats= com_cats.tolist()

    regs4a= df4a[base + com_covs + com_cats]
    table4a= statsmodels.regression.linear_model.OLS(df4a['emp'], regs4a, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df4a['pubid']}, use_t=True)

    #4.2
    df4b= df4[base + com_covs + com_covs3 + ['pubid', 'emp']].dropna()
    df4b= pd.get_dummies(df4b, columns= com_covs3, drop_first=True)

    com_cats= df4b.columns[df4b.columns.str.startswith('age') | df4b.columns.str.startswith('year')| df4b.columns.str.startswith('div')| df4b.columns.str.startswith('metro')| 
                          df4b.columns.str.startswith('urban')| df4b.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()

    regs4b= df4b[base + com_covs + com_cats]
    table4b= statsmodels.regression.linear_model.OLS(df4b['emp'], regs4b, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df4b['pubid']}, use_t=True)

    #4.3
    df4c= df4[base2 + com_covs + com_covs3 + ['pubid', 'emp']].dropna()
    df4c= pd.get_dummies(df4c, columns= com_covs3, drop_first=True)

    com_cats= df4c.columns[df4c.columns.str.startswith('age') | df4c.columns.str.startswith('year')| df4c.columns.str.startswith('div')| df4c.columns.str.startswith('metro')| 
                          df4c.columns.str.startswith('urban')| df4c.columns.str.startswith('educ')]
    com_cats= com_cats.tolist()

    regs4c= df4c[base2 + com_covs + com_cats]
    table4c= statsmodels.regression.linear_model.OLS(df4c['emp'], regs4c, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df4c['pubid']}, use_t=True)

    #4.4
    df5a= df5[base + com_covs + com_covs2 + ['pubid', 'ln_wage']].dropna()                                                  
    df5a= pd.get_dummies(df5a, columns= com_covs2, drop_first=True)  

    com_cats= df5a.columns[df5a.columns.str.startswith('age') | df5a.columns.str.startswith('year')| df5a.columns.str.startswith('div')| df5a.columns.str.startswith('metro')| 
                          df5a.columns.str.startswith('urban')]
    com_cats= com_cats.tolist() 

    regs5a= df5a[base + com_covs + com_cats]
    table5a= statsmodels.regression.linear_model.OLS(df5a['ln_wage'], regs5a, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df5a['pubid']}, use_t=True)

    #4.5
    df5b= df5[base + com_covs + com_covs3 + ['pubid', 'ln_wage']].dropna()                                                  
    df5b= pd.get_dummies(df5b, columns= com_covs3, drop_first=True)  

    com_cats= df5b.columns[df5b.columns.str.startswith('age') | df5b.columns.str.startswith('year')| df5b.columns.str.startswith('div')| df5b.columns.str.startswith('metro')| 
                          df5b.columns.str.startswith('urban')| df5b.columns.str.startswith('educ')]
    com_cats= com_cats.tolist() 

    regs5b= df5b[base + com_covs + com_cats]
    table5b= statsmodels.regression.linear_model.OLS(df5b['ln_wage'], regs5b, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df5b['pubid']}, use_t=True)
    table5b.summary()

    #4.6
    df5c= df5[base2 + com_covs + com_covs3 + ['pubid', 'ln_wage']].dropna()                                                  
    df5c= pd.get_dummies(df5c, columns= com_covs3, drop_first=True)  

    com_cats= df5c.columns[df5c.columns.str.startswith('age') | df5c.columns.str.startswith('year')| df5c.columns.str.startswith('div')| df5c.columns.str.startswith('metro')| 
                          df5c.columns.str.startswith('urban')| df5c.columns.str.startswith('educ')]
    com_cats= com_cats.tolist() 

    regs5c= df5c[base2 + com_covs + com_cats]
    table5c= statsmodels.regression.linear_model.OLS(df5c['ln_wage'], regs5c, hasconst=True).fit(cov_type='cluster', cov_kwds={'groups': df5c['pubid']}, use_t=True)

    #Forming the table
    form1= ['afqt_std', 'afqt_sample', 'soc_nlsy2_std', 'soc_nlsy2_sample', 'afqt_socnlsy2', 'afqt_socnlsy2_sample']
    form2= ['afqt_std', 'afqt_sample', 'soc_nlsy2_std', 'soc_nlsy2_sample', 'afqt_socnlsy2', 'afqt_socnlsy2_sample', 'noncog_std', 'noncog_sample']

    vals= []
    vals1= []
    tables= [table4a, table4b, table5a, table5b]
    tables1= [table4c, table5c]

    for table in tables: 
        for var in form1:
            i= format(table.params[var], ".3f")
            j= format(table.pvalues[var], ".3f")
            vals.append(i)
            vals.append(j)

    for table in tables1: 
        for var in form2:
            i= format(table.params[var], ".3f")
            j= format(table.pvalues[var], ".3f")
            vals1.append(i)
            vals1.append(j)

    final4= pd.DataFrame({'Variables': ['Cognitive skills (AQT, standardized)', '', 'Cognitive skills * NLSY97', '',
                            'Social skills (standardized)','', 'Social skills * NLSY97', '',
                            'Cognitive * Social', '', 'Cognitive * Social * NLSY97', '', 'Noncognitive skills (standardized)',
                            '','Noncognitive skills * NLSY97', '', 'Observations'],
                          'Full time employment (1)': [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                                 vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], '', '', '', '', table4a.nobs],
                          'Full time employment (2)': [vals[12], vals[13], vals[14], vals[15], vals[16], vals[17],
                                 vals[18], vals[19], vals[20], vals[21], vals[22], vals[23], '', '', '', '', table4b.nobs],
                          'Full time employment (3)': [vals1[0], vals1[1], vals1[2], vals1[3], vals1[4], vals1[5],
                                 vals1[6], vals1[7], vals1[8], vals1[9], vals1[10], vals1[11], vals1[12], vals1[13], 
                                vals1[14], vals1[15], table4c.nobs],
                          'Log hourly wage (4)': [vals[24], vals[25], vals[26], vals[27], vals[28], vals[29],
                                 vals[30], vals[31], vals[32], vals[33], vals[34], vals[35], '', '', '', '', table5a.nobs], 
                         'Log hourly wage (5)': [vals[36], vals[37], vals[38], vals[39], vals[40], vals[41],
                                 vals[42], vals[43], vals[44], vals[45], vals[46], vals[47], '', '', '', '', table5b.nobs],
                         'Log hourly wage (6)': [vals1[16], vals1[17], vals1[18], vals1[19], vals1[20], vals1[21],
                                 vals1[22], vals1[23], vals1[24], vals1[25], vals1[26], vals1[27], vals1[28], vals1[29], 
                                vals1[30], vals1[31], table4c.nobs]})
    return final4

def summary_stats1(sum_stats):
    """
      Creates Table 1 for summary stats.
    """
    variables = sum_stats[['educ', 'wage', 'soc_nlsy_std', 'afqt_std', 'female', 
                      'hisp_female','hisp_male', 'black_male', 'black_female']]

    table1 = pd.DataFrame()
    table1['Mean'] = variables.mean()
    table1['Standard Deviation'] = variables.std()
    table1 = table1.astype(float).round(2)
    table1['Description'] = [
                             "Years of completed education", 
                             "Hourly wage (2013 level)", 
                             "Social Skills measure",
                             "AFQT (Cognitive skills measure)", 
                             "Female", 
                             "Hispanic Female",
                             "Hispanc Male", 
                             "Black Male", 
                             "Black Female"]


    return table1

def summary_stats2(sum_stats2):
    """
      Creates Table 1 for summary stats.
    """
    variables = sum_stats2[['educ', 'wage', 'soc_nlsy2_std', 'afqt_std', 'female', 
                      'hisp_female','hisp_male', 'black_male', 'black_female']]

    table1 = pd.DataFrame()
    table1['Mean'] = variables.mean()
    table1['Standard Deviation'] = variables.std()
    table1 = table1.astype(float).round(2)
    table1['Description'] = [
                             "Years of completed education", 
                             "Hourly wage (2013 level)", 
                             "Social Skills measure",
                             "AFQT (Cognitive skills measure)", 
                             "Female", 
                             "Hispanic Female",
                             "Hispanc Male", 
                             "Black Male", 
                             "Black Female"]


    return table1