import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr, ttest_ind, spearmanr
from sklearn.linear_model import LogisticRegression
import json

class FeatureStats:
    def __init__(self, df, target, dtypes):
        self.df = df
        self.target = target
        with open(dtypes) as f:
            data = json.load(f)
        self.dtypes = data

    def nominal_stats(self):
        nominals = self.dtypes['nominal']

        features = []
        chi_square_statistics = []
        p_values = []
        cramers_vs = []

        for i in nominals:
            chi_square_statistic, p_value = chi2_contingency(pd.crosstab(self.df[i], 
                                                                         self.df[self.target]))[:2]
            
            n = self.df.shape[0]
            min_dim = 1
            cramers_v = np.sqrt(chi_square_statistic / (n * min_dim))
            
            features.append(i)
            chi_square_statistics.append(chi_square_statistic)
            p_values.append(p_value)
            cramers_vs.append(cramers_v)

        nominal_variables_significance = pd.DataFrame({'Feature':features, 
                                      'Chi Square Statistic':chi_square_statistics, 
                                      'p-value': p_values,
                                      'cramers v': cramers_vs}).sort_values('cramers v', ascending=False).set_index('Feature')
        return nominal_variables_significance
    
    def numerical_stats(self):
        numericals = self.dtypes['numerical']

        features = []
        correlations = []
        p_values = []
        t_stats = []
        t_stat_p_values = []

        for i in numericals:
            df_i = self.df[self.df[i] != -1] #remove missing values

            # point-biserial correlation
            correlation, p_value = pointbiserialr(df_i[i], df_i[self.target])

            # welch's t-test for mean comparison
            group_0 = df_i[df_i[self.target] == 0][i]
            group_1 = df_i[df_i[self.target] == 1][i]
            t_stat, p_value_of_t_stat = ttest_ind(group_0, group_1, equal_var=False)

            features.append(i)
            correlations.append(correlation)
            p_values.append(p_value)
            t_stats.append(t_stat)
            t_stat_p_values.append(p_value_of_t_stat)
            
        numerical_variables_significance = pd.DataFrame({'Feature':features, 
                                            'Point-Biserial Correlation':correlations, 
                                            'p-value': p_values,
                                            't_stat': t_stats,
                                            't_stat p_value': t_stat_p_values}).sort_values('Point-Biserial Correlation', ascending=False).set_index('Feature')
        return numerical_variables_significance
    
    def ordinal_stats(self):
        ordinals = self.dtypes['ordinal']

        features = []
        correlations = []
        p_values = []
        log_reg_coefs = []

        for i in ordinals:
            df_i = self.df[self.df[i] != -1] #remove missing values

            # spearman correlation
            correlation, p_value = spearmanr(df_i[i], df_i[self.target])

            # logistic regression coefficient check
            X = df_i[[i]]
            y = df_i[self.target]
            model = LogisticRegression()
            model.fit(X, y)
            log_reg_coef = model.coef_[0][0] 

            features.append(i)
            correlations.append(correlation)
            p_values.append(p_value)
            log_reg_coefs.append(log_reg_coef)

        ordinal_variables_significance = pd.DataFrame({'Feature':features, 
                                            'Spearman Correlation':correlations, 
                                            'p-value': p_values,
                                            'Log Regression coef': log_reg_coefs}).sort_values('Log Regression coef', ascending=False).set_index('Feature')
        return ordinal_variables_significance
