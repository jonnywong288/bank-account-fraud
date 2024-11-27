import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr, ttest_ind, spearmanr, ks_2samp
from sklearn.linear_model import LogisticRegression
pd.set_option("display.max_colwidth", None)

class FeatureSignificance:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def overview(self):
        columns = []
        dtype = []
        minus_one_count = []
        minus_one_percentage = []
        unique_values = []

        for i in self.df.columns:

            minus_one_c = (self.df[i] == -1).sum()
            minus_one_p = (minus_one_c/len(self.df)) * 100


            columns.append(i)
            dtype.append(self.df[i].dtype)
            minus_one_count.append(minus_one_c)
            minus_one_percentage.append(minus_one_p)
            unique_values.append(self.df[i].nunique())

        return pd.DataFrame({'Feature':columns,
                            'dtype':dtype,
                            '-1 Count':minus_one_count,
                            '-1 (%)': minus_one_percentage,
                            'unique values': unique_values}).set_index('Feature')

    def calculations(self):
        info_summary = {
            "temporal": ["Spearman Correlation", "Chi-Square Test (Goodness of Fit)"],  
            "ordinal": ["Spearman Correlation", "Logistic Regression Coefficient"],
            "numerical_discrete": ["Point-Biserial Correlation", "T-Test"],
            "numerical_continuous": ["Spearman Correlation", "T-Test", "K-S Test"],
            "nominal_multi_category": ["Chi-Square Test of Independence (Contingency)", "Cramér’s V"],
            "nominal_binary": ["Chi-Square Test of Independence (Contingency)", "Point-Biserial"],
        }
        print(pd.Series(info_summary))


    def nominal_multi_category(self, variables):

        features = []
        chi_square_statistics = []
        p_values = []
        cramers_vs = []

        for i in variables:
            chi_square_statistic, p_value = chi2_contingency(pd.crosstab(self.df[i], 
                                                                         self.df[self.target]))[:2]
            
            n = self.df.shape[0]
            min_dim = 1
            cramers_v = np.sqrt(chi_square_statistic / (n * min_dim))
            
            features.append(i)
            chi_square_statistics.append(chi_square_statistic)
            p_values.append(p_value)
            cramers_vs.append(cramers_v)

        nominal_multi_category_significance = pd.DataFrame({'Feature':features, 
                                      'Chi Square Statistic':chi_square_statistics, 
                                      'p-value (Chi-Square)': p_values,
                                      'cramers v': cramers_vs
                                      }).sort_values('cramers v', ascending=False).set_index('Feature')
        return nominal_multi_category_significance
    
    def nominal_binary(self, variables):

        features = []
        chi_square_statistics = []
        p_values = []
        point_biserial_correlations = []
        point_biserial_p_values = []

        for i in variables:
            chi_square_statistic, p_value = chi2_contingency(pd.crosstab(self.df[i], 
                                                                         self.df[self.target]))[:2]
            
            correlation, p_value_biserial = pointbiserialr(self.df[i], self.df[self.target])

            features.append(i)
            chi_square_statistics.append(chi_square_statistic)
            p_values.append(p_value)
            point_biserial_correlations.append(correlation)
            point_biserial_p_values.append(p_value_biserial)

        nominal_binary_significance = pd.DataFrame({
                                'Feature': features,
                                'Chi Square Statistic': chi_square_statistics, 
                                'p-value (Chi-Square)': p_values,
                                'Point-Biserial Correlation': point_biserial_correlations,
                                'p-value (Point-Biserial)': point_biserial_p_values
                            }).sort_values('Point-Biserial Correlation', ascending=False).set_index('Feature')
        return nominal_binary_significance
    
    def numerical_discrete(self, variables):

        features = []
        correlations = []
        p_values = []
        t_stats = []
        t_stat_p_values = []

        for i in variables:
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
            
        numerical_discrete_significance = pd.DataFrame({'Feature':features, 
                                            'Point-Biserial Correlation':correlations, 
                                            'p-value': p_values,
                                            't_stat': t_stats,
                                            't_stat p_value': t_stat_p_values}).sort_values('Point-Biserial Correlation', ascending=False).set_index('Feature')
        return numerical_discrete_significance
    
    def numerical_continuous(self, variables):

        features = []
        spearman_correlations = []
        spearman_p_values = []
        t_stats = []
        t_stat_p_values = []
        ks_statistics = []
        ks_p_values = []

        for i in variables:
            df_i = self.df[self.df[i] != -1]  # Remove missing values

            # Spearman Correlation
            spearman_corr, spearman_p = spearmanr(df_i[i], df_i[self.target])

            # Welch's T-Test for mean comparison
            group_0 = df_i[df_i[self.target] == 0][i]
            group_1 = df_i[df_i[self.target] == 1][i]
            t_stat, t_p_value = ttest_ind(group_0, group_1, equal_var=False)

            # Kolmogorov-Smirnov (K-S) Test for distribution comparison
            ks_stat, ks_p_value = ks_2samp(group_0, group_1)

            # Append results
            features.append(i)
            spearman_correlations.append(spearman_corr)
            spearman_p_values.append(spearman_p)
            t_stats.append(t_stat)
            t_stat_p_values.append(t_p_value)
            ks_statistics.append(ks_stat)
            ks_p_values.append(ks_p_value)

        numerical_continuous_significance = pd.DataFrame({
                'Feature': features,
                'Spearman Correlation': spearman_correlations,
                'Spearman p-value': spearman_p_values,
                'T-Statistic': t_stats,
                'T-Test p-value': t_stat_p_values,
                'K-S Statistic': ks_statistics,
                'K-S p-value': ks_p_values
            }).sort_values('Spearman Correlation', ascending=False).set_index('Feature')

        return numerical_continuous_significance

    def ordinal(self, variables):

        features = []
        correlations = []
        p_values = []
        log_reg_coefs = []

        for i in variables:
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
                                            'Spearman p-value': p_values,
                                            'Log Regression coef': log_reg_coefs}).sort_values('Log Regression coef', ascending=False).set_index('Feature')
        return ordinal_variables_significance

    def temporal(self, variables):

        features = []
        spearman_correlations = []
        spearman_p_values = []
        chi_square_statistics = []
        chi_square_p_values = []

        for i in variables:
            # Spearman Correlation for temporal trends
            spearman_corr, spearman_p = spearmanr(self.df[i], self.df[self.target])

            # Chi-Square Test for temporal vs target relationship (treating month as categorical)
            chi_square_stat, chi_square_p = chi2_contingency(pd.crosstab(self.df[i], self.df[self.target]))[:2]

            features.append(i)
            spearman_correlations.append(spearman_corr)
            spearman_p_values.append(spearman_p)
            chi_square_statistics.append(chi_square_stat)
            chi_square_p_values.append(chi_square_p)

        temporal_significance = pd.DataFrame({
            'Feature': features,
            'Spearman Correlation': spearman_correlations,
            'p-value (Spearman)': spearman_p_values,
            'Chi-Square Statistic': chi_square_statistics,
            'p-value (Chi-Square)': chi_square_p_values
        }).sort_values('Spearman Correlation', ascending=False).set_index('Feature')

        return temporal_significance


    

class FeatureVisualisation:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def calculations(self):
        info_summary = {
        "Temporal": {
            "Primary Test/Metric": ["Spearman Correlation", "Chi-Square Test (Goodness of Fit)"],
            "Other Techniques": ["Time bucketing", "Logistic regression"],
            "Visualization": ["Line plots", "Heatmaps"]
        },
        "Ordinal": {
        "Primary Test/Metric": ["Spearman Correlation", "Logistic Regression Coefficient"],
        "Other Techniques": ["Ordinal encoding", "Examine monotonic trends"],
        "Visualization": ["Bar plots", "Line plots"]
        },
        "Numerical Discrete": {
            "Primary Test/Metric": ["Point-Biserial Correlation", "T-Test"],
            "Other Techniques": ["Bucketing for Chi-Square", "Comparison of means"],
            "Visualization": ["Boxplots", "Histograms"]
        },
        "Numerical Continuous (Bounded)": {
            "Primary Test/Metric": ["Spearman Correlation", "T-Test", "K-S Test"],
            "Other Techniques": ["Examine distributions", "Transform skewed data"],
            "Visualization": ["Density plots", "Boxplots"]
        },
        "Numerical Continuous (Unbounded)": {
            "Primary Test/Metric": ["Spearman Correlation", "T-Test", "K-S Test"],
            "Other Techniques": ["Outlier detection", "Log transformations"],
            "Visualization": ["Scatterplots", "Histograms"]
        },
        "Nominal (Multi-Category)": {
            "Primary Test/Metric": ["Chi-Square Test of Independence (Contingency)", "Cramér’s V"],
            "Other Techniques": ["One-hot encoding", "Clustering to reduce categories"],
            "Visualization": ["Bar charts", "Stacked bar charts"]
        },
        "Nominal (Binary)": {
            "Primary Test/Metric": ["Chi-Square Test of Independence (Contingency)", "Point-Biserial"],
            "Other Techniques": ["Logistic regression coefficients"],
            "Visualization": ["Bar charts"]
        }
        }
        return pd.DataFrame(info_summary).T[['Visualization']]