import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr, ttest_ind, spearmanr, ks_2samp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

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
    

class OddsRatios:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def binary_features(self, variables):
        binary_odds_ratios = {}
        for i in variables:
            positive_df = self.df[self.df[i] == 1]
            negative_df = self.df[self.df[i] == 0]
            positive_and_target = positive_df[self.target].sum()
            positive_and_not_target = len(positive_df) - positive_and_target
            negative_and_target = negative_df[self.target].sum()
            negative_and_not_target = len(negative_df) - negative_and_target
            odds_ratio = (positive_and_target / positive_and_not_target) / (negative_and_target / negative_and_not_target)
            binary_odds_ratios[i] = odds_ratio
        return pd.DataFrame({'Odds Ratio': list(binary_odds_ratios.values())}, index=binary_odds_ratios.keys()).sort_values('Odds Ratio')

    def multi_category_features(self, variables):
        multi_category_odds_ratios = {}
        for i in variables:
            variable_odds_ratios = {}
            categories = set(self.df[i].values)
            for c in categories:
                positive_df = self.df[self.df[i] == c]
                negative_df = self.df[self.df[i] != c]
                positive_and_target = positive_df[self.target].sum()
                positive_and_not_target = len(positive_df) - positive_and_target
                negative_and_target = negative_df[self.target].sum()
                negative_and_not_target = len(negative_df) - negative_and_target
                odds_ratio = (positive_and_target / positive_and_not_target) / (negative_and_target / negative_and_not_target)
                variable_odds_ratios[c] = odds_ratio
            multi_category_odds_ratios[i] = variable_odds_ratios

        flattened = []
        for j,k in multi_category_odds_ratios.items():
            for l,m in k.items():
                flattened.append({'Variable':j, 'Category':l, 'Odds Ratio':m})
        return pd.DataFrame(flattened).set_index('Variable').sort_values('Odds Ratio')


class FeatureVisualisation:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def visualisations(self):
        info_summary = {
            "temporal": ["Line plots", "Heatmaps"],
            "ordinal": ["Bar plots", "Line plots"],
            "numerical_discrete": ["Boxplots", "Histograms"],
            "numerical_continuous_bounded": ["Density plots", "Boxplots"],
            "numerical_continuous_unbounded": ["Scatterplots", "Histograms"],
            "nominal_multi_category": ["Bar charts", "Stacked bar charts"],
            "nominal_binary": ["Bar charts"]
        }
        print(pd.Series(info_summary))
    
    def nominal_binary(self, variables):

        percent_fraud_total = self.df.fraud_bool.sum() / len(self.df) * 100
        for i in variables:
            
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x=self.target, hue=i, palette="viridis")
            plt.title(f"{self.target} vs {i}")
            plt.xlabel(self.target)
            plt.ylabel("Count")
            plt.legend(title=i, loc='upper right')
            plt.tight_layout()
            plt.show()

            percent_fraud_when_1 = self.df[self.df[i] == 1].fraud_bool.sum() / len(self.df[self.df[i] == 1].fraud_bool) * 100
            percent_fraud_when_0 = self.df[self.df[i] == 0].fraud_bool.sum() / len(self.df[self.df[i] == 0].fraud_bool) * 100

            plt.figure(figsize=(8, 6))
            plt.title(f'Percentage of fraud attempts by {i}')
            plt.bar([f'{i} == 1', f'{i} == 0'], [percent_fraud_when_1, percent_fraud_when_0], color='red')
            plt.axhline(y=percent_fraud_total, color='black', linestyle='--')
            plt.text(1, percent_fraud_total, 'overall fraud average', color='black', ha='center', va='bottom')
            plt.ylabel("Fraud %")
            plt.tight_layout()
            plt.show()
 
    def nominal_multi_category(self, variables):

        percent_fraud_total = self.df.fraud_bool.sum() / len(self.df) * 100
        for i in variables:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, hue=self.target, x=i)
            plt.title(f"Bar Chart of {i} by {self.target}")
            plt.legend(title=self.target, loc="upper right")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            crosstab = pd.crosstab(self.df[i], self.df[self.target], normalize="index")
            crosstab.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))
            plt.title(f"Stacked Bar Chart of {i}")
            plt.legend(title=self.target)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.title(f'Percentage of fraud attempts by category of {i}')
            category = []
            percentage = []
            for j in set(self.df[i].values):
                category.append(j)
                percentage.append(self.df[self.df[i] == j].fraud_bool.sum() / len(self.df[self.df[i] == j]) * 100)
            plt.bar(category, percentage, color='red')
            plt.axhline(y=percent_fraud_total, color='black', linestyle='--')
            plt.text(1, percent_fraud_total, 'overall fraud average', color='black', ha='center', va='bottom')
            plt.ylabel("Fraud %")
            plt.tight_layout()
            plt.show()

    def numerical_discrete(self, variables):
        for i in variables:

            # BOX PLOTS
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x=self.target, y=i)
            plt.title(f"Boxplot of {i} by {self.target}")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x=self.target, y=i, showfliers=False)
            plt.title(f"Boxplot of {i} by {self.target} (Outliers Ignored)")
            plt.show()

            # HISTOGRAM of percentage of 1s for each bin

            # Remove outliers similarly to boxplot above 
            Q1 = self.df[i].quantile(0.25)
            Q3 = self.df[i].quantile(0.75)
            IQR = Q3 - Q1
            min_ = Q1 - 1.5 * IQR
            max_ = Q3 + 1.5 * IQR
            df_no_outliers = self.df[(self.df[i] <= max_) & (self.df[i] >= min_)]
            data = df_no_outliers[i]  # feature of interest
            target = df_no_outliers[self.target]  # target variable

            # Bins for histogram
            bins = 10
            min_val = data.min()
            max_val = data.max()
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            # Calculate the histogram values (counts)
            hist, bin_edges = np.histogram(data, bins=bin_edges)

            # Calculate percentage of 1s for each bin
            percent_1s = []
            for j in range(len(bin_edges) - 1):
                bin_start = bin_edges[j]
                bin_end = bin_edges[j + 1]
                
                # Get the data points that fall within the current bin range
                bin_mask = (data >= bin_start) & (data < bin_end)
                
                # Calculate the percentage of 1s in the target for this bin
                target_in_bin = target[bin_mask]
                percentage_1s = (target_in_bin.sum() / len(target_in_bin)) * 100 if len(target_in_bin) > 0 else 0
                percent_1s.append(percentage_1s)

            # Create the figure and axis for a single histogram
            fig, ax_left = plt.subplots(figsize=(10, 6))

            # Plot the percentage of 1s as bars
            ax_left.bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, percent_1s, 
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, color='red', alpha=0.6)

            # Set labels and title
            ax_left.set_xlabel(i)
            ax_left.set_ylabel("Percentage of positive fraud counts")
            ax_left.set_title(f"Percentage of positive fraud counts by {i} in buckets of {bins}")

            # Add bin range labels on the x-axis (lower and rotated at 45 degrees)
            for j in range(len(bin_edges) - 1):
                bin_range = f"[{bin_edges[j]:.2f}, {bin_edges[j + 1]:.2f}]"
                ax_left.text(bin_edges[j] + (bin_edges[1] - bin_edges[0]) / 2, 
                            -max(percent_1s) * 0.1, bin_range, ha='center', va='top', rotation=45)

            plt.show()

    def numerical_continuous(self, variables):
        for i in variables:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=self.df, x=i, hue=self.target, fill=True, common_norm=False, alpha=0.5)
            plt.title(f"Density Plot of {i}")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x=self.target, y=i)
            plt.title(f"Boxplot of {i} by {self.target}")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x=self.target, y=i, showfliers=False)
            plt.title(f"Boxplot of {i} by {self.target} (Outliers Ignored)")
            plt.show()

            # HISTOGRAM of percentage of 1s for each bin

            # Remove outliers similarly to boxplot above 
            Q1 = self.df[i].quantile(0.25)
            Q3 = self.df[i].quantile(0.75)
            IQR = Q3 - Q1
            min_ = Q1 - 1.5 * IQR
            max_ = Q3 + 1.5 * IQR
            df_no_outliers = self.df[(self.df[i] <= max_) & (self.df[i] >= min_)]
            data = df_no_outliers[i]  # feature of interest
            target = df_no_outliers[self.target]  # target variable

            # Bins for histogram
            bins = 10
            min_val = data.min()
            max_val = data.max()
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            # Calculate the histogram values (counts)
            hist, bin_edges = np.histogram(data, bins=bin_edges)

            # Calculate percentage of 1s for each bin
            percent_1s = []
            for j in range(len(bin_edges) - 1):
                bin_start = bin_edges[j]
                bin_end = bin_edges[j + 1]
                
                # Get the data points that fall within the current bin range
                bin_mask = (data >= bin_start) & (data < bin_end)
                
                # Calculate the percentage of 1s in the target for this bin
                target_in_bin = target[bin_mask]
                percentage_1s = (target_in_bin.sum() / len(target_in_bin)) * 100 if len(target_in_bin) > 0 else 0
                percent_1s.append(percentage_1s)

            # Create the figure and axis for a single histogram
            fig, ax_left = plt.subplots(figsize=(10, 6))

            # Plot the percentage of 1s as bars
            ax_left.bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, percent_1s, 
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, color='red', alpha=0.6)

            # Set labels and title
            ax_left.set_xlabel(i)
            ax_left.set_ylabel("Percentage of positive fraud counts")
            ax_left.set_title(f"Percentage of positive fraud counts by {i} in buckets of {bins}")

            # Add bin range labels on the x-axis (lower and rotated at 45 degrees)
            for j in range(len(bin_edges) - 1):
                bin_range = f"[{bin_edges[j]:.2f}, {bin_edges[j + 1]:.2f}]"
                ax_left.text(bin_edges[j] + (bin_edges[1] - bin_edges[0]) / 2, 
                            -max(percent_1s) * 0.1, bin_range, ha='center', va='top', rotation=45)

            plt.show()
    
    def ordinal(self, variables):
        for i in variables:

            values = sorted(set(self.df[i].values))
            f_percentages = {}
            totals = {}
            for v in values:
                f_percentages[v] = self.df[self.df[i] == v][self.target].sum() / len(self.df[self.df[i] == v]) * 100
                totals[v] = len(self.df[self.df[i] == v])

            plt.figure(figsize=(10, 6))
            plt.bar([str(vr.round(3)) for vr in totals.keys()], totals.values())
            plt.title(f"Size of buckets of {i}")
            plt.xlabel('buckets')
            plt.ylabel('Count')
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.bar([str(vr.round(3)) for vr in f_percentages.keys()], f_percentages.values(), color='red')
            plt.title(f"Fraud percentage by bucket of {i}")
            plt.xlabel('buckets')
            plt.ylabel('Fraud Percentage (%)')
            plt.show()

