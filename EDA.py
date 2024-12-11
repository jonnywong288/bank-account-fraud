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

    def info(self):
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
    def __init__(self, df, target, missing_values):
        self.df = df
        self.target = target
        self.missing_values = missing_values

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

        plt.bar(binary_odds_ratios.keys(), binary_odds_ratios.values())
        plt.title('Odds Ratios for Positive Fraud in Each Binary Category')
        plt.xlabel('Binary Category')
        plt.ylabel('Odds Ratio')
        plt.xticks(rotation=90)
        plt.axhline(y=1, color='black', linestyle='--')
        plt.show()

        return pd.DataFrame({'Odds Ratio': list(binary_odds_ratios.values())}, index=binary_odds_ratios.keys()).sort_values('Odds Ratio')

    def multi_category_features(self, variables):
        multi_category_odds_ratios = {}
        for i in variables:
            variable_odds_ratios = {}
            categories = sorted(list(set(self.df[i].values)))
            for c in categories:
                positive_df = self.df[self.df[i] == c]
                negative_df = self.df[self.df[i] != c]
                positive_and_target = positive_df[self.target].sum()
                positive_and_not_target = len(positive_df) - positive_and_target
                negative_and_target = negative_df[self.target].sum()
                negative_and_not_target = len(negative_df) - negative_and_target
                odds_ratio = (positive_and_target / positive_and_not_target) / (negative_and_target / negative_and_not_target)
                variable_odds_ratios[str(c)] = odds_ratio
            multi_category_odds_ratios[i] = variable_odds_ratios

        n_vars = len(variables)
        cols = 3
        rows = int(np.ceil(n_vars/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Adjust figure size for readability
        fig.suptitle("Odds Ratios for Categories of Multi-Category Variables", fontsize=16, y=1.02)  # Adjust y to prevent overlap
        axes = axes.flatten()  

        for idx, v in enumerate(multi_category_odds_ratios.keys()):
            ax = axes[idx]
            ax.bar(multi_category_odds_ratios[v].keys(), multi_category_odds_ratios[v].values())
            ax.set_title(f'{v} odds ratios by category')
            ax.set_xlabel('Category')
            ax.set_ylabel('Odds Ratio')
            ax.tick_params(axis='x', rotation=45) 
            ax.axhline(y=1, color='black', linestyle='--')


        
        # Hide empty subplots
        for ax in axes[n_vars:]:
            ax.axis('off')
        
        plt.tight_layout()  
        plt.show()


        flattened = []
        for j,k in multi_category_odds_ratios.items():
            for l,m in k.items():
                flattened.append({'Variable':j, 'Category':l, 'Odds Ratio':m})

        return pd.DataFrame(flattened).set_index('Variable').sort_values(['Variable', 'Odds Ratio'])
    
    def numerical_features(self, variables):
        all_odds_ratios = []
        n_vars = len(variables)
        
        cols = 3
        rows = int(np.ceil(n_vars / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Adjust figure size for readability
        fig.suptitle("Odds Ratios for Quantiles of Numerical Variables", fontsize=16, y=1.02)  # Adjust y to prevent overlap
        axes = axes.flatten()  
        
        for idx, v in enumerate(variables):
            quantile_odds = {}
            if v in self.missing_values:
                df_no_nulls = self.df[self.df[v] != -1]
                quantiles = [df_no_nulls[v].quantile(np.round(i, 1)) for i in np.linspace(0, 1, 11)]

                for i in range(len(quantiles))[:-1]:
                    quantile = df_no_nulls[(df_no_nulls[v] >= quantiles[i]) & (df_no_nulls[v] < quantiles[i+1])]
                    not_quantile = df_no_nulls[(df_no_nulls[v] < quantiles[i]) | (df_no_nulls[v] >= quantiles[i+1])]

                    quantile_and_target = quantile[self.target].sum()
                    quantile_and_not_target = len(quantile) - quantile_and_target
                    not_quantile_and_target = not_quantile[self.target].sum()
                    not_quantile_and_not_target = len(not_quantile) - not_quantile_and_target

                    odds_ratio = (quantile_and_target / quantile_and_not_target) / (not_quantile_and_target / not_quantile_and_not_target)
                    quantile_odds[f'{i+1}/10'] = odds_ratio
                    
            else:
                quantiles = [self.df[v].quantile(np.round(i, 1)) for i in np.linspace(0, 1, 11)]
            
                for i in range(len(quantiles))[:-1]:
                    quantile = self.df[(self.df[v] >= quantiles[i]) & (self.df[v] < quantiles[i+1])]
                    not_quantile = self.df[(self.df[v] < quantiles[i]) | (self.df[v] >= quantiles[i+1])]

                    quantile_and_target = quantile[self.target].sum()
                    quantile_and_not_target = len(quantile) - quantile_and_target
                    not_quantile_and_target = not_quantile[self.target].sum()
                    not_quantile_and_not_target = len(not_quantile) - not_quantile_and_target

                    odds_ratio = (quantile_and_target / quantile_and_not_target) / (not_quantile_and_target / not_quantile_and_not_target)
                    quantile_odds[f'{i+1}/10'] = odds_ratio
            
            quantile_odds_df = pd.DataFrame({
                'Quantile': quantile_odds.keys(),
                f'Odds Ratio: {v}': quantile_odds.values()
            }).set_index('Quantile')
            
            # Plot subplot
            ax = axes[idx]
            ax.plot(quantile_odds_df.index, quantile_odds_df[f'Odds Ratio: {v}'])
            ax.set_title(f'{v} odds ratios by quantile')
            ax.set_xlabel('Quantile')
            ax.set_ylabel('Odds Ratio')
            ax.axhline(y=1, color='black', linestyle='--')

            
            all_odds_ratios.append(quantile_odds_df)
        
        # Hide empty subplots
        for ax in axes[n_vars:]:
            ax.axis('off')
        
        plt.tight_layout()  
        plt.show()
        
        return pd.concat(all_odds_ratios, axis=1)

class FeatureVisualisation:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def info(self):
        info_summary = {
            "ordinal": ["Bar charts"],
            "numerical": ["Density plots", "Boxplots", "Bar charts"],
            "nominal_multi_category": ["Bar charts", "Stacked bar charts"],
            "nominal_binary": ["Bar charts"],
            "missing value analysis": ["Bar charts"]
        }
        print(pd.Series(info_summary))
    
    def nominal_binary(self, variables):

        percent_fraud_total = self.df.fraud_bool.sum() / len(self.df) * 100
        for i in variables:

            # Create subplots with 1 row and 2 columns
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"{i}", fontsize=16, y=1.02)  # Adjust y to prevent overlap

            
            sns.countplot(data=self.df, x=self.target, hue=i, palette="viridis", ax=axes[0])
            axes[0].set_title(f"{self.target} vs {i}")
            axes[0].set_xlabel(self.target)
            axes[0].set_ylabel("Count")
            axes[0].legend(title=i, loc='upper right')
            

            percent_fraud_when_1 = self.df[self.df[i] == 1].fraud_bool.sum() / len(self.df[self.df[i] == 1].fraud_bool) * 100
            percent_fraud_when_0 = self.df[self.df[i] == 0].fraud_bool.sum() / len(self.df[self.df[i] == 0].fraud_bool) * 100

            axes[1].set_title(f'Percentage of fraud attempts by {i}')
            axes[1].bar([f'{i} == 1', f'{i} == 0'], [percent_fraud_when_1, percent_fraud_when_0], color='red')
            axes[1].axhline(y=percent_fraud_total, color='black', linestyle='--')
            axes[1].text(1, percent_fraud_total, 'overall fraud average', color='black', ha='center', va='bottom')
            axes[1].set_ylabel("Fraud %")
            
            
            plt.tight_layout()
            plt.show()
 
    def nominal_multi_category(self, variables):
        percent_fraud_total = self.df.fraud_bool.sum() / len(self.df) * 100
        for i in variables:
            # Create subplots with 1 row and 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            fig.suptitle(f"{i}", fontsize=16, y=1.02)  # Adjust y to prevent overlap
            
            # Create a temporary column with string conversion
            temp_df = self.df.assign(temp_column=self.df[i].astype(str))
            
            # Check if the variable is numerical and sort only if it is
            if pd.api.types.is_numeric_dtype(self.df[i]):
                sorted_order = sorted(temp_df['temp_column'].unique(), key=lambda x: float(x))
            else:
                sorted_order = sorted(temp_df['temp_column'].unique())
            
            # Plot the first chart (countplot)
            sns.countplot(data=temp_df, hue=self.target, x="temp_column", order=sorted_order, ax=axes[0])
            axes[0].set_title(f"Bar Chart of {i} by {self.target}")
            axes[0].legend(title=self.target, loc="upper right")
            
            # Plot the second chart (stacked bar chart of crosstab)
            crosstab = pd.crosstab(self.df[i], self.df[self.target], normalize="index")
            crosstab.plot(kind="bar", stacked=True, colormap="viridis", ax=axes[1])
            axes[1].set_title(f"Stacked Bar Chart of {i}")
            axes[1].legend(title=self.target)
            
            # Plot the third chart (percentage of fraud by category)
            axes[2].set_title(f'Percentage of fraud attempts by category of {i}')
            category = []
            percentage = []
            for j in sorted(list(set(self.df[i].values))):
                category.append(str(j))
                percentage.append(self.df[self.df[i] == j].fraud_bool.sum() / len(self.df[self.df[i] == j]) * 100)
            axes[2].bar(category, percentage, color='red')
            axes[2].axhline(y=percent_fraud_total, color='black', linestyle='--')
            axes[2].text(1, percent_fraud_total, 'overall fraud average', color='black', ha='center', va='bottom')
            axes[2].set_ylabel("Fraud %")
            
            # Ensure the layout is tight and title doesn't overlap
            plt.tight_layout()
            plt.show()


    def numerical(self, variables):
        for i in variables:
            # Create a row of 4 subplots
            fig, axes = plt.subplots(1, 4, figsize=(32, 6))  # 4 columns, adjust size for clarity
            fig.suptitle(f"{i}", fontsize=16, y=1.02)  # Adjust y to prevent overlap

            # Density Plot
            sns.kdeplot(data=self.df, x=i, hue=self.target, fill=True, common_norm=False, alpha=0.5, ax=axes[0])
            axes[0].set_title(f"Density Plot of {i}")
            
            # Boxplot
            sns.boxplot(data=self.df, x=self.target, y=i, ax=axes[1])
            axes[1].set_title(f"Boxplot of {i} by {self.target}")
            
            # Boxplot without outliers
            sns.boxplot(data=self.df, x=self.target, y=i, showfliers=False, ax=axes[2])
            axes[2].set_title(f"Boxplot of {i} by {self.target} (Outliers Ignored)")
            
            # Histogram of percentage of 1s
            # Remove outliers
            Q1 = self.df[i].quantile(0.25)
            Q3 = self.df[i].quantile(0.75)
            IQR = Q3 - Q1
            min_ = Q1 - 1.5 * IQR
            max_ = Q3 + 1.5 * IQR
            df_no_outliers = self.df[(self.df[i] <= max_) & (self.df[i] >= min_)]
            data = df_no_outliers[i]
            target = df_no_outliers[self.target]
            
            bins = 10
            bin_edges = np.linspace(data.min(), data.max(), bins + 1)
            percent_1s = []
            
            for j in range(len(bin_edges) - 1):
                bin_mask = (data >= bin_edges[j]) & (data < bin_edges[j + 1])
                target_in_bin = target[bin_mask]
                percentage_1s = (target_in_bin.sum() / len(target_in_bin)) * 100 if len(target_in_bin) > 0 else 0
                percent_1s.append(percentage_1s)
            
            axes[3].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, percent_1s,
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, color='red', alpha=0.6)
            axes[3].set_title(f"Percentage of Fraud by {i}")
            axes[3].set_xlabel(f"{i} Bins")
            axes[3].set_ylabel("Fraud Percentage (%)")

            # Add bin range labels rotated at 45 degrees
            for j in range(len(bin_edges) - 1):
                bin_range = f"[{bin_edges[j]:.2f}, {bin_edges[j + 1]:.2f}]"
                axes[3].text(
                    bin_edges[j] + (bin_edges[1] - bin_edges[0]) / 2,  # Center of the bar
                    -max(percent_1s) * 0.15,                           # Position below the x-axis
                    bin_range,                                         # Label text
                    ha='center', va='top', rotation=45                 # Center-align and rotate
                )

            
            # Adjust layout for better visualization
            plt.tight_layout()
            plt.show()


    def ordinal(self, variables):
        for i in variables:
            values = sorted(set(self.df[i].values))
            f_percentages = {}
            totals = {}
            for v in values:
                f_percentages[v] = self.df[self.df[i] == v][self.target].sum() / len(self.df[self.df[i] == v]) * 100
                totals[v] = len(self.df[self.df[i] == v])
            
            # Create subplots with 1 row and 2 columns
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"{i}", fontsize=16, y=1.02)  # Adjust y to prevent overlap

            
            # Plot the size of buckets on the first subplot
            axes[0].bar([str(vr.round(3)) for vr in totals.keys()], totals.values())
            axes[0].set_title(f"Size of buckets of {i}")
            axes[0].set_xlabel('Buckets')
            axes[0].set_ylabel('Count')
            
            # Plot the fraud percentages on the second subplot
            axes[1].bar([str(vr.round(3)) for vr in f_percentages.keys()], f_percentages.values(), color='red')
            axes[1].set_title(f"Fraud percentage by bucket of {i}")
            axes[1].set_xlabel('Buckets')
            axes[1].set_ylabel('Fraud Percentage (%)')
            
            # Adjust layout and show the plots
            plt.tight_layout()
            plt.show()

    def missing_values(self, variables, missing_value_type):
        percent_fraud_total = self.df.fraud_bool.sum() / len(self.df) * 100
        if pd.isna(missing_value_type):
            for i in variables:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"{i}", fontsize=16, y=1.02)  # Adjust y to prevent overlap

                value_counts = {
                    'Missing Values': len(self.df[self.df[i].isna()]),
                    'Provided Values': len(self.df[self.df[i].notna()])
                }
                axes[0].bar(value_counts.keys(), value_counts.values())
                axes[0].set_title(f'Volume of missing values of {i} in data set')

                target_value_counts = {
                    'Missing Values': self.df[self.df[i].isna()][self.target].sum(),
                    'Provided Values': self.df[self.df[i].notna()][self.target].sum()
                }
                axes[1].bar(target_value_counts.keys(), target_value_counts.values())
                axes[1].set_title('Comparison of fraud percentage between missing and provided values')

                # Adjust layout and show the plots
                plt.tight_layout()
                plt.show()
        else:
            for i in variables:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"{i}", fontsize=16, y=1.02)  # Adjust y to prevent overlap

                value_counts = {
                    'Missing Values': len(self.df[self.df[i] == missing_value_type]),
                    'Provided Values': len(self.df[self.df[i] != missing_value_type])
                }
                axes[0].bar(value_counts.keys(), value_counts.values())
                axes[0].set_title(f'Volume of missing values of {i} in data set')

                target_value_counts = {
                    'Missing Values': self.df[self.df[i] == missing_value_type][self.target].sum() / value_counts['Missing Values'] * 100,
                    'Provided Values': self.df[self.df[i] != missing_value_type][self.target].sum() / value_counts['Provided Values'] * 100
                }
                axes[1].bar(target_value_counts.keys(), target_value_counts.values(), color='red')
                axes[1].set_title('Comparison of fraud percentage between missing and provided values')
                axes[1].axhline(y=percent_fraud_total, color='black', linestyle='--')
                axes[1].text(1, percent_fraud_total, 'overall fraud average', color='black', ha='center', va='bottom')
            
                # Adjust layout and show the plots
                plt.tight_layout()
                plt.show()



