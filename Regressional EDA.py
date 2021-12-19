# %% [markdown]
# # Preparing the "workspace"

# %%
pip install matplotlib

# %%
pip install pandas

# %%
pip install statsmodels

# %%
# Packages and data loading

import matplotlib.pyplot as pyplot
import pandas as pandas
import statsmodels.api as statsmodels

mdata = pandas.read_csv('./Marketing-Analysis-CUSTOMERS/dataset/Marketing-Customer-Value-Analysis.csv')

# %%
# Data size overview

mdata.shape

# %%
# Data overview

mdata.head()

# %% [markdown]
# ### Target
# 
# The desired output for the costumer engagement is the "Response" column, which isn't bumeric. <b>Logistic regression models "prefers"" numbers as values, thus we'll be applying a conversion</b>

# %%
# Converting the target/output variable into a numerical

mdata['Engaged'] = mdata['Response'].apply(lambda x: 0 if x == 'No' else 1)

# %% [markdown]
# ## Examining Engagement Rate
# 
# Which is the percentage of customers that were exposed to the marketing

# %%
engagement_rate_data = pandas.DataFrame(mdata.groupby('Engaged').count()['Response'] / mdata.shape[0] * 100.0)

engagement_rate_data

# %% [markdown]
# There are more customers that did not engage with the marketing.

# %%
engagement_rate_data.plot(kind = 'pie', figsize =(15, 7), startangle = 90, subplots = True, autopct = lambda x: '%0.1f%%' % x)

pyplot.show()

# %% [markdown]
# ## Total Claim Amounts

# %%
ax = mdata[['Engaged','Total Claim Amount']].boxplot(by = 'Engaged', showfliers = False, figsize = (7,5))

ax.set_xlabel('Engaged')
ax.set_ylabel('Total Claim Amount')
ax.set_title('Total Claim Amount Distributions by Engagements')

pyplot.suptitle("")
pyplot.show()

# %% [markdown]
# ### Observation and Notes
# 
# Box plots are good method to view the distributions of continuous variables. 
# 
# The rectangles represent the first quartile to the third quartile, and the green line represent the median. 
# 
# The ends are the minimum and maximum values. 
# 
# The showfliers = false; allows to spot the suspected outliers like so

# %%
ax = mdata[['Engaged','Total Claim Amount']].boxplot(by = 'Engaged', showfliers = True, figsize = (7,5))

ax.set_xlabel('Engaged')
ax.set_ylabel('Total Claim Amount')
ax.set_title('Total Claim Amount Distributions by Engagements')

pyplot.suptitle("")
pyplot.show()

# %% [markdown]
# ### Notes
# 
# The dots are the suspected outliers based on Interquartile range (IQR). 
# The formula for suspected outliers are 
# 
#     - 1.5IQR above the third quartile or,
#     - 1.5IQR below the first quartile.

# %% [markdown]
# ## Regression analysis

# %%
# Examining the feature variables to see, 
# which fits in the logistic regression model.

mdata.dtypes


# %%
continuous_vars = ['Customer Lifetime Value', 'Income', 'Monthly Premium Auto','Months Since Last Claim', 'Months Since Policy Inception', 'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount']

# %%
# Transforming categorical variables 
# into numericals through factorizing

gender_values, gender_labels = mdata['Gender'].factorize()
print(gender_values)
print(gender_labels)

# %% [markdown]
# ### Notes
# 
# In factorization, the variables were turned into 1 or 0’s. But what if order matters? Then applying the Categorical function is also possible.

# %%
categories = pandas.Categorical(mdata['Education'], categories=['High school or Below', 'Bachelor', 'College', 'Master', 'Doctor'])

# %% [markdown]
# ### Notes
# 
# 0, 1, 2, 3, and 4 apply for the education of the High School or Below, Bachelor, College, Master, and Doctor respectively. 
# This will allow fitting the data into a logistic model.

# %%
mdata['GenderFactorized'] = gender_values
mdata['EducationFactorized'] = categories.codes

# %%
# Combination of Categorical & Continuous

logit = statsmodels.Logit(mdata['Engaged'], mdata[['Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints','Number of Policies','Total Claim Amount','GenderFactorized','EducationFactorized']])

logit_fit = logit.fit()

logit_fit.summary()

# %% [markdown]
# ### Notes
# 
# - z (short for z-score) is the number of standard deviations from the mean [3].
# 
# - The P>|z| (meaning p-value) states how likely to observe the relationship by chance. Normally, 0.05 is a standard cut-off for the p-value, and values less than 0.05 means lesser the chances of this relationship between input and the output variable to occur by coincidence. For example in numerical variables, we can see that Income, Monthly Premium Auto, Months Since Last Claim, Months Since Policy Inception, and Number of Policies variables have significant relationships with Engagement (output variable). If we look at the Months Since Last Claim variable, it is significant (very low p-value) and is negatively correlated (big negative z-score) with engagement. 
# 
# In other words, as more months pass after a claim, customers are less likely to engage with marketing.
# 
# From the categorical variables, we can see that males (0’s) are less likely to engage with marketing and the same applies to lower education (0 was for high school and went up 4 for doctorate).

# %% [markdown]
# ## Conclusion
# 
# Engagement rates were in tabular form, sales channels were in pie charts for easier interpretation, total claim amounts in box plots to see ranges and potential outliers, and regression analysis were to find strong trends. 
# 
# Instead of using logistic regression as a predictive model in the output, it was leveraged to isolate trends then prepared as a potential feed for another machine learning model.

# %% [markdown]
# 