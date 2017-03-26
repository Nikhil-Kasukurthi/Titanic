# Import libraries
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for all graphs
sns.set_style("dark")

# Read in the dataset, create dataframe
titanic_data = pd.read_csv('titanic_data.csv')

# In[309]:

# Print the first few records to review data and format
print(titanic_data.head())

# In[310]:

# Print the last few records to review data and format
print(titanic_data.tail())

titanic_data_duplicates = titanic_data.duplicated()
print('Number of duplicate entries is/are {}'.format(titanic_data_duplicates.sum()))

# In[315]:

# Let us just make sure this is working
duplicate_test = titanic_data.duplicated('Age').head()
print('Number of entries with duplicate age in top entires are {}'.format(duplicate_test.sum()))
titanic_data.head()

# #### Step 2 - Remove unnecessary columns
# Columns (PassengerId, Name, Ticket, Cabin, Fare, Embarked) removed

# In[316]:

# Create new dataset without unwanted columns
titanic_data_cleaned = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare', 'Embarked'], axis=1)
titanic_data_cleaned.head()

# #### Step 3 - Fix any missing or data format issues

# In[ ]:

# Calculate number of missing values
titanic_data_cleaned.isnull().sum()

# In[279]:

# Review some of the missing Age data
missing_age_bool = pd.isnull(titanic_data_cleaned['Age'])
titanic_data_cleaned[missing_age_bool].head()

# In[317]:

# Determine number of males and females with missing age values
missing_age_female = titanic_data_cleaned[missing_age_bool]['Sex'] == 'female'
missing_age_male = titanic_data_cleaned[missing_age_bool]['Sex'] == 'male'

print('Number for females and males with age missing are {} and {} respectively'.format(
    missing_age_female.sum(), missing_age_male.sum()))

# In[318]:

# Taking a look at the datatypes
titanic_data_cleaned.info()

# Missing Age data will affect **Q2 - Did age, regardless of sex, determine your chances of survival?** But graphing and summations shouldn't be a problem since they will be treated as zero(0) value. However, 177 is roughly 20% of our 891 sample dataset which seems like a lot to discount. Also, this needs to be accounted for if reviewing descriptive stats such as mean age.
#
# Should keep note of the proportions across male and female...
#
# - Age missing in male data: **124**
# - Age missing in female data: **53**

# ## Data Exploration and Visualization

# In[282]:

# Looking at some typical descriptive statistics
titanic_data_cleaned.describe()

# In[283]:

# Age min at 0.42 looks a bit weird so give a closer look
titanic_data_cleaned[titanic_data_cleaned['Age'] < 1]

# In[284]:

# Taking a look at some survival rates for babies
youngest_to_survive = titanic_data_cleaned[titanic_data_cleaned['Survived'] == True]['Age'].min()
youngest_to_die = titanic_data_cleaned[titanic_data_cleaned['Survived'] == False]['Age'].min()
oldest_to_survive = titanic_data_cleaned[titanic_data_cleaned['Survived'] == True]['Age'].max()
oldest_to_die = titanic_data_cleaned[titanic_data_cleaned['Survived'] == False]['Age'].max()

print('Youngest to survive: {} \nYoungest to die: {} \nOldest to survive: {} \nOldest to die: {}'.format(
    youngest_to_survive, youngest_to_die, oldest_to_survive, oldest_to_die))


# Data description states that Age can be fractional - *Age is in Years; Fractional if Age less than One (1) If the Age is Estimated, it is in the form xx.5* - Therefore, 0.42 appears to be expected and normal data
#
# **Note:** An interesting note is that all "new borns" survived. Potential **Q6** - At what age did children's survival rate match that of adults, if ever.
#
# Other notable stats
# - Oldest to survive: **80**
# - Oldest to die: **74**
# - Youngest to survive: **< 1 (0.42)**
# - Youngest to die: **1**

# ## Question 1
# Were social-economic standing a factor in survival rate?

# In[285]:

# Returns survival rate/percentage of sex and class
def survival_rate(pclass, sex):
    """
    Args:
        pclass: class value 1,2 or 3
        sex: male or female
    Returns:
        survival rate as percentage.
    """
    grouped_by_total = titanic_data_cleaned.groupby(['Pclass', 'Sex']).size()[pclass, sex].astype('float')
    grouped_by_survived_sex = titanic_data_cleaned.groupby(['Pclass', 'Survived', 'Sex']).size()[pclass, 1, sex].astype(
        'float')
    survived_sex_pct = (grouped_by_survived_sex / grouped_by_total * 100).round(2)

    return survived_sex_pct


# In[286]:

# Get the actual numbers grouped by class, suvival and sex
groupedby_class_survived_size = titanic_data_cleaned.groupby(['Pclass', 'Survived', 'Sex']).size()

# Print - Grouped by class, survival and sex
print(groupedby_class_survived_size)
print('Class 1 - female survival rate: {}%'.format(survival_rate(1, 'female')))
print('Class 1 - male survival rate: {}%'.format(survival_rate(1, 'male')))
print('-----')
print('Class 2 - female survival rate: {}%'.format(survival_rate(2, 'female')))
print('Class 2 - male survival rate: {}%'.format(survival_rate(2, 'male')))
print('-----')
print('Class 3 - female survival rate: {}%'.format(survival_rate(3, 'female')))
print('Class 3 - male survival rate: {}%'.format(survival_rate(3, 'male')))

# Graph - Grouped by class, survival and sex
g = sns.factorplot(x="Sex", y="Survived", col="Pclass", data=titanic_data_cleaned,
                   saturation=.5, kind="bar", ci=None, size=5, aspect=.8)

# Fix up the labels
(g.set_axis_labels('', 'Survival Rate')
 .set_xticklabels(["Men", "Women"])
 .set_titles("Class {col_name}")
 .set(ylim=(0, 1))
 .despine(left=True, bottom=True))

# Graph - Actual count of passengers by survival, group and sex
g = sns.factorplot('Survived', col='Sex', hue='Pclass', data=titanic_data_cleaned, kind='count', size=7, aspect=.8)

# Fix up the labels
(g.set_axis_labels('Suvivors', 'No. of Passengers')
 .set_xticklabels(["False", "True"])
 .set_titles('{col_name}')
 )
sns.plt.show()

titles = ['Men', 'Women']
for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)

# Based on the raw numbers it would appear as though passengers in Class 3 had a similar survival rate as those from Class 1 with **119 and 136 passengers surviving respectively.** However, looking at the percentages of the overall passengers per class and the total numbers across each class, it can be assumed that **a passenger from Class 1 is about 2.5x times more likely to survive than a passenger in Class 3.**
#
# Social-economic standing was a factor in survival rate of passengers.
#
# - Class 1: **62.96%**
# - Class 2: **47.28%**
# - Class 3: **24.24%**

# ## Question 2
# Did age, regardless of sex and class, determine your chances of survival?

# In[287]:

# Let us first identify and get rid of records with missing Age
print('Number of men and woman with age missing are {} and {} respectively'.format(
    missing_age_female.sum(), missing_age_male.sum()))

# Drop the NaN values. Calculations will be okay with them (seen as zero) but will throw off averages and counts
titanic_data_age_cleaned = titanic_data_cleaned.dropna()

# Find total count of survivors and those who didn't
number_survived = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == True]['Survived'].count()
number_died = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == False]['Survived'].count()

# Find average of survivors and those who didn't
mean_age_survived = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == True]['Age'].mean()
mean_age_died = titanic_data_age_cleaned[titanic_data_age_cleaned['Survived'] == False]['Age'].mean()

# Display a few raw totals
print(
    'Total number of survivors {} \nTotal number of non survivors {} \nMean age of survivors {} \nMean age of non survivors {} \nOldest to survive {} \nOldest to not survive {}'.format(
        number_survived, number_died, np.round(mean_age_survived),
        np.round(mean_age_died), oldest_to_survive, oldest_to_die))

# Graph - Age of passengers across sex of those who survived
g = sns.factorplot(x="Survived", y="Age", hue='Sex', data=titanic_data_age_cleaned, kind="box", size=7, aspect=.8)

# Fix up the labels
(g.set_axis_labels('Suvivors', 'Age of Passengers')
 .set_xticklabels(["False", "True"])
 )

# Based on the above boxplot and calculated data, it would appear that:
# - Regardless of sex and class, **age was not** a deciding factor in the passenger survival rate
# - Average age for those who survived and even those who did not survive were inline with eachother

# In[288]:

# Create Cateogry column and categorize people
titanic_data_age_cleaned.loc[
    ((titanic_data_age_cleaned['Sex'] == 'female') &
     (titanic_data_age_cleaned['Age'] >= 18)),
    'Category'] = 'Woman'

titanic_data_age_cleaned.loc[
    ((titanic_data_age_cleaned['Sex'] == 'male') &
     (titanic_data_age_cleaned['Age'] >= 18)),
    'Category'] = 'Man'

titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Age'] < 18),
    'Category'] = 'Child'

# Get the totals grouped by Men, Women and Children, and by survival
print(titanic_data_age_cleaned.groupby(['Category', 'Survived']).size())

# Graph - Compare survival count between Men, Women and Children
g = sns.factorplot('Survived', col='Category', data=titanic_data_age_cleaned, kind='count', size=7, aspect=.8)

# Fix up the labels
(g.set_axis_labels('Suvivors', 'No. of Passengers')
 .set_xticklabels(['False', 'True'])
 )

titles = ['Men', 'Women', 'Children']
for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)

# The data, and more so, the graphs tends to support the idea that "women and children first" possibly played a role in the survival of a number of people. It's a bit surprising that more children didn't survive but this could possibly be attributed to the mis-representation of what age is considered as the cut off for adults - i.e. if in the 1900's someone 15-17 were considered adults, they would not have been "saved" under the "women and children first" idea and would be made to fend for themselves. That would in turn, change the outcome of the above data and possible increase the number of children who survived.

# ## Question 4
# Did women with children have a better survival rate vs women without children (adults 18+)?
# - "Women with children" is referring to parents only

# In[289]:

# Determine number of woman that are not parents
titanic_data_woman_parents = titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Woman') &
    (titanic_data_age_cleaned['Parch'] > 0)]

# Determine number of woman over 20 that are not parents
titanic_data_woman_parents_maybe = titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Woman') &
    (titanic_data_age_cleaned['Parch'] > 0) &
    (titanic_data_age_cleaned['Age'] > 20)]

# In[290]:

titanic_data_woman_parents.head()

# In[291]:

titanic_data_woman_parents_maybe.head()

# After reviewing the data, and giving it a bit more thought, I noticed a issue which I didn't think of before i.e **A woman with Age: 23 and Parch: 2 could be onboard with her children OR onboard with her parents.** Based on the 'Parch' definition provided in the data description, *Parch - number of parents or children on board*, I don't believe it's possible to accurately determine women with children (parents) vs women with their parents onboard.

# ## Question 5
# How did children with nannies fare in comparison to children with parents. Did the nanny "abandon" children to save his/her own life?
#  - Need to review list for children with no parents. These will be children with nannies as stated in the data description
#  - Compare "normal" survival rate of children with parents against children with nannies
#
# Assumptions:
# 1. If you're classified as a 'Child' (under 18) and have Parch > 0, then the value is associated to your Parents,  eventhough it is possible to be under 18 and also have children
# 2. Classifying people as 'Child' represented by those under 18 years old is applying today's standards to the 1900 century

# In[292]:

# Separate out children with parents from those with nannies
titanic_data_children_nannies = titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Child') &
    (titanic_data_age_cleaned['Parch'] == 0)]

titanic_data_children_parents = titanic_data_age_cleaned.loc[
    (titanic_data_age_cleaned['Category'] == 'Child') &
    (titanic_data_age_cleaned['Parch'] > 0)]

# In[307]:

# Determine children with nannies who survived and who did not
survived_children_nannies = titanic_data_children_nannies.Survived.sum()
total_children_nannies = titanic_data_children_nannies.Survived.count()
pct_survived_nannies = ((float(survived_children_nannies) / total_children_nannies) * 100)
pct_survived_nannies = np.round(pct_survived_nannies, 2)
survived_children_nannies_avg_age = np.round(titanic_data_children_nannies.Age.mean())

# Display results
print(
    'Total number of children with nannies: {}\nChildren with nannies who survived: {}\nChildren with nannies who did not survive: {}\nPercentage of children who survived: {}%\nAverage age of surviving children: {}'.format(
        total_children_nannies, survived_children_nannies,
        total_children_nannies - survived_children_nannies, pct_survived_nannies, survived_children_nannies_avg_age))

# Verify counts (looked a bit too evenly divided)
titanic_data_children_nannies.loc[titanic_data_children_nannies['Survived'] == 1]

# In[304]:

# Determine children with parents who survived and who did not
survived_children_parents = titanic_data_children_parents.Survived.sum()
total_children_parents = titanic_data_children_parents.Survived.count()
pct_survived_parents = ((float(survived_children_parents) / total_children_parents) * 100)
pct_survived_parents = np.round(pct_survived_parents, 2)
survived_children_parents_avg_age = np.round(titanic_data_children_parents.Age.mean())

# Display results
print(
    'Total number of children with parents: {}\nChildren with parents who survived: {}\nChildren with parents who did not survive: {}\nPercentage of children who survived: {}%\nAverage age of surviving children: {}'.format(
        total_children_parents, survived_children_parents,
        total_children_parents - survived_children_parents, pct_survived_parents, survived_children_parents_avg_age))
