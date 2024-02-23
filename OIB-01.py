#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\Abhi\\documents\\readings')


# In[5]:


df=pd.read_csv('retail_sales_dataset.csv')


# In[6]:


df.head()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df['Age'].mean()


# In[13]:


df['Quantity'].median()


# In[15]:


df['Price per Unit'].std()


# In[17]:


df.describe()


# In[18]:


df['Quantity'].sum()


# In[24]:


df.dtypes


# In[50]:


df['Date']=pd.to_datetime(df['Date'],infer_datetime_format=True)


# # Exploratory Data Analysis

# In[44]:


df.plot(figsize=(5,4))


# In[29]:


product_cat_fr =df['Product Category'].value_counts()
product_cat_fr


# In[31]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
l=['Clothing','Electronics','Beauty']
s=[351,342,307]
plt.title('Frequency of Product Purchased')
plt.pie(s,labels=l,autopct='%0.1f%%')
plt.show()


# In[47]:


ax=sns.countplot(x='Quantity',data=df,color='b')


# In[39]:


ax=sns.countplot(x='Gender',data=df,color='y')

for bar in ax.containers:
    ax.bar_label(bar)


# In[43]:


dy=df.groupby(['Product Category'],as_index=False)['Quantity'].sum().sort_values(by='Product Category',ascending=True)
dy


# # Time Series Analysis

# In[40]:


# Assuming the dataset has a 'Date' column, convert it to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract year, quarter, and month from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df['Month'] = df['Date'].dt.month


# In[30]:


# Group by year, quarter, and month, and calculate the total amount of transactions
total_amount_by_year = df.groupby('Year')['Total Amount'].sum()
total_amount_by_quarter = df.groupby(['Year', 'Quarter'])['Total Amount'].sum()
total_amount_by_month = df.groupby(['Year', 'Month'])['Total Amount'].sum()


# In[31]:


# Print the total amount of transactions changed over years, quarters, and months
print("\nTotal Amount of Transactions Changed Over Years:")
print(total_amount_by_year)

print("\nTotal Amount of Transactions Changed Over Quarters:")
print(total_amount_by_quarter)

print("\nTotal Amount of Transactions Changed Over Months:")
print(total_amount_by_month)


# In[42]:


# Plot bar graphs for total amount of transactions changed over years, quarters, and months
plt.figure(figsize=(10,6))

# Bar graph for total amount of transactions changed over years
plt.subplot(1, 3, 1)
total_amount_by_year.plot(kind='bar', color='skyblue')
plt.title('Total Amount of Transactions Changed Over Quarters')
plt.xlabel('Qurter')
plt.ylabel('Total Amount')

# Bar graph for total amount of transactions changed over months
plt.subplot(1, 3, 3)
total_amount_by_month.plot(kind='bar', color='lightgreen')
plt.title('Total Amount of Transactions Changed Over Months')
plt.xlabel('Month')
plt.ylabel('Total Amount')

plt.tight_layout()
plt.show()


# In[15]:


df.index


# In[7]:


plt.figure(figsize=(20,5))
sns.lineplot(x='Date',y='Total Amount',data=df)
plt.show()


# In[21]:


price_stats = df['Price per Unit'].describe()
print("\nDescriptive Statistics of Price per Unit:")
print(price_stats)


# #  Plot a histogram of the price per unit to visualize its distribution

# In[23]:


plt.figure(figsize=(8,5))
plt.hist(df['Price per Unit'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Price per Unit')
plt.xlabel('Price per Unit')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[24]:


df=pd.read_csv('retail_sales_dataset.csv')


# #  Calculate the total revenue generated from sales

# In[26]:


df['Total Amount'] = df['Quantity'] * df['Price per Unit']
total_amount = df['Total Amount'].sum()
print("\nTotal Revenue Generated: $", total_amount)


# In[27]:


# Calculate the average price per unit over time
avg_price_over_time = df.groupby('Date')['Price per Unit'].mean()

# Print the average price per unit over time
print("\nAverage Price per Unit Over Time:")
print(avg_price_over_time)


# In[ ]:





# # THANK YOU
