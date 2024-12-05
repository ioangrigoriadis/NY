#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('NYC-Airbnb-2023.csv')


# # Show all columns and rows

# In[3]:


#Show all columns and rows
pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)


# In[4]:


df.head(3)


# In[5]:


df.info()


# # Missing Values

# In[6]:


#df.isnull().sum()
##Fill missing values with mean, median, or mode:
#df['column'].fillna(df['column'].mean(), inplace=True)
##Remove rows/columns with missing values
#df.dropna(inplace=True)
## Remove only specific columns with missing values
#df = df[df['column'].notna()]

# Drop 'licence'
df.drop(columns=['license'],inplace=True)
df


# # Outliers

# In[7]:


#Detect outliers using statistical methods
#from scipy import stats
#from scipy.stats import zscore

## Select only numeric columns
#numeric_df = df.select_dtypes(include=[np.number])
#numeric_df.columns
## Choose the columns that you want to exlcude from outliers
## for example price, minimum_nights
#df_out = df[np.abs((df['price'] - df['price'].mean()) / df['price'].std())<=3]
#df_out = df_out[np.abs((df_out['minimum_nights'] - df_out['minimum_nights'].mean()) / df_out['minimum_nights'].std())<=3]
#df_out
#print('Number of outliers:{}'.format(len(df)-len(df_out)))
##           OR
##Interquartile Range (IQR) Method:
#Q1 = df['price'].quantile(0.25)
#Q3 = df['price'].quantile(0.75)
#IQR = Q3 - Q1
#df_out = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
#df_out=df_out[(df_out['minimum_nights'] >= Q1 - 1.5 * IQR) & (df_out['minimum_nights'] <= Q3 + 1.5 * IQR)]
#df_out
#print('Number of outliers:{}'.format(len(df)-len(df_out)))
##          OR
## Using Percentile-Based Capping
## Define lower and upper bounds based on percentiles
#lower_bound_price = df['price'].quantile(0.01)
#upper_bound_price = df['price'].quantile(0.99)
#lower_bound_min_nights = df['minimum_nights'].quantile(0.01)
#upper_bound_min_nights = df['minimum_nights'].quantile(0.99)
# Filter out outliers
#df_out = df[(df['price'] >= lower_bound_price) & (df['price'] <= upper_bound_price)]
#df_out = df_out[(df_out['minimum_nights'] >= lower_bound_min_nights) & (df_out['minimum_nights'] <= upper_bound_min_nights)]
#df_out
#print('Number of outliers:{}'.format(len(df)-len(df_out)))


# In[8]:


# Remove price outliers
#df = df[np.abs((df['price'] - df['price'].mean()) / df['price'].std())<=3]


# In[ ]:





# # Data Type Conversion

# In[9]:


## Convert a column to float
#df['column'] = df['column'].astype(float)  # Convert to float


# # Remove Duplicates

# In[10]:


df.drop_duplicates(inplace=True)


# In[11]:


#len(df)


# In[12]:


#Delete all entries with a price value <= 0
df=df[df.price>0]


# In[13]:


df.neighbourhood_group.value_counts()


# In[14]:


df.room_type.value_counts()


# In[15]:


get_ipython().run_cell_magic('writefile', 'app.py', '\nimport pandas as pd\nimport numpy as np\nimport streamlit as st\nst.title(\'Explore the features of New York Airbnbs\')\ndf = pd.read_csv(\'NYC-Airbnb-2023.csv\')\ndf.drop(columns=[\'license\'],inplace=True)\ndf.drop_duplicates(inplace=True)\ndf=df[df.price>0]\nif st.checkbox(\'Show Data\'):\n    st.subheader(\'New York City Airbnb Data\')\n    st.write(df)\nman=df[df[\'neighbourhood_group\']==\'Manhattan\']\nmanh=man.groupby(\'room_type\').agg({\'price\': \'mean\',\'minimum_nights\': \'mean\',\n                         \'number_of_reviews\':\'mean\', \n                         \'availability_365\': \'mean\'})\nst.subheader(\'Key features of Airbnb rooms in Manhattan\')\nst.write(manh)\nst.subheader(\'Average price in Manhattan based on room type\')\nst.bar_chart(data=manh.reset_index(), x=\'room_type\', y=\'price\', x_label= \'Room Type\', y_label=\'Average Price\')\nneighbourhood_to_filter = st.select_slider(\'Neighbourhood Group\', list(df.neighbourhood_group.unique())) \nfd= df[df[\'neighbourhood_group\'] == neighbourhood_to_filter]\nfiltered_data=fd.groupby(\'room_type\').agg({\'price\': \'mean\',\'minimum_nights\': \'mean\',\n                         \'number_of_reviews\':\'mean\', \n                         \'availability_365\': \'mean\'})\nfiltered_data.rename(columns={\'minimum_nights\':\'number of minimum stay duration \',\n                  \'number_of_reviews\':\'number of reviews\',\n                    \'availability_365\': \'number of available days in a year\'},inplace=True)\nst.subheader(\'Key features of Airbnb rooms in different neighborhoods of New York ({})\'.format(neighbourhood_to_filter))\nst.write(filtered_data)\nfeature_to_filter = st.select_slider(\'Feature\', list(filtered_data.columns))\nst.subheader(\'Average {} in {} based on room type\'.format(feature_to_filter,neighbourhood_to_filter))\nst.bar_chart(data=filtered_data.reset_index(), x=\'room_type\', y=feature_to_filter, x_label= \'Room Type\', y_label=\'Average {}\'.format(feature_to_filter))\nimport pydeck as pdk\nst.subheader(\'Display the concentration of Airbnb pickups in New York on a map\')\nst.pydeck_chart(\n    pdk.Deck(\n        map_style=None,\n        initial_view_state=pdk.ViewState(\n            latitude=40.730610,\n            longitude=-73.935242,\n            zoom=11,\n            pitch=50,\n        ),\n        layers=[\n            pdk.Layer(\n                "HexagonLayer",\n                data=df[[\'latitude\', \'longitude\']],\n                get_position="[longitude, latitude]",\n                radius=200,\n                elevation_scale=4,\n                elevation_range=[0, 1000],\n                pickable=True,\n                extruded=True,\n            ),\n            pdk.Layer(\n                "ScatterplotLayer",\n                data=df[[\'latitude\', \'longitude\']],\n                get_position="[longitude, latitude]",\n                get_color="[200, 30, 0, 160]",\n                get_radius=200,\n            ),\n        ],\n    )\n)\n\n\n')


# In[16]:


get_ipython().system('streamlit run app.py')


# In[ ]:




