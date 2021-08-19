import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 18, 9
from IPython.display import display, HTML
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

st.title('Data Analytics - Vessels')
st.markdown("""
## Sister vessels data analytics
In this analysis, we look at the relationship of the sister vessels and compare the features available from the datasets provided.
To do that, we first import the datasets into Pandas DataFrame. 
We import the provided csv files *without* the index column
```python
# import csv as pd df and remove first (index) column header
vessel_0 = pd.read_csv('./vessel_data/vessel_0.csv', index_col=0)
vessel_1 = pd.read_csv('./vessel_data/vessel_1.csv', index_col=0)
vessel_2 = pd.read_csv('./vessel_data/vessel_2.csv', index_col=0)
vessel_3 = pd.read_csv('./vessel_data/vessel_3.csv', index_col=0)
```
Before moving on, it was noticed that the indexes are not in proper zero-indexed numerical sequence (e.g. 0, 1, 2, 3, ..., N).
When working with dataframes, in general, it is favourable to work with zero-indexed sequence for each given data entry.
To rectify, we reset the index to zero-indexed format and, at the same time, 
print the shape of each dataframe to have a better overview of the datasets we are dealing with:
```python
# Reset indexes and Print shape of dataframe
vessel_0.index = range(len(vessel_0.index))
print("Vessel 0 DF shape:", vessel_0.shape)
vessel_1.index = range(len(vessel_1.index))
print("Vessel 1 DF shape:", vessel_1.shape)
vessel_2.index = range(len(vessel_2.index))
print("Vessel 2 DF shape:", vessel_2.shape)
vessel_3.index = range(len(vessel_3.index))
print("Vessel 3 DF shape:", vessel_3.shape)
```

The following are the imported datasets:

#### Vessel 0
""")
@st.cache(allow_output_mutation=True)
def load_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df.index = range(len(df.index))
    return df
vessel_0 = load_data("./vessel_data/vessel_0.csv")
st.write(vessel_0)
st.write("Vessel 0 DF shape:", vessel_0.shape)
st.markdown("""
#### Vessel 1
            """)
vessel_1 = load_data("./vessel_data/vessel_1.csv")
st.write(vessel_1)
st.write("Vessel 1 DF shape:", vessel_1.shape)
st.markdown("""
#### Vessel 2
            """)
vessel_2 = load_data("./vessel_data/vessel_2.csv")
st.write(vessel_2)
st.write("Vessel 2 DF shape:", vessel_2.shape)
st.markdown("""
#### Vessel 3
            """)
vessel_3 = load_data("./vessel_data/vessel_3.csv")
st.write(vessel_3)
st.write("Vessel 3 DF shape:", vessel_3.shape)


st.markdown("""
From here, it is possible to note that all required columns are present in all 4 datasets (no missing columns).

Now that the dataframes importing, preprocessing, and initial checks are done, 
let's get to plotting to get a first sense of the data!

We first count the no. of entries by month bins to check irregularities in data collected for different months with interactive bar plots:
```python
# Quick check for irregularity in frequency of entries by month and visualize with bar charts.
def st_plot_datehist(vessel):
    vessel_name = vessel['IMO'][1]
    df_vessels_datehist = pd.DataFrame()
    df_vessels_datehist['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
    df_vessels_datehist_group = df_vessels_datehist.groupby([df_vessels_datehist["Date_UTC"].dt.year, 
                                                             df_vessels_datehist["Date_UTC"].dt.month]).count()
    st.write('#### {} - No. of records logged per month'.format(vessel_name))
    st.write('{} : lowest record of {} entry(ies), highest record of {} entry(ies), with mean at {}'.format(vessel_name, 
                                                                                                         int(df_vessels_datehist_group.min()),
                                                                                                         int(df_vessels_datehist_group.max()),
                                                                                                         int(df_vessels_datehist_group.mean())))
    st.write(vessel_name, 'Data collected from: ', 
             str(np.datetime_as_string(df_vessels_datehist.min(), unit='D')), 
             ' to: ', str(np.datetime_as_string(df_vessels_datehist.max(), unit='D')))
    df_vessels_datehist_group['Date'] = df_vessels_datehist_group.index.to_flat_index()
    df_vessels_datehist_group['Date'] = df_vessels_datehist_group['Date'].apply(lambda x: re.sub('[^A-Za-z0-9]', ' ', str(x)))
    fig = px.bar(df_vessels_datehist_group, x='Date', y='Date_UTC', labels={
        'Date_UTC': 'No. of records logged per month'
    })
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    #df_vessels_datehist_group.plot(kind="bar") # Matplotlib (non-interactive)
    #plt.show()
    #st.pyplot(fig=plt)
```

""")
def st_plot_datehist(vessel):
    vessel_name = vessel['IMO'][1]
    df_vessels_datehist = pd.DataFrame()
    df_vessels_datehist['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
    df_vessels_datehist_group = df_vessels_datehist.groupby([df_vessels_datehist["Date_UTC"].dt.year, 
                                                             df_vessels_datehist["Date_UTC"].dt.month]).count()
    st.write('#### {} - No. of records logged per month'.format(vessel_name))
    st.write('{} : lowest record of {} entry(ies), highest record of {} entry(ies), with mean at {}'.format(vessel_name, 
                                                                                                         int(df_vessels_datehist_group.min()),
                                                                                                         int(df_vessels_datehist_group.max()),
                                                                                                         int(df_vessels_datehist_group.mean())))
    st.write(vessel_name, 'Data collected from: ', 
             str(np.datetime_as_string(df_vessels_datehist.min(), unit='D')), 
             ' to: ', str(np.datetime_as_string(df_vessels_datehist.max(), unit='D')))
    df_vessels_datehist_group['Date'] = df_vessels_datehist_group.index.to_flat_index()
    df_vessels_datehist_group['Date'] = df_vessels_datehist_group['Date'].apply(lambda x: re.sub('[^A-Za-z0-9]', ' ', str(x)))
    fig = px.bar(df_vessels_datehist_group, x='Date', y='Date_UTC', labels={
        'Date_UTC': 'No. of records logged per month'
    })
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)
    #df_vessels_datehist_group.plot(kind="bar") # Matplotlib (non-interactive)
    #plt.show()
    #st.pyplot(fig=plt)
    

vessels = [vessel_0, vessel_1, vessel_2, vessel_3]
for vessel in vessels:
	st_plot_datehist(vessel)
plt.clf()



st.markdown("""
Based on the above plots, there are months with low data entry as compared to other months of the year.
With that insight, we could group the data entries by date range (e.g. days, months, years) and return the **mean** value for that particular range
to have a clearer representation to look out for trends.
The interactive plotting function is written as:
```python
def plot_dateUTC_operation_group_all_vessels_st(vessels, operation, groupby_dateRange):
    list_of_vessels = []
    rows_list = []
    col_list = ['Vessel', 'Total', 'Min', 'Max', 'Mean']
    plots = []
    for vessel in vessels:
        vessel_name = vessel['IMO'][1]
        df_vessel = pd.DataFrame()
        df_vessel['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
        df_vessel[operation] = vessel[operation]
        rows_list.append([vessel_name,
                          df_vessel[operation].sum(),
                          df_vessel[operation].min(),
                          df_vessel[operation].max(),
                          df_vessel[operation].mean()])
        df_table = pd.DataFrame(rows_list, columns=col_list)
        df_vessel_group = df_vessel.groupby(df_vessel['Date_UTC'].dt.strftime(groupby_dateRange))[operation].mean()
        list_of_vessels.append(vessel_name)
        df_vessel_group_rn = df_vessel_group.rename(vessel_name, inplace=True)
        plots.append(df_vessel_group_rn)
    df_plot = pd.concat(plots, axis=1).fillna(0)
    df_plot['Date_UTC'] = df_plot.index
    df_plot = pd.melt(df_plot, id_vars='Date_UTC', value_vars=df_plot.columns[:-1])
    fig = px.line(df_plot, x='Date_UTC', y='value', color='variable', color_discrete_sequence=px.colors.qualitative.D3,
                  labels={
                      'Date_UTC': 'Date_UTC ({})'.format(groupby_dateRange),
                      'value': operation + ' ({} mean)'.format(groupby_dateRange),
                      'variable': 'Vessel'
                  })
    st.table(df_table)
    st.plotly_chart(fig)
```
On top of plotting the measurements as a function of time, 
the above function provides a simple statistics table as well, where the sum, min, max, and mean values are presented.

We can call it with:
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_Consumption', '%Y')
```
where *vessels* is a list consisting the dataframe for each vessel, we provide the feature name that we are interested in (in this case, ```'ME_Consumption'```), 
and finally, with ```'%Y-%m-%d'``` (year, month, day) in datetime strftime format.

We will first look at the trend for the main engine fuel consumption ```'ME_Consumption'``` by yearly basis, `%Y`:

### Fuel consumption at the Main Engine (Yearly)
            """)
# matplotlib
def plot_dateUTC_operation_group_all_vessels(vessels, operation, groupby_dateRange):
    list_of_vessels = []
    rows_list = []
    col_list = ['Vessel', 'Total', 'Min', 'Max', 'Mean']
    for vessel in vessels:
        vessel_name = vessel['IMO'][1]
        df_vessel = pd.DataFrame()
        df_vessel['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
        df_vessel[operation] = vessel[operation]
        rows_list.append([vessel_name,
                          df_vessel[operation].sum(),
                          df_vessel[operation].min(),
                          df_vessel[operation].max(),
                          df_vessel[operation].mean()])
        df_table = pd.DataFrame(rows_list, columns=col_list)
        df_vessel_group = df_vessel.groupby(df_vessel['Date_UTC'].dt.strftime(groupby_dateRange))[operation].mean()
        plt.xlabel("Date")
        plt.ylabel(operation)
        plt.plot(df_vessel_group)
        list_of_vessels.append(vessel_name)
    plt.legend(list_of_vessels)
    st.table(df_table)
    st.pyplot(plt)

# plotly 
def plot_dateUTC_operation_group_all_vessels_st(vessels, operation, groupby_dateRange):
    list_of_vessels = []
    rows_list = []
    col_list = ['Vessel', 'Total', 'Min', 'Max', 'Mean']
    plots = []
    for vessel in vessels:
        vessel_name = vessel['IMO'][1]
        df_vessel = pd.DataFrame()
        df_vessel['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
        df_vessel[operation] = vessel[operation]
        rows_list.append([vessel_name,
                          df_vessel[operation].sum(),
                          df_vessel[operation].min(),
                          df_vessel[operation].max(),
                          df_vessel[operation].mean()])
        df_table = pd.DataFrame(rows_list, columns=col_list)
        df_vessel_group = df_vessel.groupby(df_vessel['Date_UTC'].dt.strftime(groupby_dateRange))[operation].mean()
        list_of_vessels.append(vessel_name)
        df_vessel_group_rn = df_vessel_group.rename(vessel_name, inplace=True)
        plots.append(df_vessel_group_rn)
    df_plot = pd.concat(plots, axis=1).fillna(0)
    df_plot['Date_UTC'] = df_plot.index
    df_plot = pd.melt(df_plot, id_vars='Date_UTC', value_vars=df_plot.columns[:-1])
    fig = px.line(df_plot, x='Date_UTC', y='value', color='variable', color_discrete_sequence=px.colors.qualitative.D3,
                  labels={
                      'Date_UTC': 'Date_UTC ({})'.format(groupby_dateRange),
                      'value': operation + ' ({} mean)'.format(groupby_dateRange),
                      'variable': 'Vessel'
                  })
    st.table(df_table)
    st.plotly_chart(fig)

def plot_dateUTC_operation_group_all_vessels_table_only(vessels, operation, groupby_dateRange):
    list_of_vessels = []
    rows_list = []
    col_list = ['Vessel', 'Total', 'Min', 'Max', 'Mean']
    for vessel in vessels:
        vessel_name = vessel['IMO'][1]
        df_vessel = pd.DataFrame()
        df_vessel['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
        df_vessel[operation] = vessel[operation]
        rows_list.append([vessel_name,
                          df_vessel[operation].sum(),
                          df_vessel[operation].min(),
                          df_vessel[operation].max(),
                          df_vessel[operation].mean()])
        df_table = pd.DataFrame(rows_list, columns=col_list)
        list_of_vessels.append(vessel_name)
    st.table(df_table)


    
    
    
    
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_Consumption', '%Y')
st.write('''
By reading in the fuel consumption measurements from the main engine, 
we can see that there are some similarities (in terms of fuel consumption range) between `vessel_0` and `vessel_1`, 
as well as similarities between `vessel_2` and `vessel_3`.

Also, readings from the stats table show that the sum, minimum, maximum, and mean values of
`vessel_0` and `vessel_1` are within the same range and
as with `vessel_2` and `vessel_3`.

#### Current hypothesis:
 - vessel_0 and vessel_1 are sister vessels, whereas,
 - vessel_2 and vessel_3 are sister vessels.

This is a good start!
However, we have to dive in deeper on the analytics to prove our hypothesis.
Even though the description mentioned that
"Vessels 1 & 2 are sister vessels. Similarly vessels 3 and 4 are sister vessels.", 
there could still be a chance that the csv files does not correspond sequentially 
(e.g. Vessel 1 in description could be `vessel_3` and Vessel 2 in description could be `vessel_0`, etc.).

According to the description provided, the main engine (ME) is the largest fuel consumer, 
let's look into the consumption of the different fuel types at the main engine.

### Data cleaning - Checks on fuel types
Since we know (from the description) that the total consumption is the sum of MGO consumption and HFO consumption, 
let's run some checks and processing first before diving into the fuel types.
- Check that all data entries follow `consumption_total = consumption_MGO + consumption_HFO` given the 3 types of the consumer columns (boiler, ME, and AE)
- Proposed workaround if data entry doesn't meet the requirement (pseudo):
```
- If consumption_total == 0 and consumption_HFO != 0 or consumption_MGO != 0,
    then, consumption_total = consumption_HFO + consumption_MGO
- If consumption_total > consumption_HFO + consumption_MGO, 
    then, consumption_total = consumption_HFO + consumption_MGO
- If consumption_MGO > consumption_total, 
    then, consumption_MGO = consumption_total - consumption_HFO
- If consumption_MFO >= consumption_total and consumption_HFO != 0,
    then, consumption_MGO = consumption_total - consumption_HFO
- If consumption_HFO > consumption_total, 
    then, consumption_HFO = consumption_total - consumption_MGO
- If consumption_HFO >= consumption_total and consumption_MGO != 0,
    then, consumption_HFO = consumption_total - consumption_MGO
```
- This way, we can make sure that the data follows the given formula and clean off the possible errors present in dataset.

Here are some of the data entries with errors (consumption_total != consumption_HFO + consumption_MGO) highlighted after running `np.where` with condition on dataframe:
```python
print(vessel_0['ME_Consumption'][2])        # 9.9
print(vessel_0['ME_Consumption_MGO'][2])    # 16.6
print(vessel_0['ME_Consumption_HFO'][2])    # 0.0

print(vessel_0['ME_Consumption'][1257])     # 1.8
print(vessel_0['ME_Consumption_MGO'][1257]) # 0.4
print(vessel_0['ME_Consumption_HFO'][1257]) # 1.8

print(vessel_1['ME_Consumption'][894])      # 25.2
print(vessel_1['ME_Consumption_MGO'][894])  # 7.1
print(vessel_1['ME_Consumption_HFO'][894])  # 25.2

print(vessel_3['ME_Consumption'][772])      # 14.6
print(vessel_3['ME_Consumption_MGO'][772])  # 2.2
print(vessel_3['ME_Consumption_HFO'][772])  # 14.6
```


#### Putting in to code
The code snippet below shows the implementation.
```python
def clean_consumption_cols(vessels, consumers):
    for consumer in consumers:
        consumer_HFO = consumer + '_HFO'
        consumer_MGO = consumer + '_MGO'
        for vessel in vessels:
            bool_list = np.where(vessel[consumer] == vessel[consumer_HFO] + vessel[consumer_MGO], True, False)
            for i, j in enumerate(bool_list):
                if not j:
                    if np.float(vessel.loc[[i]][consumer]) == 0 and np.float(vessel.loc[[i]][consumer_HFO]) != 0 or np.float(vessel.loc[[i]][consumer_HFO]) != 0:
                        vessel.at[i, consumer] = np.float(vessel.loc[[i]][consumer_HFO]) + np.float(vessel.loc[[i]][consumer_MGO])
                        
                    elif np.float(vessel.loc[[i]][consumer]) > np.float(vessel.loc[[i]][consumer_HFO]) + np.float(vessel.loc[[i]][consumer_MGO]):
                        vessel.at[i, consumer] = np.float(vessel.loc[[i]][consumer_HFO]) + np.float(vessel.loc[[i]][consumer_MGO])

                    elif np.float(vessel.loc[[i]][consumer_MGO]) > np.float(vessel.loc[[i]][consumer]):
                        vessel.at[i, consumer_MGO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_HFO])

                    elif np.float(vessel.loc[[i]][consumer_MGO]) >= np.float(vessel.loc[[i]][consumer]) and np.float(vessel.loc[[i]][consumer_HFO]) != 0:
                        vessel.at[i, consumer_MGO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_HFO])

                    elif np.float(vessel.loc[[i]][consumer_HFO]) > np.float(vessel.loc[[i]][consumer]):
                        vessel.at[i, consumer_HFO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_MGO])

                    elif np.float(vessel.loc[[i]][consumer_HFO]) >= np.float(vessel.loc[[i]][consumer]) and np.float(vessel.loc[[i]][consumer_MGO]) != 0:
                        vessel.at[i, consumer_HFO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_MGO])

```
And we call it with
```python
consumers = ['ME_Consumption', 'Boiler_Consumption']
clean_consumption_cols(vessels, consumers)
```

Check on some of the data entry with errors previously:
```python
print(vessel_0['ME_Consumption'][2])        # 9.9
print(vessel_0['ME_Consumption_MGO'][2])    # 9.9
print(vessel_0['ME_Consumption_HFO'][2])    # 0.0

print(vessel_0['ME_Consumption'][1257])     # 1.8
print(vessel_0['ME_Consumption_MGO'][1257]) # 0.4
print(vessel_0['ME_Consumption_HFO'][1257]) # 1.4

print(vessel_1['ME_Consumption'][894])      # 25.2
print(vessel_1['ME_Consumption_MGO'][894])  # 7.1
print(vessel_1['ME_Consumption_HFO'][894])  # 18.1

print(vessel_3['ME_Consumption'][772])      # 14.6
print(vessel_3['ME_Consumption_MGO'][772])  # 2.2
print(vessel_3['ME_Consumption_HFO'][772])  # 12.4
```

That's great! 
Now we have suffice the condition of `conumption_total = consumption_MGO + consumption_HFO` for both `'ME_Consumption'` and `'Boiler_Consumption'`.
''')

def clean_consumption_cols(vessels, consumers):
    for consumer in consumers:
        consumer_HFO = consumer + '_HFO'
        consumer_MGO = consumer + '_MGO'
        for vessel in vessels:
            # Ensure formula adds up
            bool_list = np.where(vessel[consumer] == vessel[consumer_HFO] + vessel[consumer_MGO], True, False)
            for i, j in enumerate(bool_list):
                if not j:
                    if np.float(vessel.loc[[i]][consumer]) == 0 and np.float(vessel.loc[[i]][consumer_HFO]) != 0 or np.float(vessel.loc[[i]][consumer_HFO]) != 0:
                        vessel.at[i, consumer] = np.float(vessel.loc[[i]][consumer_HFO]) + np.float(vessel.loc[[i]][consumer_MGO])
                        
                    elif np.float(vessel.loc[[i]][consumer]) > np.float(vessel.loc[[i]][consumer_HFO]) + np.float(vessel.loc[[i]][consumer_MGO]):
                        vessel.at[i, consumer] = np.float(vessel.loc[[i]][consumer_HFO]) + np.float(vessel.loc[[i]][consumer_MGO])

                    elif np.float(vessel.loc[[i]][consumer_MGO]) > np.float(vessel.loc[[i]][consumer]):
                        vessel.at[i, consumer_MGO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_HFO])

                    elif np.float(vessel.loc[[i]][consumer_MGO]) >= np.float(vessel.loc[[i]][consumer]) and np.float(vessel.loc[[i]][consumer_HFO]) != 0:
                        vessel.at[i, consumer_MGO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_HFO])

                    elif np.float(vessel.loc[[i]][consumer_HFO]) > np.float(vessel.loc[[i]][consumer]):
                        vessel.at[i, consumer_HFO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_MGO])

                    elif np.float(vessel.loc[[i]][consumer_HFO]) >= np.float(vessel.loc[[i]][consumer]) and np.float(vessel.loc[[i]][consumer_MGO]) != 0:
                        vessel.at[i, consumer_HFO] = np.float(vessel.loc[[i]][consumer]) - np.float(vessel.loc[[i]][consumer_MGO])

consumers = ['ME_Consumption', 'Boiler_Consumption']
clean_consumption_cols(vessels, consumers)


st.write('''
### Column wrangling - the AE_Consumption feature column
Before moving on, it was noticed that the provided datasets do not include an `'AE_Consumption'` feature column.

We know (from the description) that the total consumption is the sum of MGO consumption and HFO consumption,
we can create a new feature column, `'AE_Consumption'`, by adding up `'AE_Consumption_MGO'` and `'AE_Consumption_HFO'` with the following code snippet:
```python
for vessel in vessels:
    vessel['AE_Consumption'] = vessel['AE_Consumption_MGO'] + vessel['AE_Consumption_HFO']

# Print shape of dataframe
print(vessel_0.shape) # (1338, 34)
print(vessel_1.shape) # (1061, 34)
print(vessel_2.shape) # (1376, 34)
print(vessel_3.shape) # (1260, 34)
```
By checking the shape of the dataframes, we will now have 34 feature columns instead of the previously 33 feature columns.

With that, we can now plot the total consumption for AE with `'AE_Consumption'` feature column, and consistently, plot the consumption of 
the different fuel types at the AE.


For now, let's look at the different fuel types at the main engine!
         ''')

for vessel in vessels:
    vessel['AE_Consumption'] = vessel['AE_Consumption_MGO'] + vessel['AE_Consumption_HFO']



st.write('''
### Fuel consumption [Heavy Fuel Oil (HFO)] at the Main Engine (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_Consumption_HFO', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_Consumption_HFO', '%Y')

st.write('''
### Fuel consumption [Marine Gas Oil (MGO)] at the Main Engine (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_Consumption_MGO', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_Consumption_MGO', '%Y')

st.write('''
The plots and tables for ME (HFO vs MGO) show that `vessel_2` and `vessel_3` consumes a higher amount of heavy fuel oil (HFO) than `vessel_0` and `vessel_1`.
However, `vessel_0` and `vessel_1` consumes a higher amount of marine gas oil (MGO) than `vessel_2` and `vessel_3`.

Up till now, the readings at the ME for "`vessel_0` with `vessel_1`" and "`vessel_2` with `vessel_3`" are still closely correlated. 
However, we have found new insights when looking at the different consumption of HFO and MGO for the different vessels.

With that, we update our hypothesis with the new insights. 
#### Current hypothesis
 - `vessel_0` and `vessel_1` are sister vessels (Unchanged)
 - `vessel_2` and `vessel_3` are sister vessels (Unchanged)
 - `vessel_0` and `vessel_1` are sister vessels that consumes higher MGO (New)
 - `vessel_2` and `vessel_3` are sister vessels that consumes higher HFO (New)

Next, let's look at the auxiliary engine (AE) to test our hypothesis and gather further insights on the different fuel types!

''')

st.write('''
         ### Fuel consumption at Auxiliary Engine (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'AE_Consumption', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'AE_Consumption', '%Y')



st.write('''
         ### Fuel consumption (HFO) at Auxiliary Engine (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'AE_Consumption_HFO', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'AE_Consumption_HFO', '%Y')

st.write('''
         ### Fuel consumption (MGO) at Auxiliary Engine (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'AE_Consumption_MGO', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'AE_Consumption_MGO', '%Y')

st.write('''
Similar to the ME, plots and tables for AE (HFO vs MGO) show that vessel_2 and vessel_3 consumes a higher amount of heavy fuel oil (HFO) than vessel_0 and vessel_1. 
However, vessel_0 and vessel_1 consumes a higher amount of marine gas oil (MGO) than vessel_2 and vessel_3.

Our hypothesis remained unchanged.

#### Current hypothesis
- vessel_0 and vessel_1 are sister vessels (Unchanged)
- vessel_2 and vessel_3 are sister vessels (Unchanged)
- vessel_0 and vessel_1 are sister vessels that consumes higher MGO (Unchanged)
- vessel_2 and vessel_3 are sister vessels that consumes higher HFO (Unchanged)

Let's look at the boiler consumption to further consolidate our hypothesis on the different fuel types!
         ''')

st.write('''
         ### Boiler Consumption in MT (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Boiler_Consumption', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Boiler_Consumption', '%Y')

st.write('''
         ### Boiler Consumption (HFO) in MT (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Boiler_Consumption_HFO', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Boiler_Consumption_HFO', '%Y')

st.write('''
         ### Boiler Consumption (MGO) in MT (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Boiler_Consumption_MGO', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Boiler_Consumption_MGO', '%Y')

st.write('''
Interestingly, when we look at the total boiler consumption (regardless of fuel types), all 4 vessels did not contrast much.
However, the different fuel types tell a different story.

Over the span from 2018-2021,
vessels `vessel_0` and `vessel_1` boiler consumption (HFO) have a sum of 387.6924 MT and 337.6500 MT, respectively, whereas,
Vessels `vessel_2` and `vessel_3` boiler consumption (HFO) have a sum of 731.8500 MT and 1,408.1000 MT, respectively.

Vessels `vessel_0` and `vessel_1` boiler consumption (MGO) have a sum of 927.7067 MT and 905.5500 MT, respectively, whereas,
Vessels `vessel_2` and `vessel_3` boiler consumption (MGO) have a sum of 272.8950 MT and 371.3500 MT, respectively.

It seems that our hypothesis of MGO/HFO consumption for the sister vessels remains unchanged.

According to the description provided:
"MGO is a cleaner but more expensive fuel than HFO. 
Vessel owners/operators prefer to use the cheaper fuel unless some regulations prevent it (such as near coastal areas or in ports)."


With that, we update our hypothesis with the new insights. 
#### Current hypothesis
 - `vessel_0` and `vessel_1` are sister vessels (Unchanged)
 - `vessel_2` and `vessel_3` are sister vessels (Unchanged)
 - `vessel_0` and `vessel_1` are sister vessels that consumes higher MGO (Unchanged)
 - `vessel_2` and `vessel_3` are sister vessels that consumes higher HFO (Unchanged)
 - `vessel_0` and `vessel_1` are sister vessels that generally travel near coastal areas. (New)
    - because of that, `vessel_0` and `vessel_1` sister vessels clock lower distance travelled per year. (New)
 - `vessel_2` and `vessel_3` are sister vessels that generally travel out in the seas. (New)
    - because of that, `vessel_2` and `vessel_3` sister vessels clock higher distance travelled per year. (New)

The `'Distance'` feature column comes into mind when we want to identify if a vessel generally travel out in the seas or travel near coastal area. 

The basis for this assumption stems from: 
vessels that travel more in a year tend to be out in the seas (travelling longer distance, therefore, higher distance (NM, Yearly) clocked), 
as compared to vessels that travel along/near coastal areas (travelling shorter distance, therefore, lower distance (NM, Yearly) clocked).

Let's plot the yearly distance travelled for each of the vessel to find out if the data coincides with the assumption.
''')

st.write('''
         ### Distance travelled in Nautical Miles (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Distance', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Distance', '%Y')

st.write('''
The `'Distance'` plot shows that both `vessel_2` and `vessel_3` travels more (in Nautical Miles) per year than `vessel_0` and `vessel_1`.
The `'Distance'` plot supports our previous assumption that vessels 0 and 1 clock lower distance travelled per year as compared to vessels 2 and 3.

#### Current hypothesis
- vessel_0 and vessel_1 are sister vessels (Unchanged)
- vessel_2 and vessel_3 are sister vessels (Unchanged)
- vessel_0 and vessel_1 are sister vessels that consumes higher MGO (Unchanged)
- vessel_2 and vessel_3 are sister vessels that consumes higher HFO (Unchanged)
- vessel_0 and vessel_1 are sister vessels that generally travel near coastal areas. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels clock lower distance travelled per year. (Unchanged)
- vessel_2 and vessel_3 are sister vessels that generally travel out in the seas. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels clock higher distance travelled per year. (Unchanged)

In order to suffice the assumption of
- vessel_0 and vessel_1 are sister vessels that generally travel near coastal areas. (Unchanged)
- vessel_2 and vessel_3 are sister vessels that generally travel out in the seas. (Unchanged)

the line plots so far do not supply a good foundation for these assumptions.

However, we are given the logitudinal and latitudinal coordinates for each data entry! 
Let's check out where the vessles have visited.

We write a function that leverages on Mapbox OpenStreetMap to plot the coordinates over the world map with the streamlit library:

```python
def plot_vessels_on_map(vessel):
    vessel_name = vessel['IMO'][1]
    df_vessels_datehist = pd.DataFrame()
    df_vessels_datehist['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
    st.write('### ' + vessel_name, '\nLongitudinal and latitudinal data collected from: ', 
             str(np.datetime_as_string(df_vessels_datehist.min(), unit='D')), 
             ' to: ', str(np.datetime_as_string(df_vessels_datehist.max(), unit='D')))
    df_map = vessel[['Latitude_Degree', 'Longitude_Degree']]
    df_map.columns = ['lat', 'lon']
    st.map(df_map)
```

Now we can plot the coordinates and observe the frequented areas of the vessles:
         ''')


def plot_vessels_on_map(vessel):
    vessel_name = vessel['IMO'][1]
    df_vessels_datehist = pd.DataFrame()
    df_vessels_datehist['Date_UTC'] = pd.to_datetime(vessel['Date_UTC'], infer_datetime_format=True)
    st.write('### ' + vessel_name, '\nLongitudinal and latitudinal data collected from: ', 
             str(np.datetime_as_string(df_vessels_datehist.min(), unit='D')), 
             ' to: ', str(np.datetime_as_string(df_vessels_datehist.max(), unit='D')))
    df_map = vessel[['Latitude_Degree', 'Longitude_Degree']]
    df_map.columns = ['lat', 'lon']
    st.map(df_map)

for i in vessels:
    plot_vessels_on_map(i)



st.write('''
From the coordinates plot over the world map, we observe that both vessels 0 and 1 frequented routes and ports around the 
Mediterranean Sea, Indian Ocean, Middle East region, and South Asia region.

Whereas, vessels 2 and 3 frequented to further places and ports out towards South East Asia, South China Sea, and even the Pacific region.

The observations coincide with our previous assumption that vessels 0 and 1 travelled and frequented near coastal areas more so than vessles 2 and 3.

An update on the hypothesis:

#### Current hypothesis
- vessel_0 and vessel_1 are sister vessels (Unchanged)
- vessel_2 and vessel_3 are sister vessels (Unchanged)
- vessel_0 and vessel_1 are sister vessels that consumes higher MGO (Unchanged)
- vessel_2 and vessel_3 are sister vessels that consumes higher HFO (Unchanged)
- vessel_0 and vessel_1 are sister vessels that generally travel near coastal areas. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels clock lower distance travelled per year. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels carry a smaller cargo load on board (MT) per year. (New) 
- vessel_2 and vessel_3 are sister vessels that generally travel out in the seas. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels clock higher distance travelled per year. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels carry a greater cargo load on board (MT) per year. (New) 

Let's plot the average weight of cargo carried on board per year.
    ''')



st.write('''
         #### Weight of cargo carried on board in MT (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_table_only(vessels, 'Cargo_Mt', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_table_only(vessels, 'Cargo_Mt', '%Y')

st.write('''
Before we plot, a quick table of summary showed that the min of cargo load (MT) for `vessel_3` has a value of -323.1000.
This seems like an error in data entry considering the negative weight on board, measured in metric tonnes (MT).

To rectify the error in data entry, we implemented a quick function that replaces negative numbers to 0:

```python
def df_col_replace_neg_to_zero(vessels, feature):
    for vessel in vessels:
        vessel[feature] = np.where(vessel[feature] < 0, 0, vessel[feature])
```

and called the function for the `'Cargo_MT'` feature.
```python
df_col_replace_neg_to_zero(vessels, 'Cargo_Mt')
```
         ''')

def df_col_replace_neg_to_zero(vessels, feature):
    for vessel in vessels:
        vessel[feature] = np.where(vessel[feature] < 0, 0, vessel[feature])

df_col_replace_neg_to_zero(vessels, 'Cargo_Mt')

st.write('''
We plot the corrected feature column for `Cargo_Mt`: 
         #### Weight of cargo carried on board in MT (Yearly) after correction
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'Cargo_Mt', '%Y')
```
         ''')

plot_dateUTC_operation_group_all_vessels_st(vessels, 'Cargo_Mt', '%Y')

st.write('''
Based on the `'Cargo_Mt'` plots, 
it seems that our previous hypothesis at which vessels 0 and 1 carry a smaller cargo load on board than vessels 2 and 3 holds true.

#### Current hypothesis
- vessel_0 and vessel_1 are sister vessels (Unchanged)
- vessel_2 and vessel_3 are sister vessels (Unchanged)
- vessel_0 and vessel_1 are sister vessels that consumes higher MGO (Unchanged)
- vessel_2 and vessel_3 are sister vessels that consumes higher HFO (Unchanged)
- vessel_0 and vessel_1 are sister vessels that generally travel near coastal areas. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels clock lower distance travelled per year. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels carry a smaller cargo load on board (MT) per year. (Unchanged) 
        - because of that, `vessel_0` and `vessel_1` sister vessels have lower emissions of CO2 and NOX (New)
- vessel_2 and vessel_3 are sister vessels that generally travel out in the seas. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels clock higher distance travelled per year. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels carry a greater cargo load on board (MT) per year. (Unchanged) 
        - because of that, `vessel_2` and `vessel_3` sister vessels have higher emissions of CO2 and NOX (New)
    
We hypothesize that since vessels 2 and 3 travels a higher distance a year, with greater cargo load, the CO2 and NOX emssions of vessels 2 and 3 will be
much higher than that of vessels 0 and 1.

Let's look at the CO2 emssions for the vessles.

    ''')

st.write('''
         #### CO2 Emissions (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'CO2_Emitted', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'CO2_Emitted', '%Y')
st.write('''
         #### NOX Emissions (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'NOX_Emitted', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'NOX_Emitted', '%Y')

st.write('''
Based on the `'Cargo_Mt'` plots, 
it seems that our previous hypothesis, where vessels 2 and 3 has higher CO2 and NOX emissions than vessels 0 and 1, holds true.

#### Current hypothesis
- vessel_0 and vessel_1 are sister vessels (Unchanged)
- vessel_2 and vessel_3 are sister vessels (Unchanged)
- vessel_0 and vessel_1 are sister vessels that consumes higher MGO (Unchanged)
- vessel_2 and vessel_3 are sister vessels that consumes higher HFO (Unchanged)
- vessel_0 and vessel_1 are sister vessels that generally travel near coastal areas. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels clock lower distance travelled per year. (Unchanged)
    - because of that, `vessel_0` and `vessel_1` sister vessels carry a smaller cargo load on board (MT) per year. (Unchanged) 
        - because of that, `vessel_0` and `vessel_1` sister vessels have lower emissions of CO2 and NOX (Unchanged)
- vessel_2 and vessel_3 are sister vessels that generally travel out in the seas. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels clock higher distance travelled per year. (Unchanged)
    - because of that, `vessel_2` and `vessel_3` sister vessels carry a greater cargo load on board (MT) per year. (Unchanged) 
        - because of that, `vessel_2` and `vessel_3` sister vessels have higher emissions of CO2 and NOX (Unchanged)
    
For now, all of the previous hypotheses made remain unchanged.
Let's explore other feature columns. 
Next, on the fuel efficiency, `ME_calculated_SFOC`.

    ''')



st.write('''
         #### Specific Fuel Oil Consumption (SFOC) at Main Engine (Yearly)
```python
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_calculated_SFOC', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_calculated_SFOC', '%Y')

st.write('''
Based on the above plot of average `ME_calculated_SFOC` per year, and the stats table,
it seems that there could be an error in data entry - 
where the max value at `vessel_0` is at g/kWh, almost 10 times the value of the other vessels.

Let's zoom in and take a closer look at the plots for average at monthly and daily entries.
         ''')

st.write('''
         #### Specific Fuel Oil Consumption (SFOC) at Main Engine (Monthly)
```python
plot_dateUTC_operation_group_all_vessels(vessels, 'ME_calculated_SFOC', '%Y-%m')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_calculated_SFOC', '%Y-%m')

st.write('''
         #### Specific Fuel Oil Consumption (SFOC) at Main Engine (Daily)
```python
plot_dateUTC_operation_group_all_vessels(vessels, 'ME_calculated_SFOC', '%Y-%m-%d')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels, 'ME_calculated_SFOC', '%Y-%m-%d')

st.write('''
From the above plots of average `ME_calculated_SFOC` monthly and daily, we can certainly tell that there are errors in the entries considering the huge spike
in reading at `vessel_0`.

The rectify the issue, we plot the boxplot for each vessel to measure and calculate quantifiable threshold value based on the quantile value.
The threshold values are used as the conditional marker to replace the values that doesn't meet it requirements to NA.
Then, we replace the NA values with forward-fill, where the forward-fill function propagates the 
last observed non-null value forward until another non-null value is encountered.

A function is written to forward-fill the outliers and plot the boxplots for before and after the formula is applied:

```python
def plot_box_BA_quantile(vessel, operation, hi_quantile=0.99, lo_quantile=0.01):
    vessel_name = vessel['IMO'][1]
    st.write('''
    ### {} {} (all entries) Box plot\n(before forward filling detected outliers)
            '''.format(vessel_name, operation))
    fig = px.box(vessel , y=operation)
    st.plotly_chart(fig)

    v_q_hi  = vessel[operation].quantile(hi_quantile)
    v_q_lo  = vessel[operation].quantile(lo_quantile)
    st.write("Threshold cut-off (higher quantile={}): ".format(hi_quantile), v_q_hi)
    st.write("Threshold cut-off (lower quantile={}): ".format(lo_quantile), v_q_lo)

    vessel[operation] = vessel[operation].mask(vessel[operation] > v_q_hi)
    vessel[operation] = vessel[operation].mask(vessel[operation] <= v_q_lo)
    vessel[operation].fillna(method='ffill')

    st.write('''
    ### {} {} (all entries) Box plot\n(after forward filling detected outliers)
            '''.format(vessel_name, operation))
 
    fig = px.box(vessel , y=operation)
    st.plotly_chart(fig)
    return vessel
```

We call the function and applied it to all of the vessel dataframes:
```python
vessel_0_filtered = plot_box_BA_quantile(vessel_0, 'ME_calculated_SFOC')
vessel_1_filtered = plot_box_BA_quantile(vessel_1, 'ME_calculated_SFOC')
vessel_2_filtered = plot_box_BA_quantile(vessel_2, 'ME_calculated_SFOC')
vessel_3_filtered = plot_box_BA_quantile(vessel_3, 'ME_calculated_SFOC')
```

Let's look at the box plots before and after the function is applied.
         ''')


def plot_box_BA_quantile(vessel, operation, hi_quantile=0.99, lo_quantile=0.01):
    vessel_name = vessel['IMO'][1]
    st.write('''
    ### {} {} (all entries) Box plot\n(before forward filling detected outliers)
            '''.format(vessel_name, operation))
    fig = px.box(vessel , y=operation)
    st.plotly_chart(fig)

    v_q_hi  = vessel[operation].quantile(hi_quantile)
    v_q_lo  = vessel[operation].quantile(lo_quantile)
    st.write("Threshold cut-off (higher quantile={}): ".format(hi_quantile), v_q_hi)
    st.write("Threshold cut-off (lower quantile={}): ".format(lo_quantile), v_q_lo)

    vessel[operation] = vessel[operation].mask(vessel[operation] > v_q_hi)
    vessel[operation] = vessel[operation].mask(vessel[operation] <= v_q_lo)
    vessel[operation].fillna(method='ffill')

    st.write('''
    ### {} {} (all entries) Box plot\n(after forward filling detected outliers)
            '''.format(vessel_name, operation))
 
    fig = px.box(vessel , y=operation)
    st.plotly_chart(fig)
    return vessel

vessel_0_filtered = plot_box_BA_quantile(vessel_0, 'ME_calculated_SFOC')
vessel_1_filtered = plot_box_BA_quantile(vessel_1, 'ME_calculated_SFOC')
vessel_2_filtered = plot_box_BA_quantile(vessel_2, 'ME_calculated_SFOC')
vessel_3_filtered = plot_box_BA_quantile(vessel_3, 'ME_calculated_SFOC')

vessels_filtered = [vessel_0_filtered, vessel_1_filtered, vessel_2_filtered, vessel_3_filtered]

st.write('''
Let's look at the line charts after the error outliers are removed:
         ''')

st.write('''
         #### Specific Fuel Oil Consumption (SFOC) at Main Engine (Yearly) - After removing outliers
```python
plot_dateUTC_operation_group_all_vessels(vessels, 'ME_calculated_SFOC', '%Y')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels_filtered, 'ME_calculated_SFOC', '%Y')

st.write('''
         #### Specific Fuel Oil Consumption (SFOC) at Main Engine (Monthly) - After removing outliers
```python
plot_dateUTC_operation_group_all_vessels(vessels, 'ME_calculated_SFOC', '%Y-%m')
```
         ''')
plot_dateUTC_operation_group_all_vessels_st(vessels_filtered, 'ME_calculated_SFOC', '%Y-%m')


st.write('''
The observations on the `ME_calculated_SFOC` line-plots do not have drastic contrast between the vessels.
Let's move on and find out the relationships between different feature columns, objectively.
To quantify the relationship between the different feature columns, we can plot a heatmap of the correlation between feature columns.
Let's find out the correlation between the different feature columns.

We write a function to expedite the process where the correlation matrix is shown to us as an interactive heatmap:
```python
def vessel_corr(vessel, color='Blues'):
    vessel_name = vessel['IMO'][1]
    v_corr = vessel.corr()
    corr_mask = np.triu(np.ones_like(v_corr, dtype=bool))
    df_mask = v_corr.mask(corr_mask)

    heat = go.Heatmap(
        z = df_mask,
        x = df_mask.columns.values,
        y = df_mask.columns.values,
        xgap = 1, # Sets the horizontal gap (in pixels) between bricks
        ygap = 1,
        colorscale = color
    )

    title = vessel_name + ' Feature Correlation Matrix'

    layout = go.Layout(
        title_text=title, 
        title_x=0.5, 
        width=700, 
        height=700,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig=go.Figure(data=[heat], layout=layout)
    st.plotly_chart(fig)
```

Let's look at the correlation between the different feature columns, per vessel.
         ''')
# correlation

def vessel_corr(vessel, color='Blues'):
    vessel_name = vessel['IMO'][1]
    v_corr = vessel.corr()
    corr_mask = np.triu(np.ones_like(v_corr, dtype=bool))
    df_mask = v_corr.mask(corr_mask)

    heat = go.Heatmap(
        z = df_mask,
        x = df_mask.columns.values,
        y = df_mask.columns.values,
        xgap = 1, # Sets the horizontal gap (in pixels) between bricks
        ygap = 1,
        colorscale = color
    )

    title = vessel_name + ' Feature Correlation Matrix'

    layout = go.Layout(
        title_text=title, 
        title_x=0.5, 
        width=700, 
        height=700,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig=go.Figure(data=[heat], layout=layout)
    st.plotly_chart(fig)

vessel_corr(vessel_0_filtered)
vessel_corr(vessel_1_filtered)
vessel_corr(vessel_2_filtered)
vessel_corr(vessel_3_filtered)


st.write('''
To provide a better presentation of the highly correlated features and negatively correlated features,
we can highlight with different correlation threshold values to have a better presentation with the function:

```python
def vessel_corr_highlight(vessel, threshold, color='Greens'):
    vessel_name = vessel['IMO'][1]
    v_corr = vessel.corr()
    if threshold > 0:
        v_corr = v_corr[v_corr>=threshold]
    else:
        v_corr = v_corr[v_corr<=threshold]
    corr_mask = np.triu(np.ones_like(v_corr, dtype=bool))
    df_mask = v_corr.mask(corr_mask)

    heat = go.Heatmap(
        z = df_mask,
        x = df_mask.columns.values,
        y = df_mask.columns.values,
        xgap = 1, # Sets the horizontal gap (in pixels) between bricks
        ygap = 1,
        colorscale = color
    )

    title = vessel_name + ' Feature Correlation Matrix (with threshold={})'.format(threshold)

    layout = go.Layout(
        title_text=title, 
        title_x=0.5, 
        width=700, 
        height=700,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig=go.Figure(data=[heat], layout=layout)
    st.plotly_chart(fig)
```

Then, we pass in the treshold value to highlight the highly correlated features and plot it in green.

And negatively correlated features and plot it in red.

```python
vessel_corr_highlight(vessel_0_filtered, 0.8)
vessel_corr_highlight(vessel_1_filtered, 0.8)
vessel_corr_highlight(vessel_2_filtered, 0.8)
vessel_corr_highlight(vessel_3_filtered, 0.8)

vessel_corr_highlight(vessel_0_filtered, -0.2, color='Reds')
vessel_corr_highlight(vessel_1_filtered, -0.2, color='Reds')
vessel_corr_highlight(vessel_2_filtered, -0.2, color='Reds')
vessel_corr_highlight(vessel_3_filtered, -0.2, color='Reds')
```
This will enable us to view the correlation matrix with ease where only the matrix cells that suffice the threshold condition are displayed.
         ''')

def vessel_corr_highlight(vessel, threshold, color='Greens'):
    vessel_name = vessel['IMO'][1]
    v_corr = vessel.corr()
    if threshold > 0:
        v_corr = v_corr[v_corr>=threshold]
    else:
        v_corr = v_corr[v_corr<=threshold]
    corr_mask = np.triu(np.ones_like(v_corr, dtype=bool))
    df_mask = v_corr.mask(corr_mask)

    heat = go.Heatmap(
        z = df_mask,
        x = df_mask.columns.values,
        y = df_mask.columns.values,
        xgap = 1, # Sets the horizontal gap (in pixels) between bricks
        ygap = 1,
        colorscale = color
    )

    title = vessel_name + ' Feature Correlation Matrix (with threshold={})'.format(threshold)

    layout = go.Layout(
        title_text=title, 
        title_x=0.5, 
        width=700, 
        height=700,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig=go.Figure(data=[heat], layout=layout)
    st.plotly_chart(fig)

vessel_corr_highlight(vessel_0_filtered, 0.8)
vessel_corr_highlight(vessel_1_filtered, 0.8)
vessel_corr_highlight(vessel_2_filtered, 0.8)
vessel_corr_highlight(vessel_3_filtered, 0.8)

vessel_corr_highlight(vessel_0_filtered, -0.2, color='Reds')
vessel_corr_highlight(vessel_1_filtered, -0.2, color='Reds')
vessel_corr_highlight(vessel_2_filtered, -0.2, color='Reds')
vessel_corr_highlight(vessel_3_filtered, -0.2, color='Reds')

st.write('''
Here, we observe that the highly correlated features are:




```python
highly_corr_features = ['CO2_Emitted', 'NOX_Emitted', 'ME_Consumption', 'ME_Consumption_MGO', 'ME_Consumption_HFO',
                        'Distance', 'Boiler_Consumption_MGO', 'Boiler_Consumption_HFO', 'AE_Consumption_MGO', 'AE_Consumption_HFO', 'Cargo_Mt']
```
And that `'ME_calculated_SFOC'` feature is negatively correlated to some of the features such as `'CO2_Emitted'`, `'Distance'`, etc.
```python
multiscatter_corr_features = ['CO2_Emitted', 'ME_Consumption', 'Distance', 'ME_calculated_SFOC']
```

We can plot the features of interest in a scatter matrix plot to view the correlation of the features.

Before doing so, let's write a function to normalize the feature values into range of 0 and 1.

```python
def normalize(vessel, feature_name):
    max_value = vessel[feature_name].max()
    min_value = vessel[feature_name].min()
    feature_name_norm = feature_name + '_norm'
    vessel[feature_name_norm] = (vessel[feature_name] - min_value) / (max_value - min_value)
    return vessel

Now, we can run through the `normalize` function to get the feature values in a normalized space.
```

         ''')
def normalize(vessel, feature_name):
    max_value = vessel[feature_name].max()
    min_value = vessel[feature_name].min()
    feature_name_norm = feature_name + '_norm'
    vessel[feature_name_norm] = (vessel[feature_name] - min_value) / (max_value - min_value)
    return vessel


highly_corr_features = ['CO2_Emitted', 'NOX_Emitted', 'ME_Consumption', 'ME_Consumption_MGO', 'ME_Consumption_HFO',
                        'Distance', 'Boiler_Consumption_MGO', 'Boiler_Consumption_HFO', 'AE_Consumption_MGO', 'AE_Consumption_HFO', 'Cargo_Mt']


vessels_w_norm_cols = []
for v in vessels_filtered:
    for i in highly_corr_features:
        vessel_norm = normalize(v, i)
    vessels_w_norm_cols.append(vessel_norm) 

vessels_w_norm_cols_group = []
for v in vessels_w_norm_cols:
    df_vessel_group = pd.DataFrame()
    vessel_name = v['IMO'][1]
    v['Date_UTC'] = pd.to_datetime(v['Date_UTC'], infer_datetime_format=True)
    for i in highly_corr_features:
        j = i + '_norm'
        df_vessel_group[j] = v.groupby(v['Date_UTC'].dt.strftime('%Y-%m'))[j].mean()
        df_vessel_group['label'] = vessel_name
    df_vessel_group = df_vessel_group[['label'] + [col for col in df_vessel_group.columns if col != 'label']]
    vessels_w_norm_cols_group.append(df_vessel_group)

    

# Multi corr plot
multiscatter_corr_features = ['CO2_Emitted', 'ME_Consumption', 'Distance', 'ME_calculated_SFOC']
ms_vessels_w_norm_cols = []
for v in vessels_filtered:
    for i in multiscatter_corr_features:
        vessel_norm = normalize(v, i)
    ms_vessels_w_norm_cols.append(vessel_norm) 

ms_vessels_w_norm_cols_group = []
for v in ms_vessels_w_norm_cols:
    ms_df_vessel_group = pd.DataFrame()
    vessel_name = v['IMO'][1]
    v['Date_UTC'] = pd.to_datetime(v['Date_UTC'], infer_datetime_format=True)
    for i in multiscatter_corr_features:
        j = i + '_norm'
        ms_df_vessel_group[j] = v.groupby(v['Date_UTC'].dt.strftime('%Y-%m'))[j].mean()
        ms_df_vessel_group['label'] = vessel_name
    ms_df_vessel_group = ms_df_vessel_group[['label'] + [col for col in ms_df_vessel_group.columns if col != 'label']]
    ms_vessels_w_norm_cols_group.append(ms_df_vessel_group)

for v in ms_vessels_w_norm_cols_group:
    #st.write(v)
    vessel_name = v['label'][1]
    st.write('### ' + vessel_name + ' scatter matrix on the features of interest')
    fig = px.scatter_matrix(v[v.columns[1:]])
    fig.update_layout(title=vessel_name,
                  dragmode='select',
                  width=900,
                  height=900,
                  hovermode='closest')
    st.plotly_chart(fig)

df_scatter_plot = pd.concat(ms_vessels_w_norm_cols_group, axis=0)
 
st.write('''
From the above scatter plots for each of the vessels, we observe that the 
The negative correlation between fuel efficiency and distance, ME consumption, CO2 emssions are observable in vessels 0 and 1 (vessels 2 and 3 not as prominent).

We can also tell from the scatter matrix that the correlation between CO2 emssions, ME consumption, and distance are highly correlated.

To get a better perspective and visualization, let's put it together and overlay the plots for the different vessels!
         ''')

st.write('### Scatter matrix on the features of interest (All 4 vessels)')
fig = px.scatter_matrix(df_scatter_plot,
    color="label")
fig.update_layout(#title=title,
                  dragmode='select',
                  width=1000,
                  height=1000,
                  hovermode='closest')
st.plotly_chart(fig)



st.write('''
### 
From the scatter plot above (for all vessels), we can see that the high correlation between distance, ME consumption, and CO2 emssion
is consistent for all the vessels.
The negative correlation of fuel efficiency against the rest of the features are prominent in vessels 0 and 1.
To summarize, the higher the distance travelled, the higher the ME consumption, the higher the CO2 emissions, the lower the fuel efficiency. 

### Dimensionality reduction
Finally, we put all the normalized values of the selected features into a list for each data entry.

The chosen features are:
```python
highly_corr_features = ['CO2_Emitted', 'NOX_Emitted', 'ME_Consumption', 'ME_Consumption_MGO', 'ME_Consumption_HFO',
                        'Distance', 'Boiler_Consumption_MGO', 'Boiler_Consumption_HFO', 'AE_Consumption_MGO', 'AE_Consumption_HFO', 'Cargo_Mt']
```
We take the mean per month for each feature and place it in a new column called `'collated_features'`.
Then, apply dimensionality reduction to visualize in 2D using PCA and t-SNE dimensionality reduction algorithms.

For the t-SNE plot, we ran the collated features into a PCA algorithm first, extracting 5 principal components `'n_components=5'`, before passing it
into the t-SNE algorithm to further reduce into 2D for visualization purposes.
         ''')


# back
df_collated_feat = []
for v in vessels_w_norm_cols_group:
    #st.write(v)
    #st.write(v.shape)
    v['collated_features'] = v[v.columns[-len(highly_corr_features):]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )
    #st.write(v)
    #st.write(v.shape)
    df_collated_feat.append(v)

df_full = pd.concat(df_collated_feat, axis=0)
#st.write(df_full)
#st.write(df_full.shape)

df_full_pca = df_full[['label', 'collated_features']]

#st.write(df_full_pca)
#st.write(df_full_pca.shape)

df_single_feat = df_full_pca['collated_features']
#st.write(df_single_feat)
#st.write(df_single_feat.shape)
labelsDF = df_full_pca['label']
#st.write(labelsDF)
#st.write(labelsDF.shape)
df_single_feat = [x.split(',') for x in df_single_feat]

def floatify(x):
    try:
        return float(x)
    except ValueError:
        return x

df_single_feat = [[floatify(x) for x in row] for row in df_single_feat]


# PCA
st.write('''
### Dimensionality reduction - 2D visualization of the extracted principle components from the highly correlated features.
- `'n_components=2'`
         ''')
pca = PCA(n_components=2)
dataPCA = pca.fit_transform(df_single_feat)
dataPCA = np.vstack((dataPCA.T, labelsDF)).T
dataPCA = pd.DataFrame(data=dataPCA, columns=("Dim_1", "Dim_2", "Vessel"))
#fig, ax = plt.subplots(figsize=(16,9))
fig = px.scatter(dataPCA, x='Dim_1', y='Dim_2', color='Vessel')
#sns.scatterplot(data=dataPCA,x='Dim_1',y='Dim_2',hue='label',legend="full",alpha=0.5)
#st.pyplot()
st.plotly_chart(fig)


st.write('''
From the plot above, we can see that vessels 0 and 1 are closely aligned in the reduced 2D space - where they are commonly 
located in quadrants 2 and 3 (negative x-axis).
And vessels 2 and 3 are closely aligned in the reduced 2D space and are commonly located in quadrants 1 and 4 (positive x-axis).

### Dimensionality reduction - 2D visualization using t-SNE algorithm.
- PCA, `'n_components=5'`
- t-SNE. `'n_components=2'`
         ''')

# tSNE
pca = PCA(n_components=5)
dataPCA = pca.fit_transform(df_single_feat)

model = TSNE(n_components=2)
tsne_data = model.fit_transform(dataPCA)

tsne_data = np.vstack((tsne_data.T, labelsDF)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "Vessel"))
#fig, ax = plt.subplots(figsize=(16,9))
#sns.scatterplot(data=tsne_df,x='Dim_1',y='Dim_2',hue='label',legend="full",alpha=0.5)
fig = px.scatter(tsne_df, x='Dim_1', y='Dim_2', color='Vessel', color_discrete_sequence=px.colors.qualitative.D3)
fig.update_traces(marker=dict(size=6),
                  selector=dict(mode='markers'))
st.plotly_chart(fig)

st.write('''
From the above t-SNE plot, we can see that the data points of vessels 0 and 1 are closely aligned
and vessels 2 and 3 are closely aligned, in the reduced 2D space.

### Conclusion
After much of the analysis, I conclude that vessels 0 and 1 are sister vessels and vessels 2 and 3 are sister vessels.
I also conclude that, due to the negatively correlated feature of ME SFOC, the fuel efficiency of drops in accordance with the distance travelled 
as well as the other highly correlated features with distance. 
The previously highlighted hypotheses remain unchanged.
Finally, all the plots are interactive and are able to be enlarged for better viewing.

Thank you for reading!

Although there are many more features and data to explore, as well as more methods and statistical models to apply, given the time limit,
I had fun working on the analysis!

         ''')