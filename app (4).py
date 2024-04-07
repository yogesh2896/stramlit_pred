import streamlit as st

# url=input("Enter the url of data")
def activate(c1,c2):
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from pmdarima import auto_arima
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  import warnings
  from sklearn import metrics
  warnings.filterwarnings("ignore", category=DeprecationWarning)



  import pandas as pd

  df=pd.read_csv(url)
  df=df[[c1,c2]]
  df.columns=['date','value']

  df['date']=pd.to_datetime(df['date'])
  df['date'] = df['date'].apply(lambda x: x.replace(day=1))
  df=df.groupby('date').sum('value')












  autoari=auto_arima(df,seasonal=True,maxiter=300,suppress_warnongs=True)
  l=round(len(df)/3)
  train=df[:-l]
  test=df[-l:]
  acc_score=[]
  params=[]
  for i in range(0,4):
    p=autoari.order[0]+i
    d=autoari.order[1]+i
    q=autoari.order[2]+i
    model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
    model=model.fit()

    #accuracy checking
    acc_model=model.get_forecast(len(test))
    # print(type(acc_model.predicted_mean))
    # print(type(test['value']))
    a=metrics.mean_absolute_error(test['value'],acc_model.predicted_mean)
    params.append(i)
    acc_score.append(a)
  acc_df=pd.DataFrame(params,acc_score)
  # acc_df.columns=['i','a']
  acc_df=acc_df.reset_index()
  acc_df.columns=['value','params']
  acc_df=acc_df.sort_values(by='params')
  best_param=acc_df['params'][1]
  print(best_param)

  p=autoari.order[0]+best_param
  d=autoari.order[1]+best_param
  q=autoari.order[2]+best_param
  model=SARIMAX(train,order=(p,d,q),seasonal_order=(p,d,q,12),trend=None)
  model=model.fit()



  trained_model=model.get_forecast(len(test)+24)

  predictions=trained_model.predicted_mean

  predictions=pd.DataFrame(predictions)



  predictions=predictions.reset_index()

  df=df.reset_index()
  predictions.columns=df.columns
  df=df.merge(predictions,how='outer',on='date')

  sns.lineplot(x=df['date'],y=df['value_x'])
  sns.lineplot(x=df['date'],y=df['value_y'])
  # print(df)

  st.write('Forecasting Visual')
  df.set_index('date',inplace=True)
  df.columns=['Actual','Predicted']
  st.line_chart(df)

st.write('Welcome to the Predictive Model by Enoah')
c1=st.text_input('Enter the date column name')
c2=st.text_input('Enter the target column name')
# url=st.text_input('Enter the URL')

url=st.file_uploader(label="Upload your data file",type='csv')
if st.button('Predict'):
  activate(c1,c2)
