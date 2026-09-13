import pandas as pd

df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding='latin-1', header=None)
df.columns = ['target','id','date','flag','user','text']
df['target'] = df['target'].map({0: 0, 4: 1})  # binary
df[['target','text']].head()