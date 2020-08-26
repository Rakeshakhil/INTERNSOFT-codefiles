import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



rak=pd.read_csv('data.csv',usecols=[0,1,2,3,4])
DPOHL_avg=rak[['Price','Open','High','Low']].mean(axis=1)


y=np.arange(1,len(rak)+1,1)
plt.plot(y,DPOHL_avg,'r',label= 'plot')
