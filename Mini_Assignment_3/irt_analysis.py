
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

irt_res = pd.read_pickle('irt_fit.pkl')
items_data = pd.read_pickle('items_data.pkl')

items_time = items_data.groupby('item_number_filtered').agg({'time' : 'mean'})
items_time.sort_values('item_number_filtered', inplace = True)
times = items_time['time']

avg_res = irt_res.agg('mean')

avg_res_alpha = avg_res.loc[avg_res.index.str.contains('alpha')]
abilities = avg_res_alpha

difficulties = avg_res.loc[avg_res.index.str.contains('beta.')]

plt.scatter(difficulties, times)
#plt.xlim([-0.25, 0.25])
#plt.ylim([0, 15])
plt.show()

print(np.correlate(difficulties, times))