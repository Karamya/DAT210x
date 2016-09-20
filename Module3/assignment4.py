import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')
os.chdir('D:/Data analysis/data/DAT210x/Module3/Datasets')
#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
wheat_data = pd.read_csv('wheat.data', header = 0)
#print(wheat_data)



#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
# .. your code here ..

wheat_data.drop(labels = ['id', 'area', 'perimeter'], axis = 1, inplace = True)
print(wheat_data)
#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..
plt.figure()
parallel_coordinates(wheat_data, 'wheat_type', alpha = 0.4 )


plt.show()


