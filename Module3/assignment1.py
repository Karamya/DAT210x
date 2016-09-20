import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
# Look pretty...
matplotlib.style.use('ggplot')

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
os.chdir('D:/Data analysis/data/DAT210x/Module3/Datasets')
wheat_data  = pd.read_csv('wheat.data', header = 0, index_col = 0)

#print(wheat_data)
#
# TODO: Create a slice of your dataframe (call it s1)
# that only includes the 'area' and 'perimeter' features
# 
# .. your code here ..
s1 = wheat_data[['area', 'perimeter']]
print(s1.columns)
#
# TODO: Create another slice of your dataframe (call it s2)
# that only includes the 'groove' and 'asymmetry' features
# 
# .. your code here ..

s2 = wheat_data[['groove', 'asymmetry']]
print(s2.columns)
#
# TODO: Create a histogram plot using the first slice,
# and another histogram plot using the second slice.
# Be sure to set alpha=0.75
# 
# .. your code here ..
s1.plot.hist(alpha = 0.75, binning = 2)
s2.plot.hist(alpha = 0.75)

plt.show()
