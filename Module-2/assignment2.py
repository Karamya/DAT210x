# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:43:01 2016

@author: Karthick Perumal
"""

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
wheat_data = pd.read_csv('wheat.data', header = 0, index_col = 0)

print(wheat_data)
#
# TODO: Create a 2d scatter plot that graphs the
# area and perimeter features
# 
# .. your code here ..

wheat_data.plot.scatter(x = 'area', y = 'perimeter', marker = '^', s = 50)
#
# TODO: Create a 2d scatter plot that graphs the
# groove and asymmetry features
# 
# .. your code here ..

wheat_data.plot.scatter(x = 'groove', y = 'asymmetry', marker = '.', s = 100)
#
# TODO: Create a 2d scatter plot that graphs the
# compactness and width features
# 
# .. your code here ..

wheat_data.plot.scatter(x = 'compactness', y = 'width', marker = 'o', s = 50)

# BONUS TODO:
# After completing the above, go ahead and run your program
# Check out the results, and see what happens when you add
# in the optional display parameter marker with values of
# either '^', '.', or 'o'.


plt.show()
