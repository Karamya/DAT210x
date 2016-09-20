import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
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


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
# .. your code here ..
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(wheat_data.area, wheat_data.perimeter, wheat_data.asymmetry, color = 'r', marker = 'o')
fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
# .. your code here ..
ax1 = fig.add_subplot(111, projection = '3d')
ax1.set_xlabel('width')
ax1.set_ylabel('groove')
ax1.set_zlabel('length')
ax1.scatter(wheat_data.width, wheat_data.groove, wheat_data.length, color = 'green', marker = '^')

plt.show()





fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
# .. your code here ..


plt.show()


