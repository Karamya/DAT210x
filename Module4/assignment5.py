import pandas as pd
import numpy as np
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import os, glob2
# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []
colors = []
#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
os.chdir('D:/Data analysis/data/DAT210x/Module4/Datasets/ALOI/32')
for file in glob2.glob('*.png'):
    img = misc.imread(file).reshape(-1)  
    samples.append(img)
   
print(len(samples))
for i in range(len(samples)):
    colors.append('b')

    
#print(samples)    

#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
os.chdir('D:/Data analysis/data/DAT210x/Module4/Datasets/ALOI/32i')
for file in glob2.glob('*.png'):
    img = misc.imread(file).reshape(-1)
    samples.append(img)

print(len(samples))


temp = len(samples)-len(colors)
for i in range(temp):
    colors.append('r')

#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 

df = pd.DataFrame(samples)
#print(df)

#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors = 6, n_components = 3)
z = iso.fit_transform(df)

#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
def Plot2D(T, title, x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    x_size = (max(T[:,x])-min(T[:,x]))*0.08
    y_size = (max(T[:,y])-min(T[:,y]))*0.08
    ax.scatter(T[:,x], T[:,y], marker = 'o', c = colors, alpha = 0.7)
    return 

    

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
def Plot3D(T, title, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    
    x_size = (max(T[:,x])-min(T[:,x]))*0.08
    y_size = (max(T[:,y])-min(T[:,y]))*0.08
    z_size = (max(T[:,z])-min(T[:,z]))*0.08
    ax.scatter(T[:,x], T[:,y],T[:,z], marker = 'o', c = colors, alpha = 0.7)
    return




Plot2D(z, '2D Isomap transformed data ', 0 , 1 )
Plot3D(z, '2D Isomap transformed data ', 0 , 1, 2 )
plt.show()

