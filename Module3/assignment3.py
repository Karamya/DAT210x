import pandas as pd
import os
# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
# .. your code here ..
os.chdir('D:/Data analysis/data/DAT210x/Module2/Datasets/')

df = pd.read_csv('servo.data', names =['motor', 'screw', 'pgain', 'vgain', 'class'])
#print(df)
# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
# .. your code here ..
print(len(df[df.vgain == 5]))

# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
# .. your code here ..

print(len(df[(df.motor == 'E') & (df.screw == 'E')]))

# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
# .. your code here ..
y = df[df.pgain ==4]
print(y.vgain.mean())                 #print(df[df.pgain ==4].describe())  or print(df[df.pgain ==4].vgain.mean())

# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!



print(df.dtypes)