import pandas as pd
import os
# TODO: Load up the 'tutorial.csv' dataset
#
# .. your code here ..

os.chdir('D:\Data analysis\data\DAT210x\Module2\Datasets')
df = pd.read_csv('tutorial.csv')

# TODO: Print the results of the .describe() method
#
# .. your code here ..
print(df)
print(df.describe())

# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
# .. your code here ..

print(df.loc[2:4, 'col3'])
