# %%
import os
import pandas as pd
import re

# Define the root directory path
root_directory = "/home/sem23/fastrocs/rp1-sem-egbers/DrugEx_script/"

# Get the list of directories containing the files
directories = []

# Iterate through all folders in the root directory
for dirpath, dirnames, filenames in os.walk(root_directory):
    for dirname in dirnames:
        # Check if the directory name contains "NS" and "eps" and doesn't contain "bs256"
        if "ns" in dirname and "eps" in dirname: #and "bs256" not in dirname:
            directories.append(os.path.join(dirpath, dirname))

# Sort the directories based on 'ns' and 'eps' values from low to high
directories.sort(key=lambda x: (int(x.split('_')[2][2:]), float(x.split('_')[3][3:]),int(x.split('_')[4][4:])))

# Create a dictionary to store the IDs
id_counter = 1
id_mapping = {}

# Iterate through each directory and assign an ID
# for directory in directories:
#     id_mapping[directory] = f"ID_{id_counter}" # parse information with ns, eps and bs 
#     id_counter += 1
# Iterate through each directory and assign an ID
for directory in directories:
    ns_value = directory.split('_')[2][2:]  
    eps_value = directory.split('_')[3][3:] 
    bs_value = directory.split('_')[4]      
    id_mapping[directory] = f"ID_{id_counter}_ns{ns_value}_eps{eps_value}_{bs_value}"
    id_counter += 1

# Remove RL in name and add whole name to each data input 
# Iterate through each directory, locate the TSV files, and read them into a DataFrame
dataframes = []
fit_dataframes = []


for directory, directory_id in id_mapping.items():
    # Construct the full file path
    file_path = os.path.join(directory, "CCR2_agent_smiles.tsv")
    fit_path = os.path.join(directory, "CCR2_agent_fit.tsv") 
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t')
        df1 = pd.read_csv(fit_path, sep='\t')
        # Assign directory ID to each row in the DataFrame
        df['ID'] = directory_id
        df1['ID'] = directory_id
        dataframes.append(df) #df1
        fit_dataframes.append(df1) 
    
#Concatenate these DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)
fit_df = pd.concat(fit_dataframes, ignore_index=True)

for directory, directory_id in id_mapping.items():
    # Extract NS and eps values from the directory path
    ns_value = directory.split('_')[2]
    eps_value = directory.split('_')[3]
    bs_value = directory.split('_')[4]
    print(f"ID: {directory_id}, ns: {ns_value}, eps: {eps_value}, bs: {bs_value}")



# %% [markdown]
# ID selection and filtering

# %%
# Initialize a list to store IDs with bs256
bs256_ids = []

# Iterate over the keys of id_mapping dictionary
for directory, directory_id in id_mapping.items():
    # Check if the directory ID contains 'bs256'
    if 'bs256' in directory_id:
        # Append the directory ID to the list
        bs256_ids.append(directory_id)

# bs256_ids now contains all the IDs with 'bs256'


# %%
# Define the list of IDs to iterate over
target_data = bs256_ids
# Dictionary to store the filtered DataFrames for the last epoch of each ID
last_epoch_dfs = {}

# Columns to keep
columns_to_keep = ['SMILES', 'ROCS', 'SA', 'Epoch']

# Iterate over each ID
for id_ in target_data:
    # Filter the DataFrame for the current ID
    id_df = fit_df[fit_df['ID'] == id_]
    
    # Find the maximum epoch value for the current ID
    max_epoch_value = id_df['best_epoch'].max()
    
    # Filter out the rows with the highest epoch value in the 'Epoch' column
    filtered_df = combined_df[combined_df['Epoch'] == max_epoch_value]
  
    # Keep only the specified columns
    last_epoch_df = filtered_df[columns_to_keep]
    
    # Store the DataFrame for the last epoch in the dictionary
    last_epoch_df['ID'] = id_
    last_epoch_dfs[id_] = last_epoch_df
    
epoch_all = pd.concat(last_epoch_dfs.values())
print(epoch_all)
    

# Accessing individual DataFrames
# id_6_df = last_epoch_dfs['ID_6']
# id_7_df = last_epoch_dfs['ID_7']




# %% [markdown]
# ROCS per epoch Visualisation

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Group by 'Epoch' and calculate the mean of 'ROCS' for each epoch
average_rocs_per_epoch = combined_df.groupby(['ID','Epoch'])['ROCS'].mean().reset_index()

# Get the unique IDs and sort them
unique_ids = average_rocs_per_epoch['ID'].unique()
sorted_ids = sorted(unique_ids, key=lambda x: int(x.split('_')[1]))

# Plotting
sns.lineplot(data=average_rocs_per_epoch, x='Epoch', y='ROCS', hue='ID', hue_order=sorted_ids)

plt.xlabel('Epoch')
plt.ylabel('Average ROCS')
plt.title('Average ROCS per Epoch for each ID')

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()


# %% [markdown]
# Scatterplot overview of selected ID's to visualise chemical space

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from qsprpred.data.descriptors.fingerprints import MorganFP
from scaffviz.clustering.manifold import TSNE
from qsprpred.data.sources.papyrus import Papyrus
from scaffviz.depiction.plot import Plot
from qsprpred.data.tables.mol import MoleculeTable

# Assuming you have a DataFrame called 'dataset' with the required data
# Add Morgan fingerprints to the dataset
dataset = MoleculeTable('ID_best_epoch', epoch_all, overwrite=True)
dataset.addDescriptors([MorganFP(radius=3, nBits=2048)], recalculate=False)
print(dataset)
# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(dataset.getDescriptors())  

 




# %%
tsne_data[:,0]

# %%
# Create a DataFrame with the t-SNE results
dataset.addProperty("TSNE_1", tsne_data[:,0])
dataset.addProperty("TSNE_2", tsne_data[:,1])

tsne_df = dataset.getDF() 

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x='TSNE_1', y='TSNE_2', hue='ID', alpha=0.3)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE plot')

# Show the plot
# plt.legend(title='ID', loc='upper right')
plt.legend(title='ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
tsne_df['ID'].str.startswith('ID_6')

# %%
tsne_df_1 = tsne_df[tsne_df["ID"].str.startswith("ID_6")|tsne_df['ID'].str.startswith('ID_8')|tsne_df['ID'].str.startswith('ID_14')]

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df_1, x='TSNE_1', y='TSNE_2', hue='ID', alpha=0.3)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE plot')

# Show the plot
# plt.legend(title='ID', loc='upper right')
plt.legend(title='ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Define the bundles of IDs
id_bundles = [
    ['ID_1', 'ID_2', 'ID_3','ID_4','ID_5','ID_7','ID_9','ID_11','ID_13','ID_15'],
    ['ID_6', 'ID_7', 'ID_8','ID_9','ID_10','ID_11','ID_12','ID_13','ID_14','ID_15'],
    ['ID_6', 'ID_8', 'ID_10','ID_12','ID_14']
]

# Define the order of IDs
id_order = [f'ID_{i}' for i in range(1, 16)]

# Iterate over each bundle of IDs
for i, ids in enumerate(id_bundles, 1):
    # Filter the DataFrame to include only the specified IDs
    filtered_data = average_rocs_per_epoch[average_rocs_per_epoch['ID'].isin(ids)]
    
    # Set the categorical order of the 'ID' column
    filtered_data['ID'] = pd.Categorical(filtered_data['ID'], categories=id_order, ordered=True)
    
    # Create a new figure for each bundle of IDs
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.lineplot(data=filtered_data, x='Epoch', y='ROCS', hue='ID')
    plt.xlabel('Epoch')
    plt.ylabel('Average ROCS')
    plt.title(f'Bundle {i}: Average ROCS per Epoch')
    plt.legend(title='ID')
    plt.show()



# %% [markdown]
# Interactive plot calculated with moldescriptors

# %%
import pandas as pd 

 
from qsprpred.data.chem.scaffolds import BemisMurcko
from qsprpred.data.descriptors.fingerprints import MorganFP
from scaffviz.clustering.manifold import TSNE
from qsprpred.data.sources.papyrus import Papyrus
from scaffviz.depiction.plot import Plot
from qsprpred.data.tables.mol import MoleculeTable


dataset = MoleculeTable('ID_6', id_6_df, overwrite=True)

# add generic scaffolds to the data set
# these will be used to group the molecules
dataset.addScaffolds([BemisMurcko()])

# add Morgan fingerprints to the data set
# these will be used to calculate the t-SNE embedding in 2D
dataset.addDescriptors([MorganFP(radius=3, nBits=2048)], recalculate=False)

# make an interactive plot that will use t-SNE to embed the data set in 2D
# (all available descriptors in the data set will be used, not just selected features)
plt = Plot(TSNE(perplexity=50))
plt.plot(
    dataset,
    recalculate=False,
    color_by='ROCS', 
    card_data=["ROCS","SA"],  # what to show on the molecule cards
    title_data='Molecules'   # Data to show in the title of the molecule cards
)

# %%
dataset.getDF()
sns.plot()

# %% [markdown]
# Mols2grid viewers

# %%
# I want to portray here the top 50 smiles of ID1, ID6 and ID7

import pandas as pd 
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw
import mols2grid


# specific_variable_name = "df_ranked_1"

# # Check if the variable name exists in the dictionary
# if specific_variable_name in result_dataframes:
#     specific_dataframe = result_dataframes[specific_variable_name]

mols2grid.display(id_1_df.head(50), smiles_col='SMILES', subset=['ROCS','SA'], tooltip=["SMILES", "ROCS", "SA"])




# %%
# I want to portray here the top 50 smiles of ID1, ID6 and ID7

import pandas as pd 
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw
import mols2grid


# specific_variable_name = "df_ranked_1"

# # Check if the variable name exists in the dictionary
# if specific_variable_name in result_dataframes:
#     specific_dataframe = result_dataframes[specific_variable_name]

mols2grid.display(id_6_df, smiles_col='SMILES', subset=['ROCS','SA'], tooltip=["SMILES", "ROCS", "SA"])




# %%
# I want to portray here the top 50 smiles of ID1, ID6 and ID7

import pandas as pd 
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw
import mols2grid



mols2grid.display(id_7_df, smiles_col='SMILES', subset=['ROCS','SA'], tooltip=["SMILES", "ROCS", "SA"])




# %% [markdown]
# Visualisation of Desired ratio progression

# %%
import seaborn as sns
import matplotlib.pyplot as plt


unique_ids = fit_df['ID'].unique()
sorted_ids = sorted(unique_ids, key=lambda x: int(x.split('_')[1]))

# Plotting
sns.lineplot(data=fit_df, x='Epoch', y='desired_ratio', hue='ID', hue_order=sorted_ids)

plt.xlabel('Epoch')
plt.ylabel('Desired Ratio')
plt.title('Desired Ratio per Epoch for each ID')

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Example list of IDs
ids_list = ['ID_1', 'ID_2', 'ID_3','ID_4','ID_5','ID_6','ID_7','ID_8','ID_9','ID_10','ID_11','ID_12','ID_13','ID_14','ID_15']

# Create an empty dictionary to store dataframes for each ID
id_dataframes = {}

# Iterate over each ID
for id_ in ids_list:
    # Filter fit_df for the current ID
    filtered_df = fit_df[fit_df['ID'] == id_]
    # Store the filtered dataframe for the current ID in the dictionary
    id_dataframes[id_] = filtered_df

# Iterate over each ID and plot the data
for id_, df in id_dataframes.items():
    # Plot the desired columns for the current ID
    df[['desired_ratio']].plot.line()
    # Set plot title and labels
    plt.title(f"Plot for ID: {id_}")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # Show the plot
    plt.show()


# %%
import pandas as pd
df_info = pd.read_csv(f'DrugEx_script/RL_ns10000_eps0.2_bs128_pt30_ne1000/CCR2_agent_fit.tsv', sep='\t')
df_info

# %%
df_info[['desired_ratio', 'loss_train', 'avg_amean']].plot.line()


