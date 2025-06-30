import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

''' 1. loading data '''
data_path_NV =r"D:\PY_Project\Paper1\Paper1_data\Shapely\EI_shape_NV.csv"
data_path_SiV =r"D:\PY_Project\Paper1\Paper1_data\Shapely\EI_shape_SiV.csv"
data_path_GeV =r"D:\PY_Project\Paper1\Paper1_data\Shapely\EI_shape_GeV.csv"

a = pd.read_csv(data_path_NV)  
b = pd.read_csv(data_path_SiV) 
c = pd.read_csv(data_path_GeV)  

 
NV_EnergyDoseTemp = np.around(np.array(a),3)[...,0:3]
NV_Time = np.around(np.array(a),3)[...,3]

SiV_EnergyDoseTemp = np.around(np.array(b),3)[...,0:3]
SiV_Time = np.around(np.array(b),3)[...,3]

GeV_EnergyDoseTemp = np.around(np.array(c),3)[...,0:3]
GeV_Time = np.around(np.array(c),3)[...,3]

X = pd.DataFrame(NV_EnergyDoseTemp, columns=['Irridation', 'Dose', 'Annealing']) 
y = pd.DataFrame(NV_Time, columns=['Time'])

X1 = pd.DataFrame(SiV_EnergyDoseTemp, columns=['Irridation', 'Dose', 'Annealing']) 
y1 = pd.DataFrame(SiV_Time, columns=['Time'])

X2 = pd.DataFrame(GeV_EnergyDoseTemp, columns=['Irridation', 'Dose', 'Annealing']) 
y2 = pd.DataFrame(GeV_Time, columns=['Time'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

# define model
model = DecisionTreeRegressor(
                              criterion = 'friedman_mse',   
                              splitter='random', 
                              max_features = None,
                              max_depth = 100,  # 100
                              min_samples_split = 2,
                              min_samples_leaf = 1,
                              min_weight_fraction_leaf = 0,
                              max_leaf_nodes = None, 
                              random_state = 42         
                              ) 

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_train_NV = model.predict(X_train)

model.fit(X_train1, y_train1)
y_pred1 = model.predict(X_test1)
y_train_SiV = model.predict(X_train1)

model.fit(X_train2, y_train2)
y_pred2 = model.predict(X_test2)
y_train_GeV = model.predict(X_train2)

explainer1 = shap.Explainer(model)
explainer2 = shap.Explainer(model)
explainer3 = shap.Explainer(model)

shap_values = explainer1(X_train)
shap_values1 = explainer2(X_train1)
shap_values2 = explainer3(X_train2)


fig = plt.figure()
ax0 = fig.add_subplot(131)
shap.plots.bar(shap_values, show = False)
ax1 = fig.add_subplot(132)
shap.plots.bar(shap_values1, show = False)
ax2 = fig.add_subplot(133)
shap.plots.bar(shap_values2, show = False)

font_size = 24
plt.gcf().set_size_inches(21,5)

 #  Choosing Bar, Violin, Heatmap and deleting the "#"
######### 1. Bar ############## 
ax0.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax0.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax0.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax0.text(1., 0.02, r'$NV$', transform=ax0.transAxes, ha='right', fontsize=18, color='k')

ax1.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax1.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax1.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax1.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax1.text(1., 0.02, r'$SiV$', transform=ax1.transAxes, ha='right', fontsize=18, color='k')

ax2.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax2.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax2.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax2.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax2.text(1., 0.02, r'$GeV$', transform=ax2.transAxes, ha='right', fontsize=18, color='k')

plt.tight_layout()
fig.savefig(r'D:\PY_Project\Paper1\Paper1_data\Shapely\EI_DTR_Bar.svg', format='svg', transparent=True, bbox_inches='tight')
plt.show()

######### 2. Violin ########### 
# ax0.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax0.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.text(1., 0.02, r'$NV$', transform=ax0.transAxes, ha='right', fontsize=18, color='k')

# ax1.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax1.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax1.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.text(1., 0.02, r'$SiV$', transform=ax1.transAxes, ha='right', fontsize=18, color='k')

# ax2.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax2.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax2.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.text(1., 0.02, r'$GeV$', transform=ax2.transAxes, ha='right', fontsize=18, color='k')

# plt.tight_layout()
# fig.savefig(r'D:\PY_Project\Paper1\Paper1_data\Shapely\EI_DTR_Violin.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.show()

######### 3. Heatmap #############
# ax0.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax0.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.text(1.2, -0.1, r'$NV$', transform=ax0.transAxes, ha='right', fontsize=18, color='k')

# ax1.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax1.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax1.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.text(1.2, -0.1, r'$SiV$', transform=ax1.transAxes, ha='right', fontsize=18, color='k')

# ax2.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax2.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax2.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.text(1.2, -0.1, r'$GeV$', transform=ax2.transAxes, ha='right', fontsize=18, color='k')

# plt.tight_layout()
# fig.savefig(r'D:\PY_Project\Paper1\Paper1_data\Shapely\EI_DTR_Heatmap.svg', format='svg', transparent=True, bbox_inches='tight')
# plt.show()