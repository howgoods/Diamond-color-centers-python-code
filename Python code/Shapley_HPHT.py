import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

''' 1. loading data '''
data_path_NV  = r"D:\PY_Project\Paper1\Paper1_data\Shapely\HPHT_shape_NV.csv"
data_path_SiV = r"D:\PY_Project\Paper1\Paper1_data\Shapely\HPHT_shape_SiV.csv"
data_path_GeV = r"D:\PY_Project\Paper1\Paper1_data\Shapely\HPHT_shape_GeV.csv"
data_path_SnV = r"D:\PY_Project\Paper1\Paper1_data\Shapely\HPHT_shape_SnV.csv"

a = pd.read_csv(data_path_NV)  
b = pd.read_csv(data_path_SiV)  
c = pd.read_csv(data_path_GeV)  
d = pd.read_csv(data_path_SnV) 

NV_TempPressTime = np.around(np.array(a),3)[...,0:3]
NV_Size = np.around(np.array(a),3)[...,3]

SiV_TempPressTime = np.around(np.array(b),3)[...,0:3]
SiV_Size = np.around(np.array(b),3)[...,3]

GeV_TempPressTime = np.around(np.array(c),3)[...,0:3]
GeV_Size = np.around(np.array(c),3)[...,3]

SnV_TempPressTime = np.around(np.array(d),3)[...,0:3]
SnV_Size = np.around(np.array(d),3)[...,3]

X = pd.DataFrame(NV_TempPressTime, columns=['Pressure', 'Temperature', 'Time'])  	
y = pd.DataFrame(NV_Size, columns=['Size'])

X1 = pd.DataFrame(SiV_TempPressTime, columns=['Pressure', 'Temperature', 'Time'])  	
y1 = pd.DataFrame(SiV_Size, columns=['Size'])

X2 = pd.DataFrame(GeV_TempPressTime, columns=['Pressure', 'Temperature', 'Time'])  	
y2 = pd.DataFrame(GeV_Size, columns=['Size'])

X3 = pd.DataFrame(SnV_TempPressTime, columns=['Pressure', 'Temperature', 'Time'])  	
y3 = pd.DataFrame(SnV_Size, columns=['Size'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3, random_state=42)

# define models
model = DecisionTreeRegressor(
                              criterion = 'friedman_mse',   
                              splitter = 'random', 
                              max_features = None,
                              max_depth = None,  # 100 None
                              min_samples_split = 20,  
                              min_samples_leaf = 1,
                              min_weight_fraction_leaf = 0,
                              max_leaf_nodes = None, 
                              random_state = 10  
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

model.fit(X_train3, y_train3)
y_pred3 = model.predict(X_test3)
y_train_SnV = model.predict(X_train3)

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_train)
shap_values1 = explainer(X_train1)
shap_values2 = explainer(X_train2)
shap_values3 = explainer(X_train3)

fig = plt.figure()
ax0 = fig.add_subplot(1,4,1)
shap.plots.bar(shap_values, show = False)
ax1 = fig.add_subplot(1,4,2)
shap.plots.bar(shap_values1, show = False)
ax2 = fig.add_subplot(1,4,3)
shap.plots.bar(shap_values2, show = False)
ax3 = fig.add_subplot(1,4,4)
shap.plots.bar(shap_values3, show = False)

font_size = 24
plt.gcf().set_size_inches(25,5)

# Choosing Bar, Violin, Heatmap and deleting the "#"
######### 1. Bar ############## 
ax0.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax0.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax0.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax0.text(1, 0.02, r'$NV$', transform=ax0.transAxes, ha='right', fontsize=18, color='k')

ax1.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax1.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax1.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax1.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax1.text(1, 0.02, r'$SiV$', transform=ax1.transAxes, ha='right', fontsize=18, color='k')

ax2.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax2.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax2.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax2.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax2.text(1, 0.02, r'$GeV$', transform=ax2.transAxes, ha='right', fontsize=18, color='k')

ax3.set_xlabel(f"Mean(Shap value)", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
ax3.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
ax3.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax3.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
ax3.text(1, 0.02, r'$SnV$', transform=ax3.transAxes, ha='right', fontsize=18, color='k')

plt.tight_layout()
fig.savefig(r'D:\PY_Project\Paper1\Paper1_data\Shapely\HPHT_DTR_Bar.svg', format='svg', transparent=True, bbox_inches='tight')
plt.show()

######### 2. Violin ########### 
# ax0.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax0.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.text(1, 0.02, r'$NV^{-}$', transform=ax0.transAxes, ha='right', fontsize=18, color='k')
# # axax0 = ax0.twiny()
# # axax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=0)

# ax1.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax1.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax1.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.text(1, 0.02, r'$SiV^{-}$', transform=ax1.transAxes, ha='right', fontsize=18, color='k')

# ax2.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax2.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax2.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.text(1, 0.02, r'$GeV^{-}$', transform=ax2.transAxes, ha='right', fontsize=18, color='k')

# ax3.set_xlabel(f"Impact of Shap value", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax3.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=0)
# ax3.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax3.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax3.text(1, 0.02, r'$SnV^{-}$', transform=ax3.transAxes, ha='right', fontsize=18, color='k')

# plt.tight_layout()
# fig.savefig(r'D:\PY_Project\Paper1\Paper1_fig\HPHT_ShapTree_Violin.png', transparent=True)
# plt.show()

######### 3. Heatmap ############# 
# ax0.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax0.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax0.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax0.text(1, 0.02, r'$NV^{-}$', transform=ax0.transAxes, ha='right', fontsize=18, color='k')
# # ax0.set_xlim([0,len(y_train)]) 
# # ax0.set_xticks(range(0,len(y_train),50))

# ax1.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax1.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax1.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax1.text(1, 0.02, r'$SiV^{-}$', transform=ax1.transAxes, ha='right', fontsize=18, color='k')
# # ax1.set_xlim([0,len(y_train1)]) 
# # ax1.set_xticks(range(0,len(y_train1),100))

# ax2.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax2.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax2.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax2.text(1, 0.02, r'$GeV^{-}$', transform=ax2.transAxes, ha='right', fontsize=18, color='k')
# # ax2.set_xlim([0,len(y_train2)]) 
# # ax2.set_xticks(range(0,len(y_train2),40))

# ax3.set_xlabel(f"Instance", fontsize=font_size, fontfamily='Arial', rotation = 0, labelpad=6)
# ax3.set_ylabel(f"", fontsize=font_size, fontfamily='Arial', labelpad=4)
# ax3.tick_params(axis='y',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax3.tick_params(axis='x',direction='out', width=2, labelfontfamily='Arial', labelsize = font_size, pad=4)
# ax3.text(1, 0.02, r'$SnV^{-}$', transform=ax3.transAxes, ha='right', fontsize=18, color='k')

# plt.tight_layout()
# fig.savefig(r'D:\PY_Project\Paper1\Paper1_fig\HPHT_ShapTree_Heatmap.png', transparent=True)
# plt.show()