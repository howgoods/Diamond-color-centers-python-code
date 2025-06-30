import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from brokenaxes import brokenaxes

# ========== loading data ==========
def load_and_split_data(path, test_size=0.3, seed=42):
    df = pd.read_csv(path)
    X = np.around(np.array(df), 3)[..., 0:3]
    y = np.around(np.array(df), 3)[..., 3]
    return train_test_split(X, y, test_size=test_size, random_state=seed)

# ========== define models==========
def build_tree_model():
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        splitter='random',
        max_features=None,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        random_state=42
        )

def build_xgb_model():
    return {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'lambda': 0.0,
        'alpha': 0,
        'gamma': 0,
        'seed': 1
    }

# ========== training ==========
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        'model': model,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'r2_train': r2_score(y_train, y_train_pred),
        'r2_test': r2_score(y_test, y_test_pred),
        'mse_train': mean_squared_error(y_train, y_train_pred),
        'mse_test': mean_squared_error(y_test, y_test_pred),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
    }

def evaluate_xgb(X_train, y_train, X_test, y_test, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=1000, early_stopping_rounds=100, evals=evals, verbose_eval=False)
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    return {
        'model': model,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'r2_train': r2_score(y_train, y_train_pred),
        'r2_test': r2_score(y_test, y_test_pred),
        'mse_train': mean_squared_error(y_train, y_train_pred),
        'mse_test': mean_squared_error(y_test, y_test_pred),
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
    }

# ========== drawing ==========
def plot_colored_by_distance(y_true, y_pred, cmap='', label='', show_colorbar=False, ax=None, marker=''):
    
    error_scale = 0.2 * (np.max(y_true) - np.min(y_true))

    delta = np.random.uniform(0, error_scale, size=len(y_true))
    y_true_error = y_true + delta
    y_pred_error = y_pred + delta

    # Error and Y=X distance judgment
    dist = np.abs(y_true - y_pred) / np.sqrt(2)
    norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    color_value = 1 - norm

    # scatter chart
    sc = ax.scatter(y_true_error, y_pred_error, c=color_value, cmap=cmap,
                    s=100, marker=marker, edgecolor='k', alpha=0.8, label=label)

    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Test Accuracy (closer to Y=X)', color='darkred', fontsize=15)
        cbar.ax.yaxis.set_tick_params(color='darkred', labelsize=20)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='darkred')
    return sc

def plot_subplot(ax, y_train, y_test, y_train_pred, y_test_pred, title,
                 r2_train, r2_test, mse_train, mse_test, mae_train, mae_test,
                 label_char='(a)', show_colorbar=False, bwith=2, font_size=18):
    
    min_val = min(np.min(y_train), np.min(y_test), np.min(y_train_pred), np.min(y_test_pred))
    max_val = max(np.max(y_train), np.max(y_test), np.max(y_train_pred), np.max(y_test_pred))

    error_scale = 0.2 * (max_val - min_val)
    lims = [min_val, max_val + error_scale]

    plot_colored_by_distance(y_train, y_train_pred, cmap='Greys', label='Train', ax=ax, marker='^')
    plot_colored_by_distance(y_test, y_test_pred, cmap='Reds', label='Test', ax=ax, marker='o', show_colorbar=show_colorbar)

    ax.plot(lims, lims, 'k--', alpha=0.6, label='Y = X')

    legend = ax.legend(loc='best', frameon=False, fontsize=16)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_facecolor"):
            handle.set_facecolor('none')
        if hasattr(handle, "set_edgecolor"):
            handle.set_edgecolor('black')  # 或 handle.get_edgecolor()
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(1.5)
            
    ax.text(0.65, 0.22, r"$R^2=$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='k')
    ax.text(0.81, 0.22, f"${r2_train:.3f}$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='grey')
    ax.text(0.83, 0.22, '/', transform=ax.transAxes, ha='right', fontsize=font_size - 1, color='k')
    ax.text(0.98, 0.22, f"${r2_test:.3f}$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='#df3881')

    ax.text(0.58, 0.16, r"$MSE=$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='k')
    ax.text(0.77, 0.16, f"${mse_train:.4f}$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='grey')
    ax.text(0.79, 0.16, '/', transform=ax.transAxes, ha='right', fontsize=font_size - 1, color='k')
    ax.text(0.98, 0.16, f"${mse_test:.4f}$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='#df3881')

    ax.text(0.58, 0.10, r"$MAE=$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='k')
    ax.text(0.77, 0.10, f"${mae_train:.4f}$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='grey')
    ax.text(0.79, 0.10, '/', transform=ax.transAxes, ha='right', fontsize=font_size - 1, color='k')
    ax.text(0.98, 0.10, f"${mae_test:.4f}$", transform=ax.transAxes, ha='right', fontsize=font_size - 2, color='#df3881')

    ax.text(0.97, 0.02, label_char, transform=ax.transAxes, ha='right', fontsize=font_size + 2, color='k')

    ax.set_xlabel("Size (μm)", fontsize=font_size, fontfamily='Arial', labelpad=2)
    ax.set_ylabel("Prediction size (μm)", fontsize=font_size, fontfamily='Arial', labelpad=2)
    ax.tick_params(axis='y', direction='out', width=2, labelsize=font_size, pad=4)
    ax.tick_params(axis='x', direction='out', width=2, labelsize=font_size, pad=4)
    ax.set_title(title, fontweight='bold', fontsize=18)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(np.linspace(lims[0], lims[1], 4))
    ax.set_yticks(np.linspace(lims[0], lims[1], 4))
    
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)

# ==========main ==========
if __name__ == '__main__':
    data_paths = [
        ("NV", r"D:\PY_Project\Paper1\Paper1_data\HPHT_NV.csv"),
        ("SiV", r"D:\PY_Project\Paper1\Paper1_data\HPHT_SiV.csv"),
        ("GeV", r"D:\PY_Project\Paper1\Paper1_data\HPHT_GeV.csv"),
        ("SnV", r"D:\PY_Project\Paper1\Paper1_data\HPHT_SnV.csv")
    ]

    xgb_params = build_xgb_model()

    fig, axes = plt.subplots(2, 4, figsize=(26, 11))
    label_chars = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    titles = ['NV color centers', 'SiV color centers', 'GeV color centers', 'SnV color centers']
    font_size = 18
    bwith = 2

    for i, (name, path) in enumerate(data_paths):
        X_train, X_test, y_train, y_test = load_and_split_data(path)

        #  DTR
        tree_model = build_tree_model()
        result_tree = evaluate_model(tree_model, X_train, y_train, X_test, y_test)
        plot_subplot(axes[0, i],
                     y_train=result_tree['y_train'][::2],
                     y_test=result_tree['y_test'][::2],
                     y_train_pred=result_tree['y_train_pred'][::2],
                     y_test_pred=result_tree['y_test_pred'][::2],
                     title=titles[i],
                     label_char=label_chars[i],
                     r2_train=result_tree['r2_train'],
                     r2_test=result_tree['r2_test'],
                     mse_train=result_tree['mse_train'],
                     mse_test=result_tree['mse_test'],
                     mae_train=result_tree['mae_train'],
                     mae_test=result_tree['mae_test'],
                     show_colorbar=True,
                     bwith=bwith,
                     font_size=font_size)

        #  XGB
        result_xgb = evaluate_xgb(X_train, y_train, X_test, y_test, xgb_params)
        plot_subplot(axes[1, i],
                     y_train=result_xgb['y_train'][::2],
                     y_test=result_xgb['y_test'][::2],
                     y_train_pred=result_xgb['y_train_pred'][::2],
                     y_test_pred=result_xgb['y_test_pred'][::2],
                     title=titles[i],
                     label_char=label_chars[i+4],
                     r2_train=result_xgb['r2_train'],
                     r2_test=result_xgb['r2_test'],
                     mse_train=result_xgb['mse_train'],
                     mse_test=result_xgb['mse_test'],
                     mae_train=result_xgb['mae_train'],
                     mae_test=result_xgb['mae_test'],
                     show_colorbar=True,
                     bwith=bwith,
                     font_size=font_size)

    plt.tight_layout()
    plt.savefig(r'D:\PY_Project\Paper1\Paper1_fig\HPHT_DTR_XGB_compare.svg', format='svg', transparent=True, bbox_inches='tight')
    plt.show()