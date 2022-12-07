from typing import Sequence
import os, shutil
from utils.PathManager import PathManager
from datetime import timedelta
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

class FolderUtils:
    def __init__(self, well_code, label):
        self.parent_dir = PathManager.VIZ_PATH/well_code/label
    
    def validate_path(self):
        self.parent_dir.mkdir(parents=True, exist_ok=True)
        
    def clear_path(self):
        if os.path.exists(self.parent_dir):
            shutil.rmtree(self.parent_dir)
        self.validate_path()
              
class PlotUtils:
    colour_dict = {0:'green',
                1:'purple',
                2:'yellow'}

    @classmethod
    def get_label_dict(cls, gt:pd.DataFrame,pred:pd.DataFrame,target_label:int|Sequence[int])->dict:
        gt_ = gt[gt.labels == target_label].index if isinstance(target_label,int) else gt[gt.labels.isin(target_label)].index
        pred_ = pred[pred.labels==1].index
        label_dict = {}
        for d in sorted(set(gt_)|set(pred_)):
            if d in gt_ and d in pred_: #TP:
                label_dict[d]=0
            elif d in gt_ and d not in pred_: #FN
                label_dict[d]=1
            elif d not in gt_ and d in pred_: #FP
                label_dict[d]=2
        return label_dict

    @classmethod
    def get_plot_window(cls, label_dict:dict)->list[tuple]:
        date_range = np.array(list(label_dict.keys())).reshape(-1)
        plot_window = []
        current_index = 0 
        while current_index < len(date_range):
            lb = date_range[current_index] - pd.Timedelta("10d")
            ub = date_range[current_index] + pd.Timedelta("10d")
            current_index = np.where(date_range>ub)[0]
            if len(current_index)!=0:
                current_index=current_index[0]
            else:
                current_index=len(date_range)
            plot_window.append([lb,ub])
        return plot_window

    #TESTED
    @classmethod
    def get_diagnostic_plots(cls, raw_df, label_dict, plot_window, window_index, well_code):
        label = pd.DataFrame({"labels": label_dict.values()}, index=label_dict.keys()).labels
        num_subplots = 3
        start = plot_window[window_index][0].strftime("%Y-%m-%d")
        end = plot_window[window_index][1].strftime("%Y-%m-%d")
        V_data = raw_df.ROC_VOLTAGE.loc[start:end]
        V_data = V_data[V_data!=0]
        label=label.loc[start:end]
        fig, ax = plt.subplots(num_subplots, figsize=(35,15), sharex=True)
        fig.suptitle(f"{well_code}_{start}_{end}", fontsize=20)
        ax[0].scatter(V_data.index, V_data, s=1, c='green')
        ax[1].scatter(raw_df.FLOW.loc[start:end].index, raw_df.FLOW.loc[start:end], s=1, c='green')
        ax[2].scatter(raw_df.PRESSURE_TH.loc[start:end].index, raw_df.PRESSURE_TH.loc[start:end], s=1, c='green')
            
        #Plot label    
        for i in range(len(label)):
            x_start = label.index[i]
            x_end = x_start + timedelta(days=1)
            colour = cls.colour_dict[label[i]]
            ax[0].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
            ax[1].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
            ax[2].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
            
        ax[0].grid()
        ax[0].set_ylabel("ROC VOLTAGE")
        ax[1].set_ylabel("FLOW")
        ax[2].set_ylabel("PRESSURE_TH")
        
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='',ms=8) for color in cls.colour_dict.values()]
        ax[2].legend(markers, ["TP","FN","FP"], numpoints=1,prop={'size': 20},loc='lower right')
        plt.xticks(fontsize=15)
        return f"{well_code}_{start}_{end}",fig
                