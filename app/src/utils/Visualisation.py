import matplotlib.pyplot as plt 
import pandas as pd 
from datetime import timedelta 

colour_dict = {0:'gold',
               1:'orchid',
               2:'navy',
               3:'salmon',
               4:'red', 
               5:'darkred',
               6:'wheat', 
               7:'yellowgreen',
               8:'mediumvioletred',
               9:'aqua'}

weather_color_dict ={
    "cloudcover":'dodgerblue',
    "cloudcover_low":'lightskyblue',
    "cloudcover_mid":'deepskyblue',
    "cloudcover_high":'steelblue',
    "shortwave_radiation":'beige',
    "direct_radiation":'lawngreen',
    "diffuse_radiation":'hotpink',
    "direct_normal_irradiance":'peru'
}

def plot_label(ax: plt.axes, ax_id:int, label:pd.Series, start:str=None, end:str=None):
    #Slice dataframe based on start and end 
    if start is None and end is None:
        label_df = label
    elif start is None and end is not None:
        label_df = label.loc[:end]
    elif start is not None and end is None:
        label_df = label.loc[start:]
    else:
        label_df = label.loc[start:end]
        
    #Overlaying label columns
    for i in range(len(label_df)):
        x_start = label_df.index[i]
        x_end = x_start + timedelta(days=1)
        colour = colour_dict[label_df[i]]
        ax[ax_id].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
        
def plot_data(ax: plt.Axes, ax_id:int, df:pd.Series, start:str=None, end:str=None, yticks=None, color:str='g', set_ylabel:bool=False):
    #Slice dataframe based on start and end 
    if start is None and end is None:
        plot_df = df
    elif start is None and end is not None:
        plot_df = df.loc[:end]
    elif start is not None and end is None:
        plot_df = df.loc[start:]
    else:
        plot_df = df.loc[start:end]
        
    #Plot overlaying features 
    ax[ax_id].scatter(plot_df.index, plot_df.values, c = color, s=5, marker='.', label=plot_df.name)
    ax[ax_id].grid()
    if yticks is not None:
        ax[ax_id].set_yticks(yticks)
    if set_ylabel:
        ax[ax_id].set_ylabel(plot_df.name)
    ax[ax_id].legend()
    
def plot_ROC(raw_df, well_name, weather_df=None, label_df=None, start=None, end=None, 
                    raw_features:list=["ROC_VOLTAGE","FLOW","PRESSURE_TH"],
                    weather_features:list = ["cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high"]):

    fig, ax = plt.subplots(len(raw_features), figsize=(50,35), sharex=True)
    for idx, feature in enumerate(raw_features):
        plot_data(ax, idx, raw_df[feature], start, end)
        if label_df is not None:
            plot_label(ax,idx,label_df, start, end)
        if weather_df is not None:
            for weather_feature in weather_features:
                plot_data(ax, idx, weather_df[weather_feature], start, end, color=weather_color_dict[weather_feature])
        
    # handleList = [plt.plot([], marker="o", ls="", color=color)[0] for color in colour_dict.values()]

    legend = plt.legend(loc='right')
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    fig.suptitle(f'{well_name}',fontsize=96)
    plt.xlabel("TS")
    