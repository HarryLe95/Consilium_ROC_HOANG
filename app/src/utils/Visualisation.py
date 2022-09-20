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
    "cloudcover":'red',
    "cloudcover_low":'blue',
    "cloudcover_mid":'orange',
    "cloudcover_high":'brown',
    "shortwave_radiation":'black',
    "direct_radiation":'purple',
    "diffuse_radiation":'pink',
    "direct_normal_irradiance":'orange'
}

def plot_label(ax: plt.axes, label:pd.Series|pd.DataFrame, start:str=None, end:str=None):
    #Slice dataframe based on start and end 
    if isinstance(label, pd.DataFrame):
        label_df = label.labels
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
        ax.axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
        
def plot_data(ax: plt.Axes, 
              df:pd.Series, 
              start:str=None, 
              end:str=None, 
              color:str='g',
              size:int=20,
              marker:str='.'):
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
    ax.scatter(plot_df.index, plot_df.values, c=color, s=size, marker=marker, label=plot_df.name)
    ax.grid()
    
def plot_ROC(raw_df, well_name, weather_df=None, label_df=None, start=None, end=None, 
                    raw_features:list=["ROC_VOLTAGE","FLOW","PRESSURE_TH"],
                    weather_features:list = ["cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
                    "shortwave_radiation", "direct_radiation","diffuse_radiation","direct_normal_irradiance"],
                    ylim:dict = {"ROC_VOLTAGE":[25, 30]}):

    fig, ax = plt.subplots(len(raw_features), figsize=(50,35), sharex=True)
    for idx, feature in enumerate(raw_features):
        cloud_axis = ax[idx].twinx()
        radiation_axis = ax[idx].twinx()
        radiation_axis.spines.right.set_position(("axes", 1.03))
        cloud_axis.set_ylabel("Cloud Cover", size=20)
        radiation_axis.set_ylabel("Radiation", size=20)
        cloud_axis.set_ylim(0,100)
        radiation_axis.set_ylim(0,1000)
        ax[idx].yaxis.set_tick_params(labelsize=20)
        cloud_axis.yaxis.set_tick_params(labelsize=20)
        radiation_axis.yaxis.set_tick_params(labelsize=20)

        plot_data(ax[idx], raw_df[feature], start, end)
        if label_df is not None:
            plot_label(ax[idx],label_df, start, end)
        if weather_df is not None:
            for weather_feature in weather_features:
                if "cloud" in weather_feature:
                    plot_data(cloud_axis, weather_df[weather_feature], start, end, color=weather_color_dict[weather_feature],size=30, marker="X")
                if "rad" in weather_feature:
                    plot_data(radiation_axis, weather_df[weather_feature], start, end, color=weather_color_dict[weather_feature],size=30, marker="D")
        if feature in ylim:
            ax[idx].set_ylim(ylim[feature])
        ax[idx].set_ylabel(feature, size=20)
        ax[idx].legend(loc='upper left', prop={'size': 15})
        cloud_axis.legend(loc='upper right', prop={'size': 15})
        radiation_axis.legend(loc='lower left', prop={'size': 15})
        
    fig.suptitle(f'{well_name}',fontsize=96)
    plt.xlabel("TS")
    return fig 
    