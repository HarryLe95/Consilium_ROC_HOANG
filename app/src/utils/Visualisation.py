import matplotlib.pyplot as plt 
import pandas as pd 
from datetime import timedelta 

colour_dict = {0:'gold',
               1:'orchid',
               2:'purple',
               3:'pink',
               4:'orange', 
               5:'darkred',
               6:'wheat', 
               7:'yellowgreen',
               8:'wheat',
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

def plot_regression_label(ax, label, start=None,end=None, colour='b'):
    if isinstance(label, pd.DataFrame):
        label_df = label['days_to_failure']
    if start is None and end is None:
        label_df = label
    elif start is None and end is not None:
        label_df = label.loc[:end]
    elif start is not None and end is None:
        label_df = label.loc[start:]
    else:
        label_df = label.loc[start:end]
    
    ax.plot(label_df, c=colour,label="Days to Failure")
    ax.grid()

def plot_label(ax: plt.axes, label:pd.Series|pd.DataFrame, start:str=None, end:str=None):
    #Slice dataframe based on start and end 
    label_df=label.labels
    if start is None and end is None:
        label_df = label_df
    elif start is None and end is not None:
        label_df = label_df.loc[:end]
    elif start is not None and end is None:
        label_df = label_df.loc[start:]
    else:
        label_df = label_df.loc[start:end]
        
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
    
def plot_ROC(raw_df, well_name, weather_df=None, label_df=None, regression_label=None, 
             start=None, end=None,
             raw_features:list=["ROC_VOLTAGE","FLOW","PRESSURE_TH"],
             weather_features:list = ["cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
             "shortwave_radiation", "direct_radiation","diffuse_radiation","direct_normal_irradiance"],
             ylim:dict = {"ROC_VOLTAGE":[25, 30]}):

    fig, ax = plt.subplots(len(raw_features), figsize=(50,35), sharex=True)
    for idx, feature in enumerate(raw_features):
        ax[idx].yaxis.set_tick_params(labelsize=20)
        plot_data(ax[idx], raw_df[feature], start, end)
        if regression_label is not None:
            regression_axis = ax[idx].twinx()
            regression_axis.spines.right.set_position(("axes",1.08))
            regression_axis.set_ylabel("Days to Failure", size=20)
            regression_axis.set_ylim(0,30)
            plot_regression_label(regression_axis, regression_label, start, end)
            regression_axis.legend(loc='best', prop={'size': 15})
            regression_axis.yaxis.set_tick_params(labelsize=20)
        if label_df is not None:
            plot_label(ax[idx],label_df, start, end)
        if weather_df is not None:
            cloud_axis = ax[idx].twinx()
            radiation_axis = ax[idx].twinx()
            radiation_axis.spines.right.set_position(("axes", 1.03))
            cloud_axis.set_ylabel("Cloud Cover", size=20)
            radiation_axis.set_ylabel("Radiation", size=20)
            cloud_axis.set_ylim(0,100)
            radiation_axis.set_ylim(0,1000)
            cloud_axis.yaxis.set_tick_params(labelsize=20)
            radiation_axis.yaxis.set_tick_params(labelsize=20)
            for weather_feature in weather_features:
                if "cloud" in weather_feature:
                    plot_data(cloud_axis, weather_df[weather_feature], start, end, color=weather_color_dict[weather_feature],size=30, marker="X")
                if "rad" in weather_feature:
                    plot_data(radiation_axis, weather_df[weather_feature], start, end, color=weather_color_dict[weather_feature],size=30, marker="D")
            cloud_axis.legend(loc='upper right', prop={'size': 15})
            radiation_axis.legend(loc='lower left', prop={'size': 15})
        if feature in ylim:
            ax[idx].set_ylim(ylim[feature])
        ax[idx].set_ylabel(feature, size=20)
        ax[idx].legend(loc='upper left', prop={'size': 15})
        
        
        
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='',ms=30) for color in colour_dict.values()]
    plt.legend(markers, colour_dict.keys(), numpoints=1,prop={'size': 30})
        
    fig.suptitle(f'{well_name}',fontsize=96)
    plt.xticks(fontsize=30)
    plt.xlabel("TS")
    return fig 
    
def plot_ROC_simple(raw_df, label_df, start, end, ylim=None, weather_df = None, generated_feature_df=None):
    
    label = label_df.loc[start:end].labels if label_df is not None else None  
    
    num_subplots = 3
    if weather_df is not None:
        num_subplots += 2
    
    if generated_feature_df is not None:
        num_subplots += len(generated_feature_df.columns)
    
    fig, ax = plt.subplots(num_subplots, figsize=(35,15), sharex=True)
    ax[0].scatter(raw_df.ROC_VOLTAGE.loc[start:end].index, raw_df.ROC_VOLTAGE.loc[start:end], s=1, c='green')
    ax[1].scatter(raw_df.FLOW.loc[start:end].index, raw_df.FLOW.loc[start:end], s=1, c='green')
    ax[2].scatter(raw_df.PRESSURE_TH.loc[start:end].index, raw_df.PRESSURE_TH.loc[start:end], s=1, c='green')
    
    subplot_idx = 3
    
    if weather_df is not None:
        ax[subplot_idx].plot(weather_df.cloudcover[start:end], c='green', label = 'cloud_cover')
        ax[subplot_idx].set_ylabel("Cloud cover")
        ax[subplot_idx].legend()
        
        ax[subplot_idx+1].plot(weather_df.direct_radiation.loc[start:end], c='green', label='direct_radiation')
        ax[subplot_idx+1].set_ylabel("Radiation")
        ax[subplot_idx+1].legend()
        subplot_idx += 2
    
    if generated_feature_df is not None:
        for feature in generated_feature_df.columns:
            ax[subplot_idx].plot(generated_feature_df.loc[start:end][feature], c='green', label = feature)
            ax[subplot_idx].set_ylabel(feature)
            ax[subplot_idx].legend()
            subplot_idx+=1

    #Plot label    
    if label is not None: 
        if len(label)!=0:
            for i in range(len(label)):
                x_start = label.index[i]
                x_end = x_start + timedelta(days=1)
                colour = colour_dict[label[i]]
                ax[0].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
                ax[1].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
                ax[2].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
                subplot_idx = 3
                if weather_df is not None:
                    ax[subplot_idx].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
                    ax[subplot_idx+1].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
                    subplot_idx = 5
                if generated_feature_df is not None:
                    for _ in generated_feature_df.columns:
                        ax[subplot_idx].axvspan(x_start, x_end, ymin=0, ymax=1, color=colour, alpha = 0.5)
                        ax[subplot_idx].grid()
                        subplot_idx += 1
                
    if ylim is not None:
        if "ROC_VOLTAGE" in ylim:
            ax[0].set_ylim(ylim["ROC_VOLTAGE"])
        if "FLOW" in ylim:
            ax[1].set_ylim(ylim["FLOW"])
        if "PRESSURE_TH" in ylim:
            ax[2].set_ylim(ylim["PRESSURE_TH"])
    ax[0].grid()
    ax[0].set_ylabel("ROC VOLTAGE")
    ax[1].set_ylabel("FLOW")
    ax[2].set_ylabel("PRESSURE_TH")
    
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='',ms=8) for color in colour_dict.values()]
    ax[2].legend(markers, colour_dict.keys(), numpoints=1,prop={'size': 8})
    return fig
