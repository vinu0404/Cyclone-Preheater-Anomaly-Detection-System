
import pandas as pd
import numpy as np

def create_features(df):
    """
    Create engineered features from raw cyclone sensor data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with columns: time, Cyclone_Inlet_Gas_Temp, Cyclone_Material_Temp,
        Cyclone_Outlet_Gas_draft, Cyclone_cone_draft, Cyclone_Gas_Outlet_Temp,
        Cyclone_Inlet_Draft
    
    Returns:
    --------
    pd.DataFrame with engineered features
    """
    df_feat = df.copy()
    for col in df_feat.columns:
        if col != 'time':
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
    df_feat['time'] = pd.to_datetime(df_feat['time'], errors='coerce')
    df_feat = df_feat.set_index('time')
    df_feat = df_feat.sort_index()
    df_feat = df_feat[~df_feat.index.isna()]
    df_feat = df_feat.interpolate(
        method="time",
        limit=12,
        limit_direction="both"
    )
    df_feat['temp_drop'] = (df_feat['Cyclone_Inlet_Gas_Temp'] - 
                            df_feat['Cyclone_Gas_Outlet_Temp'])
    
    df_feat['temp_ratio'] = (df_feat['Cyclone_Gas_Outlet_Temp'] / 
                             (df_feat['Cyclone_Inlet_Gas_Temp'] + 1e-6)) 
    df_feat['pressure_drop'] = (df_feat['Cyclone_Inlet_Draft'] - 
                                df_feat['Cyclone_Outlet_Gas_draft'])
    
    df_feat['pressure_ratio'] = (df_feat['Cyclone_Outlet_Gas_draft'] / 
                                 (df_feat['Cyclone_Inlet_Draft'] + 1e-6))
    
    df_feat['dT_dt'] = df_feat['Cyclone_Inlet_Gas_Temp'].diff()
    df_feat['dP_dt'] = df_feat['Cyclone_Inlet_Draft'].diff()
    df_feat['temp_roll_mean_1h'] = df_feat['Cyclone_Inlet_Gas_Temp'].rolling(12).mean()
    df_feat['temp_roll_std_1h'] = df_feat['Cyclone_Inlet_Gas_Temp'].rolling(12).std()
    df_feat['pressure_roll_mean_1h'] = df_feat['Cyclone_Inlet_Draft'].rolling(12).mean()
    df_feat['pressure_roll_std_1h'] = df_feat['Cyclone_Inlet_Draft'].rolling(12).std()
    
    return df_feat


def get_model_features():
    """
    Returns the list of features used for model training
    """
    return [
        "Cyclone_Inlet_Gas_Temp",
        "Cyclone_Gas_Outlet_Temp",
        "Cyclone_Inlet_Draft",
        "Cyclone_Outlet_Gas_draft",
        "temp_drop",
        "pressure_drop",
        "dT_dt",
        "dP_dt"
    ]