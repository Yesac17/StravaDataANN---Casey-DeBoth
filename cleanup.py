import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_strava_csv():

    df = pd.read_csv("activities.csv", thousands=',',parse_dates=["Activity Date"],date_format="%b %d, %Y, %I:%M:%S %p")

    df['Distance']           = pd.to_numeric(df['Distance'])
    df['Moving Time']        = pd.to_numeric(df['Moving Time'])
    df['Average Heart Rate'] = pd.to_numeric(df['Average Heart Rate'])
    df['Elevation Gain']     = pd.to_numeric(df['Elevation Gain'])

    df = df[df['Activity Type'] == 'Run']

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=4)
    df_recent = df[df["Activity Date"] >= cutoff].copy()

    features = ['Distance','Average Speed','Average Heart Rate','Max Speed','Max Heart Rate','Idle Time','Moving Time']
    df_clean = df_recent[features + ['Relative Effort']]

    df_clean = df_clean.apply(pd.to_numeric).dropna()

    train_df, valid_df = train_test_split(df_clean, test_size=0.2)
    train_df.to_csv("TrainClean.csv", index=False)
    valid_df.to_csv("ValidClean.csv", index=False)
    print(f"Prepared {len(train_df)} train rows and {len(valid_df)} valid rows.")

prepare_strava_csv()