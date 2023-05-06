

def drop_na_and_duplicates(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df