import pandas as pd

def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:

    #ensure that publishedAt columns exist
    if 'publishedAt' not in df.columns:
        raise KeyError("Column 'publishedAt' column not found in news data.")
    
    #Convert to datetime and extract date
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce').dt.date
    df.dropna(subset=['publishedAt'],inplace=True)

    #Combine title and description:
    if 'title' in df.columns and 'description' in df.columns:
        df['text']=df['title'].fillna('') + '. ' + df['description'].fillna('')
    elif 'title' in df.columns:
        df['text'] = df['title'].fillna('')
    elif 'description' in df.columns:
        df['text'] = df['description'].fillna('')
    else:
        raise KeyError("No 'title' or 'description' column found for text analysis.")
    
    # Remove duplicates and empty text
    df.drop_duplicates(subset=['text'], inplace=True)
    df = df[df['text'].str.strip() != '']

    return df
