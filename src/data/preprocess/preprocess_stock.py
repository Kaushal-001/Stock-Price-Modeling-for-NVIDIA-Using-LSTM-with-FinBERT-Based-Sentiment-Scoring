import pandas as pd

def preprocess_stock(df: pd.DataFrame) -> pd.DataFrame:
    print("🔍 Initial Columns:", df.columns.tolist())
    print(df.head(5))

    # ✅ CASE 100% MATCH FOR YOUR CSV:
    # Row0 = fake header, Row1 = fake header, real data starts at row2
    if str(df.iloc[0, 0]) == "Price" and str(df.iloc[1, 0]) == "Date":
        print("✅ Detected NVDA CSV with DOUBLE HEADER → fixing...")

        # Use row1 as header
        df.columns = df.iloc[1]        # row1 → header
        df = df.iloc[2:].reset_index(drop=True)

        print("✅ After header fix, columns:", df.columns.tolist())
        print(df.head(5))

    # ✅ Now the FIRST column is actually the Date column
    first_col = df.columns[0]
    print(f"✅ Treating '{first_col}' as Date column")

    df.rename(columns={first_col: "Date"}, inplace=True)

    # ✅ Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # ✅ Convert numeric columns
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ✅ Drop rows missing Close price
    df = df.dropna(subset=["Close"])

    # ✅ Sort and reset
    df = df.sort_values("Date").reset_index(drop=True)

    print("✅ Cleaned & Processed Data:")
    print(df.head(5))

    return df
