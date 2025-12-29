import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# הגדר את הנתיב לקובץ ה-CSV הספציפי שאתה עובד עליו כרגע
FILE_PATH = "../Datasets/Model_1_and_2/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
#'../Datasets/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv'

def analyze_dataset(file_path):
    print(f"--- Starting Analysis for: {os.path.basename(file_path)} ---")
    
    # 1. טעינת הנתונים
    df = pd.read_csv(file_path)
    
    # ניקוי רווחים בשמות העמודות (בעיה ידועה ב-CIC-IDS)
    df.columns = df.columns.str.strip()
    
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    
    # 2. בדיקת התפלגות ה-Labels (תקין מול התקפה)
    print("\n--- Label Distribution ---")
    if 'Label' in df.columns:
        print(df['Label'].value_counts())
        
        # ויזואליזציה של ההתפלגות
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Label', data=df)
        plt.title('Traffic Distribution (Benign vs Attacks)')
        plt.show()
    else:
        print("Column 'Label' not found!")

    # 3. בדיקת ערכים חסרים (Missing Values & Infinity)
    print("\n--- Data Quality Check ---")
    # המרת אינסוף ל-NaN כדי שנוכל לספור אותם
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    
    if not columns_with_nulls.empty:
        print("Columns with missing/infinite values:")
        print(columns_with_nulls)
    else:
        print("No missing values found (Clean dataset).")

    # 4. בדיקת שונות (Variance) - מציאת עמודות "מתות"
    # עמודה שיש בה רק ערך אחד (למשל תמיד 0) לא תורמת ללמידה
    numeric_df = df.select_dtypes(include=[np.number])
    std_dev = numeric_df.std()
    zero_variance_cols = std_dev[std_dev == 0].index
    
    print("\n--- Redundant Features (Zero Variance) ---")
    if len(zero_variance_cols) > 0:
        print(f"Found {len(zero_variance_cols)} columns with single value (useless):")
        print(list(zero_variance_cols))
    else:
        print("All columns have variance.")

    # 5. מפת חום (Correlation Heatmap) - אופציונלי
    # זה מראה לנו אם יש עמודות שאומרות בדיוק אותו דבר
    print("\nGenerating Correlation Heatmap (this might take a moment)...")
    plt.figure(figsize=(12, 10))
    # לוקחים מדגם קטן כי הדאטה ענק
    sns.heatmap(numeric_df.sample(5000).corr(), cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    analyze_dataset(FILE_PATH)