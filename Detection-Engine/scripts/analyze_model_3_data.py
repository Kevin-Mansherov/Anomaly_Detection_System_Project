import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# עדכן לנתיב של אחד הקבצים שחילצת, למשל logon.csv או device.csv
FILE_PATH = '../Datasets/Model_3/r1/r1/logon.csv' 

def analyze_cert_data(file_path):
    print(f"--- Analyzing CERT Data: {os.path.basename(file_path)} ---")
    
    # טעינת הנתונים
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    
    # 1. בדיקת סוגי פעילות (Activity Distribution)
    # נבדוק איזו עמודה מתארת את הפעילות (בדרך כלל 'activity' או דומה)
    activity_col = 'activity' if 'activity' in df.columns else df.columns[-1] # ברירת מחדל
    
    print(f"\nActivity Distribution ('{activity_col}'):")
    print(df[activity_col].value_counts())
    
    plt.figure(figsize=(10, 5))
    sns.countplot(y=activity_col, data=df, order=df[activity_col].value_counts().index)
    plt.title(f'Activity Distribution in {os.path.basename(file_path)}')
    plt.show()

    # 2. בדיקת כמות משתמשים ייחודיים
    if 'user' in df.columns:
        unique_users = df['user'].nunique()
        print(f"\nUnique Users: {unique_users}")
        
        # המשתמשים הפעילים ביותר
        print("Top 5 Most Active Users:")
        print(df['user'].value_counts().head(5))

    # 3. בדיקת טווחי זמנים
    if 'date' in df.columns:
        print("\nTime Range:")
        print(f"Start: {df['date'].min()}")
        print(f"End:   {df['date'].max()}")

if __name__ == "__main__":
    # אפשר להריץ בלולאה על כל הקבצים בתיקייה
    cert_dir = '../Datasets/Model_3/r1/r1/'
    # שנה את השם לקובץ שמעניין אותך, למשל: logon.csv, http.csv, device.csv
    analyze_cert_data(os.path.join(cert_dir, 'logon.csv'))