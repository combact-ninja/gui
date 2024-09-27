import pandas as pd


# -------------- already done useful for reference ---------------
# new_df = pd.read_csv('processed.csv')
# # Create new DataFrame with original, lowercase, and uppercase rows
# df_lower = new_df.copy()
# df_lower['drugName'] = df_lower['drugName'].str.lower()
#
# df_upper = new_df.copy()
# df_upper['drugName'] = df_upper['drugName'].str.upper()
#
# # Concatenate the original, lowercase, and uppercase dataframes
# new_df1 = pd.concat([new_df, df_lower, df_upper], ignore_index=True)
# new_df1.to_csv('processed.csv')
# print('hi')



import wikipedia

# Function to get summary from Wikipedia
def get_medicine_summary(medicine_name):
    try:
        summary = wikipedia.summary(medicine_name, sentences=2)
        return summary
    except wikipedia.exceptions.PageError:
        return "Medicine not found on Wikipedia."

# Get summary for a specific medicine
summary = get_medicine_summary("LUBIPROSTONE")
print(summary)
