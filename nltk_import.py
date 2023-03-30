import pandas as pd
import pandas_profiling


data = pd.read_csv(r'dataset/spam.csv',encoding='latin1')
pr = data.profile_report()
pr.to_file('report.html')