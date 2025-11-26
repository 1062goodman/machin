import FinanceDataReader as fdr

df = fdr.DataReader('005930', '2000')
x = list(df.Close)

print(df.head())


