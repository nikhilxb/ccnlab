import pandas as pd


def correlation(df1, df2, value_name='value'):
  # Match corresponding rows based on all column values except the value column,
  # ignoring rows that don't have a match.
  merged = df1.merge(df2, on=[x for x in df1.columns if x != value_name])
  values1 = merged['{}_x'.format(value_name)]
  values2 = merged['{}_y'.format(value_name)]
  return values1.corr(values2, method='pearson')
