import pandas as pd
import statsmodels.api as sm


class Analysis():
    def __init__(self):
        self.columns = [
            'cluster_category_0',
            'cluster_category_1',
            'cluster_category_2',
            'cluster_category_3'
            ]

    def basic_characteristics(self, x, y, filename_for_save):
        print(f'RUN basic_characteristics - {filename_for_save}')
        result = pd.DataFrame()
        for categoty in self.columns:
            df = x[x[categoty]==1]
            n = df.shape[0]
            if n == 0:
                continue
            df = df.describe()
            print(df)
            df = df.astype('int').astype('str')
            df = df.transpose()
            result[f'{categoty}(n={n})'] = df[['mean','std']].apply(lambda x: '(+-'.join(x)+')', axis=1)
        
        x2 = sm.add_constant(x)
        
        for col in y.columns:
            y2 = y[col]
            
            est = sm.OLS(y2.astype(float), x2.astype(float))
            est2 = est.fit()
            p_values = est2.pvalues
            p_values = p_values.drop('const')
            
            p_values_df = pd.DataFrame(p_values,columns=[col])
            result = pd.merge(result, p_values_df, left_index=True, right_index=True)        

        result = result.reset_index().rename(columns={'index':'column'})
        result.to_csv(f"./result/basic_characteristics_{filename_for_save}")
        print(f"./result/basic_characteristics_{filename_for_save}")
        return result