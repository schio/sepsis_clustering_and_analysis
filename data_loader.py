import pandas as pd
import numpy as np
import os
import warnings


class DataLoader():
    def __init__(self):
        warnings.filterwarnings('ignore')
        self.origin_csv_path = "~/Workspace/paper/mimic"    

    def extract_labeled_after_feature(self):
        print("RUN extract_labeled_after_feature")
        label_df = self.load_label()
        label_col = label_df.columns

        after_feature_df = self.get_csv_path("new_feature_v1_after_mice.csv", is_row_mimic=False)        
        result = pd.merge(label_df, after_feature_df, on=['hadm_id'], how='right')
        result[label_col] = result[label_col].fillna(0)
        result.to_csv('./labeled_new_feature_v1_after_mice.csv', index=False)

        return result

    def load_labeled_after_feature(self):
        print("RUN load_labeled_after_feature")
        return self.load_or_extarct("labeled_new_feature_v1_after_mice.csv", self.extract_labeled_after_feature)

    def extract_labeled_for_feature(self):
        print("RUN extract_labeled_for_feature")
        label_df = self.load_label()
        label_col = label_df.columns

        for_feature_df = self.get_csv_path("new_feature_v1_for_mice.csv", is_row_mimic=False)        
        result = pd.merge(label_df, for_feature_df, on=['hadm_id'], how='right')
        result[label_col] = result[label_col].fillna(0)
        result.to_csv('./labeled_new_feature_v1_for_mice.csv', index=False)

        return result

    def load_labeled_for_feature(self):
        print("RUN load_labeled_for_feature")
        return self.load_or_extarct("labeled_new_feature_v1_for_mice.csv", self.extract_labeled_for_feature)

    def load_or_extarct(self, file_name, func):
        csv_path = os.path.join(".", file_name)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            return func()

    def get_csv_path(self, file_name, is_row_mimic=True):
        if is_row_mimic:
            df = pd.read_csv(os.path.join(self.origin_csv_path, "csv", file_name))
        else:
            df = pd.read_csv(os.path.join(self.origin_csv_path, file_name))
        df.columns = map(str.lower, df.columns)
        return df
        
    def extract_icu_readmission(self):
        print("RUN extract_icu_readmission")
        visit_df = self.get_csv_path("TRANSFERS.csv")
        visit_df = visit_df[['subject_id','hadm_id','eventtype','intime','outtime','icustay_id']]
        filter_list = ['admit', 'discharge']
        visit_df = visit_df[visit_df.eventtype.isin(filter_list)]
        visit_df = visit_df.sort_values(by=['hadm_id', 'intime','icustay_id'], ascending=True)


        visit_df['intime'] = pd.to_datetime(visit_df['intime'])
        visit_df['outtime'] = pd.to_datetime(visit_df['outtime'])
        SUBJECT_ID=0
        HADM_ID=1
        EVENTTYPE=2
        INTIME=3
        OUTTIME=4
        ICUSTAY_ID=5

        visit_np = visit_df.values

        new_visit = []
        what = []
        for i in range(0, visit_np.shape[0]-1):
            if visit_np[i][HADM_ID] == visit_np[i+1][HADM_ID] and visit_np[i][EVENTTYPE] == 'admit' and visit_np[i+1][EVENTTYPE] == 'discharge':
                visit_np[i][OUTTIME] = visit_np[i+1][INTIME]
                new_visit.append(visit_np[i])
                
        col_name = visit_df.columns
        new_visit_df = pd.DataFrame(new_visit, columns=col_name)

        def get_target_days(days):
            res = ((new_visit_df.intime.values <= new_visit_df.outtime.values[:, None] + pd.to_timedelta('%d days' % days))
                    & (new_visit_df.intime.values > new_visit_df.intime.values[:, None])
                    & (new_visit_df.subject_id.values == new_visit_df.subject_id.values[:, None]))
            res = np.sum(res, axis=1)

            return pd.Series(res, name=f"readmit_{days}d").clip(0, 1)

        icu_readmission = new_visit_df[['subject_id', 'hadm_id','icustay_id']]
        target_days = [2, 7, 28]
        for day in target_days:
            icu_readmission = pd.concat([icu_readmission, get_target_days(day)], axis=1)
        print(f"ICU 입원 수: {visit_df[visit_df.eventtype == 'admit'].shape[0]}")
        print(f"2일 내 icu 재입원 환자 수: {icu_readmission['readmit_2d'].value_counts()[1]}")
        print(f"7일 내 icu 재입원 환자 수: {icu_readmission['readmit_7d'].value_counts()[1]}")
        print(f"28일 내 icu 재입원 환자 수: {icu_readmission['readmit_28d'].value_counts()[1]}")
        icu_readmission.to_csv("./icu_readmission.csv", index=False)
        return icu_readmission

    def load_icu_readmission(self):
        print("RUN load_icu_readmission")
        return self.load_or_extarct("icu_readmission.csv", self.extract_icu_readmission)

    def extract_dead_in_hosp(self):
        print("RUN extract_dead_in_hosp")
        admission_df = self.get_csv_path("ADMISSIONS.csv")

        dead_in_hosp = admission_df[admission_df.hospital_expire_flag ==1][['hadm_id','hospital_expire_flag']]
        dead_in_hosp.columns=['hadm_id','dead_in_hosp']
        print('병원 내 사망자 수:',dead_in_hosp.shape[0])
        dead_in_hosp.to_csv("./dead_in_hosp.csv",index=False)
        return dead_in_hosp
    
    def load_dead_in_hosp(self):
        print("RUN load_dead_in_hosp")
        return self.load_or_extarct("dead_in_hosp.csv", self.extract_dead_in_hosp)

    def extract_key(self):
        print("RUN extract_key")
        patients = self.get_csv_path("PATIENTS.csv")
        patients = patients.fillna(0)
        patients = patients[patients.dod != 0].drop(columns=['row_id'])
        print('미믹 내 사망자 수:',patients.shape[0])

        last_admit_df = self.get_csv_path("ADMISSIONS.csv").fillna(0)
        last_admit_df.columns = map(str.lower, last_admit_df.columns)
        last_admit_df = last_admit_df.sort_values(by=['admittime'])
        last_admit_df = last_admit_df.groupby(['subject_id']).tail(1).drop(columns=['row_id'])
        
        merge_df = pd.merge(patients, last_admit_df, how='right', on='subject_id')
        merge_df['dod'] = pd.to_datetime(merge_df.dod)
        merge_df['admittime'] = pd.to_datetime(merge_df.admittime)

        key_df = merge_df[['subject_id','hadm_id']]
        key_df['dead_after_out_hosp'] = (merge_df.dod - merge_df.admittime).dt.days
        key_df['dead_in_28d'] =key_df['dead_after_out_hosp'].apply(lambda x: 1 if x<29 else 0)
        key_df['dead_in_6m'] =key_df['dead_after_out_hosp'].apply(lambda x: 1 if x<30*6 else 0)
        key_df['dead_los']  =key_df['dead_after_out_hosp']
        print(f"입원 후 28일 내 사망자(병원내 사망 제외): {key_df['dead_in_28d'].value_counts()[1]}")
        print(f"입원 후 6개월 내 사망자(병원내 사망 제외): {key_df['dead_in_6m'].value_counts()[1]}")
        print(f"사망자(병원내 사망 제외): {key_df['dead_los'].value_counts().sum()}")
        key_df.to_csv("./key.csv",index=False)
        return key_df

    def load_key(self):
        print("RUN load_key")
        return self.load_or_extarct("key.csv", self.extract_key)

    def extract_label(self):
        print("RUN extract_label")
        dead_in_hosp_df = self.load_dead_in_hosp()
        key_df = self.load_key()
        icu_readmission_df = self.load_icu_readmission()

        labled_feature = pd.merge(icu_readmission_df, dead_in_hosp_df, on=['hadm_id'], how='left')
        labled_feature = pd.merge(labled_feature, key_df[['dead_in_28d','dead_in_6m','dead_los','hadm_id']], on=['hadm_id'], how='left').fillna(0).drop(columns=['subject_id','icustay_id'])
        labled_feature.to_csv('./label.csv', index=False)
        return labled_feature

    def load_label(self):
        print("RUN load_label")
        return self.load_or_extarct("./label.csv", self.extract_label)

    def ohe(self, df, columns):
        ohes = []
        for col in columns:
            df[col] = df[col].astype(str)
            temp_df = pd.get_dummies(df[col])
            temp_df.columns=list(map(lambda x: col+'_'+x, temp_df.columns))
            df = df.drop(columns=[col])
            ohes.append(temp_df)
        
        temp_df = pd.concat(ohes, axis=1)
        return pd.concat([df,temp_df], axis=1)

    def get_xy(self, df):
        y_columns = ['readmit_2d', 'readmit_7d', 'readmit_28d', 'dead_in_hosp', 'dead_in_28d', 'dead_in_6m', 'dead_los']
        need_ohe_cols = ['gender']
        df = self.ohe(df, need_ohe_cols)

        y = df[y_columns].fillna(0)
        x = df.drop(columns=y_columns)
        x = x.drop(columns=['hadm_id','icustay_id', 'unnamed: 0'])
        x = x.fillna(-1)
        x = x.drop(columns=['sofa'])

        return [x, y]