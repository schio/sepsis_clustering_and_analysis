# import pandas as pd

# def get_for_mice_df():
#     return pd.read_csv('~/Workspace/paper/mimic/labeled_new_for_mice.csv')

# def get_after_mice_df():
#     return pd.read_csv('~/Workspace/paper/mimic/labeled_new_after_mice.csv')

import pandas as pd
import numpy as np
import os

class DataLoader():
    def get_origin_csv_path(self, file_name):
        origin_csv_path = "~/Workspace/paper/mimic"
        df = pd.read_csv(os.path.join(origin_csv_path, "csv", file_name))
        df.columns = map(str.lower, df.columns)
        return df
    def extract_icu_readmission(self):
        visit_df = self.get_origin_csv_path("TRANSFERS.csv")
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

        target_df = new_visit_df[['subject_id', 'hadm_id','icustay_id']]
        target_days = [2, 7, 28]
        for day in target_days:
            target_df = pd.concat([target_df, get_target_days(day)], axis=1)
        print(f"ICU 입원 수: {visit_df[visit_df.eventtype == 'admit'].shape[0]}")
        print(f"2일 내 icu 재입원 환자 수: {target_df['readmit_2d'].value_counts()[1]}")
        print(f"7일 내 icu 재입원 환자 수: {target_df['readmit_7d'].value_counts()[1]}")
        print(f"28일 내 icu 재입원 환자 수: {target_df['readmit_28d'].value_counts()[1]}")
        return target_days


    def extract_dead_in_hosp(self):
        admission_df = self.get_origin_csv_path("ADMISSIONS.csv")

        dead_in_hosp = admission_df[admission_df.hospital_expire_flag ==1][['hadm_id','hospital_expire_flag']]
        dead_in_hosp.columns=['hadm_id','dead_in_hosp']
        print('병원 내 사망자 수:',dead_in_hosp.shape[0])
        return dead_in_hosp

    def extract_key(self):
        patients = self.get_origin_csv_path("PATIENTS.csv")
        patients = patients.fillna(0)
        patients = patients[patients.dod != 0].drop(columns=['row_id'])
        print('미믹 내 사망자 수:',patients.shape[0])

        last_admit_df = self.get_origin_csv_path("ADMISSIONS.csv").fillna(0)
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

        return key_df