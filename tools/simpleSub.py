"""
This is a simple submission file by https://github.com/demonicCode/FutureChallenge/blob/master/simpleSub.py
"""
##########################
#### Future Challenge ####
#### Author: CZB      ####
#### Time:2017-10-21  ####
##########################
import os
import pandas as pd
import numpy as np
from datetime import *


# simple submit
def simSub(sXid, sYid, eXid, eYid, target, date):
    #### create one submit path
    sub_df = pd.DataFrame(columns=['target', 'date', 'time', 'xid', 'yid'])
    x = np.arange(sXid, eXid+((eXid-sXid)/abs(eXid-sXid)), (eXid-sXid)/abs(eXid-sXid))
    y = np.arange(sYid,eYid+((eYid-sYid)/abs(eYid-sYid)),  (eYid-sYid)/abs(eYid-sYid))
    length = len(x)+len(y)-1
    #### path array
    sub = np.zeros((length, 2))
    sub[0:len(x), 0] = x
    sub[len(x):length, 0] = x[-1]
    sub[0:len(x), 1] = y[0]
    sub[len(x)-1:length, 1] = y
    sub_df['xid'] = sub[:, 0]
    sub_df['yid'] = sub[:, 1]
    sub_df.xid = sub_df.xid.astype(np.int32)
    sub_df.yid = sub_df.yid.astype(np.int32)
    sub_df.target = target
    sub_df.date = date
    #### add time
    # because we start from hour 3, we can leave the below as it is.
    ti = datetime(2017, 11, 21, 3, 0)
    tm = [ti.strftime('%H:%M')]
    for i in range(length-1):
        ti = ti + timedelta(minutes=2)
        tm.append(ti.strftime('%H:%M'))
    sub_df.time = tm
    return sub_df


def submit_phase(cf):
    city = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    city_array = city.values
    sub_csv = pd.DataFrame(columns=['target', 'date', 'time', 'xid', 'yid'])
    for date in range(5):
        for tar in range(10):
            sub_df = simSub(city_array[0][1], city_array[0][2], city_array[tar+1][1], city_array[tar+1][2],
                            tar+1, date+cf.add_day)
            sub_csv = pd.concat([sub_csv, sub_df], axis=0)
    sub_csv.target = sub_csv.target.astype(np.int32)
    sub_csv.date = sub_csv.date.astype(np.int32)
    sub_csv.xid = sub_csv.xid.astype(np.int32)
    sub_csv.yid = sub_csv.yid.astype(np.int32)

    sub_csv.to_csv(os.path.join(cf.submission_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'.csv'),
                   header=False, index=False)
    return sub_csv


def a_star_submission(day, goal_city, start_loc, goal_loc, total_path):
    #### create one submit path
    # A random time to get the time string right
    ti = datetime(2017, 11, 21, 3, 0)
    row_list = []
    dict = {'target': goal_city,
            'date': day,
            'time': ti.strftime('%H:%M'),
            'xid': start_loc[0]+1,
            'yid': start_loc[1]+1}
    ti = ti + timedelta(minutes=2)
    row_list.append(dict)

    for p in total_path:
        for ip in p[::-1]:
            dict = {'target': goal_city,
                    'date': day,
                    'time': ti.strftime('%H:%M'),
                    'xid': ip[0]+1,
                    'yid': ip[1]+1}
            ti = ti + timedelta(minutes=2)
            row_list.append(dict)

    dict = {'target': goal_city,
            'date': day,
            'time': ti.strftime('%H:%M'),
            'xid': goal_loc[0]+1,
            'yid': goal_loc[1]+1}
    row_list.append(dict)

    sub_df = pd.DataFrame(row_list)
    return sub_df


def a_star_submission_3d(day, goal_city, start_loc, goal_loc, route_list):
    #### create one submit path
    # A random time to get the time string right
    ti = datetime(2017, 11, 21, 3, 0)
    row_list = []
    dict = {'target': goal_city,
            'date': day,
            'time': ti.strftime('%H:%M'),
            'xid': start_loc[0]+1,
            'yid': start_loc[1]+1}
    ti = ti + timedelta(minutes=2)
    row_list.append(dict)

    for ip in route_list:
        dict = {'target': goal_city,
                'date': day,
                'time': ti.strftime('%H:%M'),
                'xid': ip[0]+1,
                'yid': ip[1]+1}
        ti = ti + timedelta(minutes=2)
        row_list.append(dict)

    dict = {'target': goal_city,
            'date': day,
            'time': ti.strftime('%H:%M'),
            'xid': goal_loc[0]+1,
            'yid': goal_loc[1]+1}
    row_list.append(dict)

    sub_df = pd.DataFrame(row_list)
    return sub_df

