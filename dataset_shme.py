import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import holidays
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from Model.gcmd_mxier import MVMD
from tqdm import tqdm
import pickle



def get_point_df(flows,shme_data_time,climate,point):

    days_flow=flows[:,:,point]

    weekday=[]
    saturday=[]
    sunday=[]
    holiday=[]
    holiday_list=[76,77,78]
    for i in range(92):
        dayi=days_flow[i]
        dayi_array = np.array(dayi).reshape(-1)  
        if(i in holiday_list):
            holiday.append(dayi_array)
        else:
            if(i%7==1):
                saturday.append(dayi_array)
            elif(i%7==2):
                sunday.append(dayi_array)
            else:
                weekday.append(dayi_array)
    saturday=np.array(saturday)
    sunday=np.array(sunday)
    weekday=np.array(weekday)
    holiday=np.array(holiday)

    def outliers_nan(arr):
        mean = np.mean(arr,axis=0)
        std =np.std(arr,axis=0)
        arr[np.abs(arr - mean) > 2 * std] = np.nan
        return arr 
    
    saturday=outliers_nan(saturday)
    sunday=outliers_nan(sunday)
    weekday=outliers_nan(weekday)
    holiday=outliers_nan(holiday)
    
    saturday_avg=np.zeros(72)
    sunday_avg=np.zeros(72)
    weekday_avg=np.zeros(72)
    holiday_avg=np.zeros(72)
    for i in range(72):
        not_nan_num=0
        not_nan_sum=0
        for j in range(len(weekday)):
            arr=weekday[j]
            if not np.isnan(arr[i]):
                not_nan_sum+=arr[i]
                not_nan_num+=1
        weekday_avg[i]=not_nan_sum/not_nan_num
        
    for i in range(72):
        not_nan_num=0
        not_nan_sum=0
        for j in range(len(sunday)):
            arr=sunday[j]
            if not np.isnan(arr[i]):
                not_nan_sum+=arr[i]
                not_nan_num+=1
        sunday_avg[i]=not_nan_sum/not_nan_num
        
    for i in range(72):
        not_nan_num=0
        not_nan_sum=0
        for j in range(len(saturday)):
            arr=saturday[j]
            if not np.isnan(arr[i]):
                not_nan_sum+=arr[i]
                not_nan_num+=1
        saturday_avg[i]=not_nan_sum/not_nan_num   
        
    for i in range(72):
        not_nan_num=0
        not_nan_sum=0
        for j in range(len(holiday)):
            arr=holiday[j]
            if not np.isnan(arr[i]):
                not_nan_sum+=arr[i]
                not_nan_num+=1
        holiday_avg[i]=not_nan_sum/not_nan_num  
    def nan_processing(data,avg):
        for i in range(72):
            if  np.isnan(data[i]):
                    data[i]=avg[i]
        return data
    
    for i in range(92):
        dayi=days_flow[i]
        dayi_array = np.array(dayi).reshape(-1)  
        if(i in holiday_list):
            dayi=nan_processing(dayi,holiday_avg)
        else:
            if(i%7==1):
                dayi=nan_processing(dayi,saturday_avg)
            elif(i%7==2):
                dayi=nan_processing(dayi,sunday_avg)
            else:
                dayi=nan_processing(dayi,weekday_avg)    
        days_flow[i]=dayi

    for i in range(len(weekday)):
        dayi=weekday[i]
        dayi_array = np.array(dayi).reshape(-1)  
        weekday[i]=nan_processing(dayi,weekday_avg)
    for i in range(len(saturday)):
        dayi=saturday[i]
        dayi_array = np.array(dayi).reshape(-1)  
        saturday[i]=nan_processing(dayi,saturday_avg)
    for i in range(len(sunday)):
        dayi=sunday[i]
        dayi_array = np.array(dayi).reshape(-1)  
        sunday[i]=nan_processing(dayi,sunday_avg)
    for i in range(len(holiday)):
        dayi=holiday[i]
        dayi_array = np.array(dayi).reshape(-1)  
        holiday[i]=nan_processing(dayi,holiday_avg)
    
    alldays=[]
    for i in range(14,92):
        for j in range(72):
            alldays.append(days_flow[i][j])

    timelist=shme_data_time
    data = {"ds": timelist, "y": alldays}
    flow_df = pd.DataFrame(data)
    flow_df['ds'] = pd.to_datetime(flow_df['ds'])
    flow_df['time_month']=flow_df['ds'].apply(lambda x: x.month)
    flow_df['time_day']=flow_df['ds'].apply(lambda x: x.day)
    flow_df['time_hour']=flow_df['ds'].apply(lambda x: x.hour)
    flow_df['time_minute']=flow_df['ds'].apply(lambda x: x.minute//15)
    flow_df['week']=flow_df['ds'].apply(lambda x: x.dayofweek)
    flow_df['weekend']=flow_df['week'].apply(lambda x:0 if x<5 else 1)
    flow_df['holiday']=flow_df['ds'].apply(lambda x: 1 if x in holiday_list else 0)
    flow_df['weekend'] = flow_df.apply(lambda x: 1 if x['holiday'] == 1 else x['weekend'], axis=1)
    flow_df['week'] = flow_df.apply(lambda x: 10 if x['holiday']==1 else x['week'], axis=1)


    flow_df.drop(['ds'],axis=1,inplace=True)

    def time_num(df):
        df['time_num']=df['time_hour']*60+df['time_minute']*15-330
        return df

    def ls1(item):
    
        dayi=int(item.name//72)+14
        # if(i in holiday_list):
        #     return holiday[0][int(item['time_num']//15)]
        if dayi in holiday_list:
            return holiday_avg[int(item['time_num']//15)]
        # elif item['week']==5:
        #     return saturday[int(dayi//7)-1][int(item['time_num']//15)]
        # elif item['week']==6:
        #     return sunday[int(dayi//7)-1][int(item['time_num']//15)]
        elif dayi ==79:
            return days_flow[dayi-9][int(item['time_num']//15)]
        elif dayi in [76+7,77+7,78+7,79+7]:
            return days_flow[dayi-14][int(item['time_num']//15)]
        else:
            return days_flow[dayi-7][int(item['time_num']//15)]
    def ls2(item):
        dayi=int(item.name//72)+14
        if dayi in holiday_list:
            return holiday_avg[int(item['time_num']//15)]
        elif dayi in [76+14,77+14]:
            return days_flow[dayi-21][int(item['time_num']//15)]
        # # if dayi==3:
        # #     return holiday[0][int(item['time_num']//15)]
        # if item['week']==5:
        #     return saturday[int(dayi//7)-2][int(item['time_num']//15)]
        # elif item['week']==6:
        #     return sunday[int(dayi//7)-2][int(item['time_num']//15)]
        else:
            return days_flow[dayi-14][int(item['time_num']//15)]        



    def yesterday(item):
        dayi=int(item.name//72)+14
        if dayi ==79:
            return days_flow[dayi-4][int(item['time_num']//15)]
        else:
            return days_flow[int(item.name//72)+13][int(item['time_num']//15)]
    
    climate_repeat=pd.concat([pd.DataFrame(climate.iloc[14]).T]*72,ignore_index=True,axis=0)
    for i in range(15,92):
        climate_I=pd.concat([pd.DataFrame(climate.iloc[i]).T]*72,ignore_index=True,axis=0)
        climate_repeat=pd.concat([climate_repeat,climate_I],ignore_index=True,axis=0)
    flow_df=time_num(flow_df)
    flow_df['point']=point
    flow_df=pd.concat([flow_df,climate_repeat],axis=1)
    flow_df['yes']=flow_df.apply(yesterday, axis=1)
    flow_df['ls1']=flow_df.apply(ls1, axis=1)
    flow_df['ls2']=flow_df.apply(ls2, axis=1)

    return flow_df



def get_shme_data():
    with open('./Datasets/SHME/data.pkl', 'rb') as f:
        shme_data = pickle.load(f)
    with open('./Datasets/SHME/data_time.pkl', 'rb') as f:
        shme_data_time = pickle.load(f)
    climate = pd.read_csv('./Datasets/SHME/shanghai.csv')
    
    return shme_data, shme_data_time, climate

class Pems04_Dataset(Dataset):
    def __init__(self,config,mode=0):
        # mode:0:all，mode:1 train，mode:2 val，mode:3 test
           
        alpha=config["alpha"]
        tau=config["tau"]
        K=config["K"]
        DC=config["DC"]
        init=config["init"]
        tol=config["tol"]
        max_N=config["max_N"]

        path = (
            "./DataProcess/shme_process_result/shme_alpha" + str(alpha) +'_tau'+str(tau)+ "_K" + str(K) +'_DC'+str(DC)+'_init'+str(init)+'_tol'+str(tol)+'_max_N'+str(max_N)+ ".csv"
        )

        if os.path.isfile(path) == False:
            shme_data, shme_data_time, climate= get_shme_data()
            # (92, 73, 288, 2)
            all_point_dfs = [] 
            for i in range(288):
                pointi_df = get_point_df(shme_data[:,:72,:,0], shme_data_time[14:,:72].reshape(-1), climate, i)
                all_point_dfs.append(pointi_df)

            # 一次性合并所有DataFrame
            all_point_df = pd.concat(all_point_dfs, ignore_index=True, axis=0)
            yll=all_point_df[['yes','ls1','ls2']]
            yll_t=torch.tensor(yll.values).reshape(288,78,72,3).permute(1,3,0,2)
            device = torch.device(config['device'])
            mvmd=MVMD(alpha, tau, K, DC, init, tol, max_N).to(device)
            md=torch.zeros([78,3,288,72]).to(device)
            for i in tqdm(range(78),desc="Processing"):
                for j in range(3):
                    u,_,_=mvmd(yll_t[i][j].to(device))
                    md[i][j]=u[0].permute(1,0)
            md=md.permute(2,0,3,1).reshape(-1,3).cpu().detach().numpy()
            md_df = pd.DataFrame(md,columns=['yes_v','ls1_v','ls2_v'])
            df=pd.concat([all_point_df,md_df],axis=1)
            df.to_csv(path,sep=',',index=False,header=True) 
        else :
            df=pd.read_csv(path)

        df=df.drop(['time_day','time_month','time_num','Day','SLP','H','VV','V','VM','VG','SN','TS','PP','T','TM','Tm','FG'],axis=1)


        train_index=[0,48*72]
        valid_index=[48*72,55*72]
        test_index=[55*72,78*72]
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=[0,78*72]



        feature=df.drop(['y'],axis=1).values
        flow=df['y'].values
        feature=feature.reshape(288,78*72,13)
        flow=flow.reshape(288,78*72)


        train_X = feature[:,train_index[0]:train_index[1],:].reshape(-1,13)
        train_y = flow[:,train_index[0]:train_index[1]].reshape(-1,1)
        max_x=np.max(train_X,axis=0)
        min_x=np.min(train_X,axis=0)
        self.max=np.max(train_y,axis=0)
        self.min=np.min(train_y,axis=0)

        feature=(feature.reshape(-1,13)-min_x)/(max_x-min_x+1e-9)
        flow=(flow.reshape(-1,1)-self.min)/(self.max-self.min+1e-9)
        feature=feature.reshape(288,78*72,13)
        flow=flow.reshape(288,78*72)

        self.data_x=feature[:,use_index_list[0]:use_index_list[1],:]
        self.data_y=flow[:,use_index_list[0]:use_index_list[1]]

    def __getitem__(self, index):
        # index = self.use_index_list[org_index]
        start = index*72
        end = (index+1)*72
        x=torch.tensor(self.data_x[:,start:end,:10],dtype=torch.float32)
        md=torch.tensor(self.data_x[:,start:end,10:],dtype=torch.float32)
        y=torch.tensor(self.data_y[:,start:end],dtype=torch.float32)
        return x,md,y

    def __len__(self):
        return len(self.data_y[0])//72
    
    def get_max_min(self):
        return self.max, self.min
    



def get_dataloader(config, batch_size=16):

    dataset = Pems04_Dataset(config,mode=1)
    max,min=dataset.get_max_min()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Pems04_Dataset(config,mode=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Pems04_Dataset(config,mode=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,max,min