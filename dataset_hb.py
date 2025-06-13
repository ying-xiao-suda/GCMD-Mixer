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


# def get_point_df(flows,climate,point):
#     days_flow=flows[:,point].reshape(-1,480)

#     weekday=[]
#     saturday=[]
#     sunday=[]
#     holiday=[]
#     holiday_list=[37,66]
#     for i in range(104):
#         dayi=days_flow[i]
#         dayi_array = np.array(dayi).reshape(-1)  
#         if(i in holiday_list):
#             holiday.append(dayi_array)
#         else:
#             if(i%7==1):
#                 saturday.append(dayi_array)
#             elif(i%7==2):
#                 sunday.append(dayi_array)
#             else:
#                 weekday.append(dayi_array)
#     saturday=np.array(saturday)
#     sunday=np.array(sunday)
#     weekday=np.array(weekday)
#     holiday=np.array(holiday)
    
#     alldays=[]
#     for i in range(14,104):
#         for j in range(480):
#             alldays.append(days_flow[i][j])

#     def get_time_list(start,end,interval= 3):
#         start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
#         end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
#         delta = timedelta(minutes=interval)
#         time_list = []
#         while start_time <= end_time:
#             time_list.append(start_time.strftime('%Y-%m-%d %H:%M:%S'))
#             start_time += delta
#         return(time_list)
#     timelist=get_time_list('2016-09-10 00:00:00','2016-12-08 23:57:00',interval=3)
#     data = {"ds": timelist, "y": alldays}
#     flow_df = pd.DataFrame(data)
#     flow_df['ds'] = pd.to_datetime(flow_df['ds'])
#     flow_df['time_month']=flow_df['ds'].apply(lambda x: x.month)
#     flow_df['time_day']=flow_df['ds'].apply(lambda x: x.day)
#     flow_df['time_hour']=flow_df['ds'].apply(lambda x: x.hour)
#     flow_df['time_minute']=flow_df['ds'].apply(lambda x: x.minute//3)
#     flow_df['week']=flow_df['ds'].apply(lambda x: x.dayofweek)
#     flow_df['weekend']=flow_df['week'].apply(lambda x:0 if x<5 else 1)
#     holiday_us=holidays.US(years=2018)
#     flow_df['holiday']=flow_df['ds'].apply(lambda x: 1 if x in holiday_us else 0)
#     flow_df['weekend'] = flow_df.apply(lambda x: 1 if x['holiday'] == 1 else x['weekend'], axis=1)
#     flow_df['week'] = flow_df.apply(lambda x: 10 if x['holiday']==1 else x['week'], axis=1)


#     flow_df.drop(['ds'],axis=1,inplace=True)

#     def time_num(df):
#         df['time_num']=df['time_hour']*60+df['time_minute']*3
#         return df

#     def ls1(item):
#         dayi=int(item.name//480)+14
#         if item['week']==3:
#             return saturday[int(dayi//7)-1][int(item['time_num']//3)]
#         elif item['week']==6:
#             return sunday[int(dayi//7)-1][int(item['time_num']//3)]
#         else:
#             return days_flow[dayi-7][int(item['time_num']//3)]
#     def ls2(item):
#         dayi=int(item.name//480)+14
#         if item['week']==3:
#             return saturday[int(dayi//7)-2][int(item['time_num']//3)]
#         elif item['week']==6:
#             return sunday[int(dayi//7)-2][int(item['time_num']//3)]
#         else:
#             return days_flow[dayi-14][int(item['time_num']//3)]        

#     def yesterday(item):
#         return days_flow[int(item.name//480)+13][int(item['time_num']//3)]
    
#     climate_repeat=pd.concat([pd.DataFrame(climate.iloc[14]).T]*480,ignore_index=True,axis=0)
#     for i in range(15,104):
#         climate_I=pd.concat([pd.DataFrame(climate.iloc[i]).T]*480,ignore_index=True,axis=0)
#         climate_repeat=pd.concat([climate_repeat,climate_I],ignore_index=True,axis=0)
#     flow_df=time_num(flow_df)
#     flow_df['point']=point
#     flow_df=pd.concat([flow_df,climate_repeat],axis=1)
#     flow_df['yes']=flow_df.apply(yesterday, axis=1)
#     flow_df['ls1']=flow_df.apply(ls1, axis=1)
#     flow_df['ls2']=flow_df.apply(ls2, axis=1)

#     return flow_df



def get_all_points_df(flows, climate):
    # ===== 1. 向量化数据加载 =====

    days_flow = flows.reshape(-1,480,378).astype(np.float64)  
    num_points=378
    # ===== 2. 分类加速 =====
    holiday_list=[37,66]
    mask_holiday = np.isin(np.arange(104), holiday_list)
    mask_sat = (~mask_holiday) & (np.arange(104) % 7 == 3)
    mask_sun = (~mask_holiday) & (np.arange(104) % 7 == 4)
    mask_wd = (~mask_holiday) & ~mask_sat & ~mask_sun
    
    weekday = days_flow[mask_wd]
    saturday = days_flow[mask_sat]
    sunday = days_flow[mask_sun]
    holiday = days_flow[mask_holiday]
    
    # # ===== 3. 异常值处理 =====
    # def outliers_nan(arr):
    #     mean = np.mean(arr,axis=0)
    #     std =np.std(arr,axis=0)
    #     arr[np.abs(arr - mean) > 2 * std] = np.nan
    #     return arr 
    
    # saturday=outliers_nan(saturday)
    # sunday=outliers_nan(sunday)
    # weekday=outliers_nan(weekday)
    # holiday=outliers_nan(holiday)
    
    
    
    # # ===== 4. 计算平均值 =====
    # weekday_avg = np.nanmean(weekday, axis=0)
    # saturday_avg = np.nanmean(saturday, axis=0)
    # sunday_avg = np.nanmean(sunday, axis=0)
    # holiday_avg = np.nanmean(holiday, axis=0)
    
    # # ===== 5. 填充NaN值 =====
    # def vector_nan_fill(data, avg):
    #     nan_mask = np.isnan(data)
    #     rows, cols,points = np.where(nan_mask)  # 获取NaN位置的行列索引
    #     data[rows, cols,points] = avg[cols,points]     # 按列索引获取对应平均值并填充
    #     return data
    # weekday = vector_nan_fill(weekday, weekday_avg)
    # saturday = vector_nan_fill(saturday, saturday_avg)
    # sunday= vector_nan_fill(sunday, sunday_avg)
    # holiday = vector_nan_fill(holiday, holiday_avg)


    
    # ===== 6. 时间序列生成 =====
    timelist = pd.date_range('2016-09-10 00:00:00','2016-12-08 23:57:00', freq='3T').strftime('%Y-%m-%d %H:%M:%S')
    timelist = np.repeat(timelist, num_points).tolist()
    M = len(timelist) // num_points  # 原始时间点数量
    point_col = np.tile(np.arange(num_points), M) 

    flow_df = pd.DataFrame({
        'ds': timelist,
        'y': days_flow[14:].flatten(),
        'point': point_col
    })
    
    # ===== 7. 特征工程 =====
    flow_df['ds'] = pd.to_datetime(flow_df['ds'])
    dt = flow_df['ds'].dt
    flow_df['time_month'] = dt.month
    flow_df['time_day'] = dt.day
    flow_df['time_hour'] = dt.hour
    flow_df['time_minute'] = dt.minute // 3
    flow_df['week'] = dt.dayofweek
    
    # ===== 8. 节假日处理 =====
    flow_df['weekend'] = (flow_df['week'] >= 5).astype(int)
    flow_df['holiday'] = flow_df['ds'].isin(holiday_list).astype(int)
    flow_df['weekend'] = np.where(flow_df['holiday'] == 1, 1, flow_df['weekend'])
    flow_df['week'] = np.where(flow_df['holiday'] == 1, 10, flow_df['week'])
    
    # ===== 9. 气候数据拼接 =====
    sliced = climate.iloc[14:104]  # 提取45天的气候数据（假设每天一行）
    climate_repeat = sliced.loc[sliced.index.repeat(480 * 378)].reset_index(drop=True)  

    # 合并到流量数据
    flow_df = pd.concat([flow_df, climate_repeat], axis=1)

    
    # ===== 10. 滞后特征向量化 =====
    flow_df['time_num'] = flow_df['time_hour'] * 60 + flow_df['time_minute'] * 3

    def ls1(item):
        point=item['point']
        dayi=int(item.name//(480*num_points))+14
        if item['week']==5:
            return saturday[int(dayi//7)-1][int(item['time_num']//3)][point]
        elif item['week']==6:
            return sunday[int(dayi//7)-1][int(item['time_num']//3)][point]
        else:
            return days_flow[dayi-7][int(item['time_num']//3)][point] 
    def ls2(item):
        point=item['point']
        dayi=int(item.name//(480*num_points))+14
        if item['week']==5:
            return saturday[int(dayi//7)-2][int(item['time_num']//3)][point]
        elif item['week']==6:
            return sunday[int(dayi//7)-2][int(item['time_num']//3)][point]
        else:
            return days_flow[dayi-14][int(item['time_num']//3)][point]       

    def yesterday(item):
        point=item['point']
        return days_flow[int(item.name//(480*num_points))+13][int(item['time_num']//3)][point]    


    flow_df['yes']=flow_df.apply(yesterday, axis=1)
    flow_df['ls1']=flow_df.apply(ls1, axis=1)
    flow_df['ls2']=flow_df.apply(ls2, axis=1)

    
    return flow_df.drop('ds', axis=1)


def get_hb_data():
    hb=np.load('./Datasets/hb/hb.npy')
    climate=pd.read_csv("./Datasets/hb/climate.csv")
    return hb, climate


class hb_Dataset(Dataset):
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
            "./DataProcess/hb_process_result/hb_alpha" + str(alpha) +'_tau'+str(tau)+ "_K" + str(K) +'_DC'+str(DC)+'_init'+str(init)+'_tol'+str(tol)+'_max_N'+str(max_N)+ ".csv"
        )

        if os.path.isfile(path) == False:
            hb, climate = get_hb_data()

            # all_point_dfs = []  # 初始化列表存储临时结果

            # for i in range(378):
            #     pointi_df = get_point_df(hb, climate, i)
            #     all_point_dfs.append(pointi_df)

            # # 一次性合并所有DataFrame
            # all_point_df = pd.concat(all_point_dfs, ignore_index=True, axis=0)
            all_point_df = get_all_points_df(hb, climate)

            original_array = all_point_df.values.reshape(90, 480, 378, -1)
            transposed_array = original_array.transpose(2, 0, 1, 3)
            new_shape = (-1, transposed_array.shape[-1])
            reshaped_array = transposed_array.reshape(new_shape)
            all_point_df = pd.DataFrame(reshaped_array, columns=all_point_df.columns)



            yll=all_point_df[['yes','ls1','ls2']]
            yll_t=torch.tensor(yll.values).reshape(378,90,480,3).permute(1,3,0,2)
            device = torch.device(config['device'])
            mvmd=MVMD(alpha, tau, K, DC, init, tol, max_N).to(device)
            md=torch.zeros([90,3,378,480]).to(device)
            for i in tqdm(range(90),desc="Processing"):
                for j in range(3):
                    u,_,_=mvmd(yll_t[i][j].to(device))
                    md[i][j]=u[0].permute(1,0)
            md=md.permute(2,0,3,1).reshape(-1,3).cpu().detach().numpy()
            md_df = pd.DataFrame(md,columns=['yes_v','ls1_v','ls2_v'])
            df=pd.concat([all_point_df,md_df],axis=1)
            df.to_csv(path,sep=',',index=False,header=True) 
        else :
            df=pd.read_csv(path)

        df=df.drop(['time_day','time_month','time_num','Day','SLP','H','VV','V','VM','VG','SN','IS','PP','I','IM','Im','FG'],axis=1)


        train_index=[0,62*480]
        valid_index=[62*480,69*480]
        test_index=[69*480,90*480]
        if mode==1:
            use_index_list=train_index
        elif mode==2:
            use_index_list=valid_index
        elif mode==3:
            use_index_list=test_index
        elif mode==0:
            use_index_list=[0,90*480]



        feature=df.drop(['y'],axis=1).values
        flow=df['y'].values

        feature=feature.reshape(378,480*90,13)
        flow=flow.reshape(378,480*90)


        # train_X = feature[:,train_index[0]:train_index[1],:].reshape(-1,13)
        # train_y = flow[:,train_index[0]:train_index[1]].reshape(-1,1)
        
        train_X = feature[:,train_index[0]:train_index[1],:]
        train_y = flow[:,train_index[0]:train_index[1]]

        # feature = (feature - train_X.min(axis=1, keepdims=True) )/(train_X.max(axis=1, keepdims=True)- train_X.min(axis=1, keepdims=True) + 1e-9)


        # 分割训练集的各特征部分
        train_X_part1 = train_X[:, :, :-6]  # 前D-6个特征
        train_X_part2 = train_X[:, :, -6:]  # 后6个特征

        # 计算前D-6特征的全局最小最大值（跨样本和时间步）
        min_part1 = train_X_part1.min(axis=(0, 1), keepdims=True)  # 形状 (1, 1, D-6)
        max_part1 = train_X_part1.max(axis=(0, 1), keepdims=True)

        # 计算后6特征的每个样本在训练时间段的最小最大值（跨时间步）
        min_part2 = train_X_part2.min(axis=1, keepdims=True)  # 形状 (N, 1, 6)
        max_part2 = train_X_part2.max(axis=1, keepdims=True)

        # 分割整个feature
        feature_part1 = feature[:, :, :-6]
        feature_part2 = feature[:, :, -6:]

        # 归一化各部分
        feature_part1 = (feature_part1 - min_part1) / (max_part1 - min_part1 + 1e-9)
        feature_part2 = (feature_part2 - min_part2) / (max_part2 - min_part2 + 1e-9)

        # 合并归一化后的特征
        feature = np.concatenate([feature_part1, feature_part2], axis=2)



        
        # max_x=np.max(train_X,axis=0)
        # min_x=np.min(train_X,axis=0)
        self.max=np.max(train_y,axis=1, keepdims=True)
        self.min=np.min(train_y,axis=1, keepdims=True)

        # feature=(feature-min_x)/(max_x-min_x+1e-9)
        flow=(flow-self.min)/(self.max-self.min+1e-9)

        # feature=feature.reshape(378,480*90,13)
        # flow=flow.reshape(378,480*90)

        self.data_x=feature[:,use_index_list[0]:use_index_list[1],:]
        self.data_y=flow[:,use_index_list[0]:use_index_list[1]]

    def __getitem__(self, index):
        # index = self.use_index_list[org_index]
        start = index*480
        end = (index+1)*480
        x=torch.tensor(self.data_x[:,start:end,:10],dtype=torch.float32)
        md=torch.tensor(self.data_x[:,start:end,10:],dtype=torch.float32)
        y=torch.tensor(self.data_y[:,start:end],dtype=torch.float32)
        return x,md,y

    def __len__(self):
        return len(self.data_y[0])//480
    
    def get_max_min(self):
        return self.max, self.min
    



def get_dataloader(config, batch_size=16):

    dataset = hb_Dataset(config,mode=1)
    max,min=dataset.get_max_min()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = hb_Dataset(config,mode=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = hb_Dataset(config,mode=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader,max,min