import numpy as np 
import math
import copy
import pandas as pd
import os
from math import sin, cos, asin, radians, sqrt
import time
import pulp
import datetime
import re
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import subprocess
import os
#import datetime as dt
#from datetime import datetime
import itertools
#os.chdir(r'D:\Ideal Fleet\\')
       
start_time=time.time()

def get_distances_matrix_format(road_distance_matrix):
    road_distance_matrix_v1=road_distance_matrix[(road_distance_matrix['FromLatitude']!=0.00) ].reset_index(drop=True)
    road_distance_matrix_v1=road_distance_matrix_v1[(road_distance_matrix_v1['ToLatitude']!=0.00) ].reset_index(drop=True)
    road_distance_matrix_v1['Kilometres']=road_distance_matrix_v1['Kilometres'].astype(object)
    wo_infinty_dist=road_distance_matrix_v1[road_distance_matrix_v1['Kilometres']!='Infinity'].reset_index(drop=True)
    infinty_dist=road_distance_matrix_v1[road_distance_matrix_v1['Kilometres']=='Infinity'].reset_index(drop=True)
    dist_mat_v1=[]
    for i in range(0,len(infinty_dist)):    
        lat1=infinty_dist['FromLatitude'][i]
        lon1=infinty_dist['FromLongitude'][i]
        lat2=infinty_dist['ToLatitude'][i]
        lon2=infinty_dist['ToLongitude'][i]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon=lon2-lon1
        dlat=lat2-lat1
        a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        d=c*r
        #print(d)
        dist_mat_v1.append(d)
    infinty_dist['new_dist']=dist_mat_v1
    del infinty_dist['Kilometres']
    infinty_dist.rename(columns={'new_dist':'Kilometres'},inplace=True)
    final_dist=pd.concat([wo_infinty_dist,infinty_dist]).reset_index(drop=True)
    final_road_dist=pd.DataFrame(final_dist.pivot(index='FromID', columns='ToID', values='Kilometres'))
    distance_matrix=final_road_dist.copy(deep=True)
    distance_matrix=distance_matrix.apply(pd.to_numeric, downcast='float', errors='coerce')
    return distance_matrix



def form_shikar_bike_beats(ord_typ,start_node,van_id,mids,outlet_bill_dict,outlet_basepack_dict,ol_plg,ol_channel,ol_weights,ol_volume,ol_service_time):
    global outlets_allowed_forvan,van_speed_dict,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict

    weight=van_weight_dict[van_id]
    volume=van_volume_dict[van_id]
    bills=van_bill_dict[van_id]
    outs=van_outlet_dict[van_id]
    max_time=van_endtime_dict[van_id]
    
    if((van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') ):
        max_time=multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]
                
    sequence=[]
    wgt_sequence=[0]
    vol_sequence=[0]
    time_sequence=[0]
    plgs=[['']]
    channels=[['']]
    bills_cov=[['']]
    base_pack_cov=[['']]
    cnt=0
    speed=1/van_speed_dict[van_id]
    cnt=cnt+1
    cumulative_weight_seq=0
    cumulative_volume_seq=0
    end_time_seq=0
    ni_to_rs_time=0
    num_stores_seq=0
    num_bills=0
    bills_list=[]
    sequence.append('Distributor') 
    #mids=ids.copy()
    closuretime_overlap=[]
    
    if(start_node in exclusive_outlets):
        mids=list(set(mids).intersection(exclusive_outlets))
    else:
        mids=list(set(mids)-set(exclusive_outlets))

    while True:
        prev_sequence=sequence.copy()
        prev_wgt_sequence=wgt_sequence.copy()
        prev_vol_sequence=vol_sequence.copy()
        prev_time_sequence=time_sequence.copy()
        prev_bills_cov=bills_cov.copy()
        prev_base_pack_cov=base_pack_cov.copy()
        prev_plgs=plgs.copy()
        prev_channels=channels.copy()
        #print(prev_sequence)
        prev_end_time_seq=end_time_seq
        prev_ni_to_rs_time=ni_to_rs_time
        prev_num_stores_seq=num_stores_seq
        prev_cumulative_weight_seq=cumulative_weight_seq
        prev_cumulative_volume_seq=cumulative_volume_seq
        if(len(list(set(mids)-set(closuretime_overlap)))<=0):
            #print('mids zero')
            break
        dist = distance_matrix[sequence[-1]][distance_matrix[sequence[-1]].index.isin(list(set(mids)-set(closuretime_overlap)))]
        dist = pd.Series(dist)
        if(cnt==1):
            nearest_index = start_node
            cnt = cnt+1
        else:
            nearest_index = dist.idxmin()

        distance = dist[nearest_index]
        travel_time = math.ceil(distance * speed)
        end_time_seq = end_time_seq + travel_time
        st_time = end_time_seq
        service_time=ol_service_time[nearest_index]
        end_time_seq=end_time_seq+ service_time
        fi_time = end_time_seq
        ni_to_rs_dist = distance_matrix[nearest_index][sequence[0]]
        ni_to_rs_time = math.ceil(ni_to_rs_dist * speed)
        
        '''
        if(nearest_index in list(ol_closure_time.keys())):
            if(((st_time > ol_closure_time[nearest_index][0]) & (st_time < ol_closure_time[nearest_index][1])) | ((fi_time > ol_closure_time[nearest_index][0]) & (fi_time < ol_closure_time[nearest_index][1]))):
                end_time_seq = end_time_seq - (travel_time + service_time)
                closuretime_overlap.append(nearest_index)
                continue
        
        if(((cumulative_weight_seq + ol_weights[nearest_index]) > weight) | ((end_time_seq + ni_to_rs_time) > max_time) | ((num_stores_seq + 1) > outs)):
            break
        '''
        
        cumulative_weight_seq += ol_weights[nearest_index]
        cumulative_volume_seq +=ol_volume[nearest_index]
        wgt_sequence.append(ol_weights[nearest_index])
        cumulative_volume_seq=cumulative_volume_seq+ol_volume[nearest_index]
        vol_sequence.append(ol_volume[nearest_index])
        service_time=ol_service_time[nearest_index]
        end_time_seq=end_time_seq+ service_time
        bills_=list(outlet_bill_dict[nearest_index])
        bills_cov.append(bills_)
        #num_bills=num_bills+len(bills_)
        bills_list.extend(bills_)
        base_pack_cov.append(list(outlet_basepack_dict[nearest_index]))
        plgs.append(ol_plg[nearest_index])
        channels.append(ol_channel[nearest_index])
        
        distance = dist[nearest_index]  
        travel_time = math.ceil(distance * speed)
        end_time_seq = end_time_seq + travel_time
        time_sequence.append(service_time+travel_time)
        
        num_stores_seq=num_stores_seq+1
        sequence.append(nearest_index)
        num_bills=len(set(bills_list))
        mids.remove(nearest_index)
        
        for ots in closuretime_overlap:
            if(end_time_seq > ol_closure_time[ots][1]):
                closuretime_overlap.remove(ots)
        
        if((cumulative_weight_seq<=weight) and (end_time_seq+ni_to_rs_time<= max_time) and (num_stores_seq<=outs) and (cumulative_volume_seq<=volume) and (num_bills<=bills)):
            continue
        else:
            break


    prev_sequence.append('Distributor')
    prev_wgt_sequence.append(0)
    prev_vol_sequence.append(0)
    prev_time_sequence.append(math.ceil(distance_matrix[prev_sequence[-2]]['Distributor'] * speed))
    prev_bills_cov.append([''])
    prev_base_pack_cov.append([''])
    prev_plgs.append([''])
    prev_channels.append([''])
    
    return prev_sequence,prev_end_time_seq+prev_ni_to_rs_time,prev_num_stores_seq,prev_cumulative_weight_seq,prev_wgt_sequence,prev_cumulative_volume_seq,prev_vol_sequence,prev_time_sequence,prev_bills_cov,prev_base_pack_cov,prev_plgs,prev_channels




def form_shikar_beats(ord_typ,start_node,van_id,mids,outlet_bill_dict,outlet_basepack_dict,ol_plg,ol_channel,ol_weights,ol_volume,ol_service_time,high_demand_outs,rem_weight,rem_volume,rem_basepack):
    #for now i am writing for only 2 trips
    global outlets_allowed_forvan,van_speed_dict,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict        
    weight=van_weight_dict[van_id]
    volume=van_volume_dict[van_id]
    bills=van_bill_dict[van_id]
    outs=van_outlet_dict[van_id]
    max_time=van_endtime_dict[van_id]
    if((van_id in van_multitrip_dict.keys()) and (van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') ):
        max_time=multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]
    
    sequence=[]
    wgt_sequence=[0]
    vol_sequence=[0]
    time_sequence=[0]
    plgs=[['']]
    channels=[['']]
    bills_cov=[['']]
    base_pack_cov=[['']]
    cnt=0
    speed=1/van_speed_dict[van_id]
    cnt=cnt+1
    cumulative_weight_seq=0
    cumulative_volume_seq=0
    end_time_seq=0
    ni_to_rs_time=0
    num_stores_seq=0
    num_bills=0
    bills_list=[]
    sequence.append('Distributor') 
    closuretime_overlap=[]
    
    if(start_node in exclusive_outlets):
        mids=list(set(mids).intersection(exclusive_outlets))
    else:
        mids=list(set(mids)-set(exclusive_outlets))
    
    while(True):
        prev_sequence=sequence.copy()
        prev_wgt_sequence=wgt_sequence.copy()
        prev_vol_sequence=vol_sequence.copy()
        prev_time_sequence=time_sequence.copy()
        prev_bills_cov=bills_cov.copy()
        prev_base_pack_cov=base_pack_cov.copy()
        prev_plgs=plgs.copy()
        prev_channels=channels.copy()
        #print(prev_sequence)
        prev_end_time_seq=end_time_seq
        prev_ni_to_rs_time=ni_to_rs_time
        prev_num_stores_seq=num_stores_seq
        prev_cumulative_weight_seq=cumulative_weight_seq
        prev_cumulative_volume_seq=cumulative_volume_seq
        if(len(list(set(mids)-set(closuretime_overlap)))<=0):
            #print('mids zero')
            break
        dist = distance_matrix[sequence[-1]][distance_matrix[sequence[-1]].index.isin(list(set(mids)-set(closuretime_overlap)))]
        dist = pd.Series(dist)
        if(cnt==1):
            nearest_index=start_node
            cnt=cnt+1
        else:
            nearest_index = dist.idxmin()
        #print(nearest_index)
        
        if(nearest_index in rem_weight.keys()):
            bills_=list(input_data[ (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepack[nearest_index]))]['BILL_NUMBER'].unique())
            if((rem_weight[nearest_index]> (weight-cumulative_weight_seq)) or (len(bills_)> (bills-num_bills)) or (rem_volume[nearest_index]> (volume-cumulative_volume_seq))):
                basepacks,bills_,wgt,vol,ps,cs=find_ideal_weight_to_be_added(nearest_index,ord_typ,weight-cumulative_weight_seq,volume-cumulative_volume_seq,bills-num_bills)
                if(wgt>0):
                    cumulative_weight_seq = cumulative_weight_seq+wgt
                    wgt_sequence.append(wgt)
                    cumulative_volume_seq=cumulative_volume_seq+vol
                    vol_sequence.append(vol)
                    service_time=((wgt)*ol_service_time[nearest_index])/ol_weights[nearest_index]
                    end_time_seq=end_time_seq+ service_time
                    bills_cov.append(bills_)
                    bills_list.extend(bills_)
                    plgs.append(ps)       
                    channels.append(cs)
                    #num_bills=num_bills+len(bills_)
                    base_pack_cov.append(basepacks)
                else:
                    mids.remove(nearest_index)
                    continue
                    
            else:
               cumulative_weight_seq += rem_weight[nearest_index] 
               wgt_sequence.append(rem_weight[nearest_index])
               cumulative_volume_seq=cumulative_volume_seq+rem_volume[nearest_index]
               vol_sequence.append(rem_volume[nearest_index])
               service_time=(rem_weight[nearest_index]*ol_service_time[nearest_index])/ol_weights[nearest_index]
               end_time_seq=end_time_seq+ service_time
               bills_=list(input_data[ (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepack[nearest_index]))]['BILL_NUMBER'].unique())
               bills_cov.append(bills_)
               plgs.append(list(input_data[ (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepack[nearest_index]))]['SERVICING PLG'] ))       
               channels.append(list(input_data[ (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepack[nearest_index]))]['primarychannel'] ))  
               #num_bills=num_bills+len(bills_)
               bills_list.extend(bills_)
               base_pack_cov.append(rem_basepack[nearest_index])                

        else:
            cumulative_weight_seq += ol_weights[nearest_index]
            cumulative_volume_seq +=ol_volume[nearest_index]
            wgt_sequence.append(ol_weights[nearest_index])
            cumulative_volume_seq=cumulative_volume_seq+ol_volume[nearest_index]
            vol_sequence.append(ol_volume[nearest_index])
            service_time=ol_service_time[nearest_index]
            end_time_seq=end_time_seq+ service_time
            bills_=list(outlet_bill_dict[nearest_index])
            bills_cov.append(bills_)
            #num_bills=num_bills+len(bills_)
            bills_list.extend(bills_)
            base_pack_cov.append(list(outlet_basepack_dict[nearest_index]))
            plgs.append(ol_plg[nearest_index])
            channels.append(ol_channel[nearest_index])
        
        distance = dist[nearest_index]  
        travel_time = math.ceil(distance * speed)
        st_time=end_time_seq
        end_time_seq = end_time_seq + travel_time
        fi_time=end_time_seq
        
        '''
        if(nearest_index in list(ol_closure_time.keys())):
                if(((st_time > ol_closure_time[nearest_index][0]) & (st_time < ol_closure_time[nearest_index][1])) | ((fi_time > ol_closure_time[nearest_index][0]) & (fi_time < ol_closure_time[nearest_index][1]))):
                    sequence=prev_sequence.copy()
                    wgt_sequence=prev_wgt_sequence.copy()
                    vol_sequence=prev_vol_sequence.copy()
                    time_sequence=prev_time_sequence.copy()
                    bills_cov=prev_bills_cov.copy()
                    base_pack_cov=prev_base_pack_cov.copy()
                    plgs=prev_plgs.copy()
                    channels=prev_channels.copy()
                    #print(prev_sequence)
                    end_time_seq=prev_end_time_seq
                    ni_to_rs_time=prev_ni_to_rs_time
                    num_stores_seq=prev_num_stores_seq
                    cumulative_weight_seq=prev_cumulative_weight_seq
                    cumulative_volume_seq=prev_cumulative_volume_seq
                    closuretime_overlap.append(nearest_index)
                    continue
        '''
        time_sequence.append(service_time+travel_time)
        ni_to_rs_dist = distance_matrix[nearest_index][sequence[0]]
        ni_to_rs_time = math.ceil(ni_to_rs_dist * speed)
        num_stores_seq=num_stores_seq+1
        sequence.append(nearest_index)
        num_bills=len(set(bills_list))
        mids.remove(nearest_index)
        
        for ots in closuretime_overlap:
            if(end_time_seq > ol_closure_time[ots][1]):
                closuretime_overlap.remove(ots)
        
        if((cumulative_weight_seq<=weight) and (end_time_seq+ni_to_rs_time<= max_time) and (num_stores_seq<=outs) and (cumulative_volume_seq<=volume) and (num_bills<=bills)):
            continue
        else:
            break
    prev_sequence.append('Distributor')
    prev_wgt_sequence.append(0)
    prev_vol_sequence.append(0)
    prev_time_sequence.append(math.ceil(distance_matrix[prev_sequence[-2]]['Distributor'] * speed))
    prev_bills_cov.append([''])
    prev_base_pack_cov.append([''])
    prev_plgs.append([''])
    prev_channels.append([''])
    
    return prev_sequence,prev_end_time_seq+prev_ni_to_rs_time,prev_num_stores_seq,prev_cumulative_weight_seq,prev_wgt_sequence,prev_cumulative_volume_seq,prev_vol_sequence,prev_time_sequence,prev_bills_cov,prev_base_pack_cov,prev_plgs,prev_channels
    
def form_bike_beats(key, start_node, van_id):    
    global outlets_allowed_forvan,van_speed_dict,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,grup_outlets_allowed_for_bike,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict

    ids = list(set(grup_of_outlets[key]).intersection(set(outlets_allowed_for_bike[van_id])).intersection(set(grup_outlets_allowed_for_bike[van_id][key])))
    ol_service_time=grup_ol_service_time[key].copy()
    ol_weights=grup_ol_weights[key].copy()
    ol_volumes=grup_ol_volume[key].copy()
    outlet_bill_dict=grup_outlet_bill_dict[key].copy()
    outlet_basepack_dict=grup_outlet_basepack_dict[key].copy()
    ol_plg=grup_ol_plg[key].copy()
    ol_channel=grup_ol_channel[key].copy()
    weight=van_weight_dict[van_id]
    volume=van_volume_dict[van_id]
    bills=van_bill_dict[van_id]
    outs=van_outlet_dict[van_id]
    max_time=van_endtime_dict[van_id]
    
    if((van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') ):
        max_time=multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]
                
    sequence=[]
    wgt_sequence=[0]
    vol_sequence=[0]
    time_sequence=[0]
    plgs=[['']]
    channels=[['']]
    bills_cov=[['']]
    base_pack_cov=[['']]
    cnt=0
    speed=1/van_speed_dict[van_id]
    cnt=cnt+1
    cumulative_weight_seq=0
    cumulative_volume_seq=0
    end_time_seq=0
    ni_to_rs_time=0
    num_stores_seq=0
    num_bills=0
    bills_list=[]
    sequence.append('Distributor') 
    mids=ids.copy()
    if(start_node in exclusive_outlets):
        mids=list(set(mids).intersection(exclusive_outlets))
    else:
        mids=list(set(mids)-set(exclusive_outlets))
    closuretime_overlap=[]
    while True:
        prev_sequence=sequence.copy()
        prev_wgt_sequence=wgt_sequence.copy()
        prev_vol_sequence=vol_sequence.copy()
        prev_time_sequence=time_sequence.copy()
        prev_bills_cov=bills_cov.copy()
        prev_base_pack_cov=base_pack_cov.copy()
        prev_plgs=plgs.copy()
        prev_channels=channels.copy()
        #print(prev_sequence)
        prev_end_time_seq=end_time_seq
        prev_ni_to_rs_time=ni_to_rs_time
        prev_num_stores_seq=num_stores_seq
        prev_cumulative_weight_seq=cumulative_weight_seq
        prev_cumulative_volume_seq=cumulative_volume_seq
        if(len(list(set(mids)-set(closuretime_overlap)))<=0):
            #print('mids zero')
            break
        dist = distance_matrix[sequence[-1]][distance_matrix[sequence[-1]].index.isin(list(set(mids)-set(closuretime_overlap)))]
        dist = pd.Series(dist)
        if(cnt==1):
            nearest_index = start_node
            cnt = cnt+1
        else:
            nearest_index = dist.idxmin()

        

        distance = dist[nearest_index]
        travel_time = math.ceil(distance * speed)
        end_time_seq = end_time_seq + travel_time
        st_time = end_time_seq
        service_time=ol_service_time[nearest_index]
        end_time_seq=end_time_seq+ service_time
        fi_time = end_time_seq
        ni_to_rs_dist = distance_matrix[nearest_index][sequence[0]]
        ni_to_rs_time = math.ceil(ni_to_rs_dist * speed)
        
        '''
        if(nearest_index in list(ol_closure_time.keys())):
                if(((st_time > ol_closure_time[nearest_index][0]) & (st_time < ol_closure_time[nearest_index][1])) | ((fi_time > ol_closure_time[nearest_index][0]) & (fi_time < ol_closure_time[nearest_index][1]))):
                    end_time_seq = end_time_seq - (travel_time + service_time)
                    closuretime_overlap.append(nearest_index)
                    continue
        
        if(((cumulative_weight_seq + ol_weights[nearest_index]) > weight) | ((end_time_seq + ni_to_rs_time) > max_time) | ((num_stores_seq + 1) > outs)):
            break
        '''
        cumulative_weight_seq += ol_weights[nearest_index]
        cumulative_volume_seq +=ol_volumes[nearest_index]
        wgt_sequence.append(ol_weights[nearest_index])
        cumulative_volume_seq=cumulative_volume_seq+ol_volumes[nearest_index]
        vol_sequence.append(ol_volumes[nearest_index])
        service_time=ol_service_time[nearest_index]
        end_time_seq=end_time_seq+ service_time
        bills_=list(outlet_bill_dict[nearest_index])
        bills_cov.append(bills_)
        #num_bills=num_bills+len(bills_)
        bills_list.extend(bills_)
        base_pack_cov.append(list(outlet_basepack_dict[nearest_index]))
        plgs.append(ol_plg[nearest_index])
        channels.append(ol_channel[nearest_index])
        
        distance = dist[nearest_index]  
        travel_time = math.ceil(distance * speed)
        end_time_seq = end_time_seq + travel_time
        time_sequence.append(service_time+travel_time)
        
        num_stores_seq=num_stores_seq+1
        sequence.append(nearest_index)
        num_bills=len(set(bills_list))
        mids.remove(nearest_index)
        
        for ots in closuretime_overlap:
            if(end_time_seq > ol_closure_time[ots][1]):
                closuretime_overlap.remove(ots)
        
        if((cumulative_weight_seq<=weight) and (end_time_seq+ni_to_rs_time<= max_time) and (num_stores_seq<=outs) and (cumulative_volume_seq<=volume) and (num_bills<=bills)):
            continue
        else:
            break
    
    prev_sequence.append('Distributor')
    prev_wgt_sequence.append(0)
    prev_vol_sequence.append(0)
    prev_time_sequence.append(math.ceil(distance_matrix[prev_sequence[-2]]['Distributor'] * speed))
    prev_bills_cov.append([''])
    prev_base_pack_cov.append([''])
    prev_plgs.append([''])
    prev_channels.append([''])
    
    return prev_sequence,prev_end_time_seq+prev_ni_to_rs_time,prev_num_stores_seq,prev_cumulative_weight_seq,prev_wgt_sequence,prev_cumulative_volume_seq,prev_vol_sequence,prev_time_sequence,prev_bills_cov,prev_base_pack_cov,prev_plgs,prev_channels


def find_best_bike_beat(beat_list,van_id,bike_ols):
    global outlets_allowed_forvan,van_speed_dict,grup_outlets_allowed_for_bike,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict
    min_beat={}
    min_cost=100000
    maxtime=van_endtime_dict[van_id]
    weight=van_weight_dict[van_id]
    bills=van_bill_dict[van_id]
    volume=van_volume_dict[van_id]
    X=pd.DataFrame()
    
    for beat in beat_list:
        X = pd.concat([X, pd.DataFrame({'time' : [beat[1]], 'wgt' : [beat[3]], 'bills' : [beat[2]], 'vol' : [beat[6]]})])
    
    X=pd.concat([X,pd.DataFrame({'time':[0],'wgt':[0],'bills':[0],'vol':[0]})])
    X=pd.concat([X,pd.DataFrame({'time':[maxtime],'wgt':[weight],'bills':bills,'vol':volume})])
    norm=MinMaxScaler().fit(X)
    transform_X=norm.transform(X)
    
    for i in range(len(beat_list)):
        beat=beat_list[i]
        key=beat[5]
        if(len(bike_ols)<=0):
            bike_ols=list(set(grup_of_outlets[key]).intersection(set(outlets_allowed_for_bike[van_id])).intersection(set(grup_outlets_allowed_for_bike[van_id][key])))
        ids=input_data['PARTY_HLL_CODE'].unique()
        neighbour_bikeeligible={}
        nn_dist_temp={}
        nn_dist={}
        for ol in bike_ols:
            dist = distance_matrix[ol][distance_matrix[ol].index.isin(set(ids)-{ol,'Distributor'})]
            dist = pd.Series(dist)
            nearest_index = dist.idxmin() 
            
            if(nearest_index in bike_ols):
                neighbour_bikeeligible[ol]=1
            else:
                neighbour_bikeeligible[ol]=0
            dist = distance_matrix[ol][distance_matrix[ol].index.isin(set(ids)-{ol,'Distributor'})]
            dist = pd.Series(dist)
            nearest_index = dist.idxmin() 
            nn_dist_temp[ol]=dist[nearest_index]
    
        ol_max_dist = max(nn_dist_temp.keys(), key=(lambda k: nn_dist_temp[k]))
        for k in nn_dist_temp.keys():
            nn_dist[k]=nn_dist_temp[k]/nn_dist_temp[ol_max_dist]
            
        isolation_score=0
        for o in set(beat[0])-{'Distributor'}:
            isolation_score+=0.5*(nn_dist[o])+0.5*(neighbour_bikeeligible[o])
        
        isolation_score=isolation_score/(len(beat[0])-1)
        cost=0.50*(0.50*(1-transform_X[i][1])+0.5*(1-transform_X[i][3]))+0.5*(1-isolation_score)
        
        if(cost<min_cost):
            min_beat['sequence']=beat[0]
            min_beat['end_time']=beat[1]
            min_beat['bills']=beat[2]
            min_beat['cum_weight']=beat[3]
            min_beat['van_id']=van_id
            min_beat['del_type']=beat[5]
            min_beat['wgt_sequence']=beat[4]
            min_beat['cum_volume']=beat[6]
            min_beat['vol_sequence']=beat[7]
            min_beat['time_sequence']=beat[8]
            min_beat['bills_cov']=beat[9]
            min_beat['base_pack_cov']=beat[10]
            min_beat['plg']=beat[11]
            min_beat['channel']=beat[12]
            min_cost=cost
            
    return min_beat

def find_ideal_weight_to_be_added(nearest_index,key,max_wgt,max_vol,max_bills):
    #print(nearest_index)
    global outlets_allowed_forvan,van_speed_dict,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict

    #print(max_wgt,max_vol,max_bills)
    allowed_wgt=0
    allowed_vol=0
    l=key.split('_')
    bills_=[]
    #input_data[(input_data['primarychannel'].isin(r['channel'])) & (input_data['area_name'].isin(r['area'])) & (input_data['SERVICING PLG'].isin(plg)) ].copy(deep=True)

    if(len(l)>2):
        sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['SERVICING PLG'].isin(l[2].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)        
    else:
        sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)
        
    sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str)    
    if (key in common_outs_accross_grups.keys()):
        if(nearest_index in [op.split('_')[0] for op in common_outs_accross_grups[key]]):
            for opc in common_outs_accross_grups[key]:
                p=opc.split('_')[1]
                c=opc.split('_')[2]
            if(len(sub_df[(sub_df['primarychannel']==c)  & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==nearest_index)])<1):
                sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c)  & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==nearest_index)].copy(deep=True)])
    sub_df=sub_df[(sub_df['PARTY_HLL_CODE']==nearest_index) & (sub_df['BASEPACK CODE'].isin(list(grup_rem_basepack[key][nearest_index])))].copy(deep=True)

    if(party_pack=='yes'):
        vol_df=sub_df.copy(deep=True)
        vol_df['volume']=vol_df['length']*vol_df['width']*vol_df['height']*input_data['multi_fact']
        vol_dict=vol_df.groupby(['BASEPACK CODE'])['volume'].sum().to_dict()
        wgt_dict=vol_df.groupby(['BASEPACK CODE'])['weight'].sum().to_dict()
    else:
        sku_master['volume']=sku_master['unit_length']*sku_master['unit_breadth']*sku_master['unit_height']
        sku_master['volume'].fillna(0)
        vol_map=dict(zip(sku_master['SKU Code'],sku_master['volume']))
        vol_df=sub_df.copy(deep=True)
        vol_df['volume']=vol_df['BASEPACK CODE'].map(vol_map)
        vol_df['volume']=vol_df['volume']*vol_df['NET_SALES_QTY']
        vol_dict=vol_df.groupby(['BASEPACK CODE'])['volume'].sum().to_dict()
        wgt_dict=vol_df.groupby(['BASEPACK CODE'])['NET_SALES_WEIGHT_IN_KGS'].sum().to_dict()
    
    bill_dict=vol_df.groupby(['BASEPACK CODE'])['BILL_NUMBER'].unique().to_dict()
    bpc_var= pulp.LpVariable.dicts("bpc ",((bpc) for bpc in vol_dict.keys()),lowBound=0,upBound=1,cat='Binary')
    model1 = pulp.LpProblem("cost", pulp.LpMaximize)
    model1 += pulp.lpSum([bpc_var[(bpc)]*wgt_dict[bpc] for bpc in vol_dict.keys()])    

        
    model1 += pulp.lpSum([bpc_var[(bpc)]*vol_dict[bpc] for bpc in vol_dict.keys()])<=max_vol
    model1 += pulp.lpSum([bpc_var[(bpc)]*wgt_dict[bpc] for bpc in vol_dict.keys()])<=max_wgt
    #model1+= pulp.lpSum([bpc_var[(bpc)]*len(list(bill_dict[bpc])) for bpc in vol_dict.keys()])<=max_bills
       
    result=model1.solve(pulp.PULP_CBC_CMD(maxSeconds=100))  
    #print(result)
    if(result==-1):
        print(nearest_index,key,max_wgt,max_vol,max_bills)
    bpcs=[]
    if(result==1):
        for bpc in vol_dict.keys():
            if(bpc_var[(bpc)].varValue==1):
               bpcs.append(bpc)
               allowed_wgt=allowed_wgt+wgt_dict[bpc]
               allowed_vol=allowed_vol+vol_dict[bpc]
    bills_=sub_df[sub_df['BASEPACK CODE'].isin(bpcs)]['BILL_NUMBER'].unique()
    ps=list(sub_df[sub_df['BASEPACK CODE'].isin(bpcs)]['SERVICING PLG'])
    cs=list(sub_df[sub_df['BASEPACK CODE'].isin(bpcs)]['primarychannel'])
    
    return bpcs,list(bills_),allowed_wgt,allowed_vol,ps,cs


def form_normal_beats(key,start_node,van_id,shikar_outs=[]):
    #for now i am writing for only 2 trips
    
    global outlets_allowed_forvan,van_speed_dict,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict
    print(grup_of_outlets.keys())
    ids2=grup_of_outlets[key].copy()
    rem_weight=grup_rem_weight[key]
    rem_volume=grup_rem_volume[key]
    rem_basepacks=grup_rem_basepack[key]
    ol_service_time=grup_ol_service_time[key].copy()
    ol_weights=grup_ol_weights[key].copy()
    ol_volumes=grup_ol_volume[key].copy()
    ol_plg=grup_ol_plg[key].copy()
    ol_channel=grup_ol_channel[key].copy()
    outlet_bill_dict=grup_outlet_bill_dict[key].copy()
    outlet_basepack_dict=grup_outlet_basepack_dict[key].copy()
    
    weight=van_weight_dict[van_id]
    volume=van_volume_dict[van_id]
    bills=van_bill_dict[van_id]
    outs=van_outlet_dict[van_id]
    max_time=van_endtime_dict[van_id]
    if((van_id in van_multitrip_dict.keys()) and (van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') ):
        max_time=multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]
        
    sequence=[]
    wgt_sequence=[0]
    vol_sequence=[0]
    time_sequence=[0]
    plgs=[['']]
    channels=[['']]
    bills_cov=[['']]
    base_pack_cov=[['']]
    cnt=0
    speed=1/van_speed_dict[van_id]
    cnt=cnt+1
    cumulative_weight_seq=0
    cumulative_volume_seq=0
    end_time_seq=0
    ni_to_rs_time=0
    num_stores_seq=0
    num_bills=0
    bills_list=[]
    sequence.append('Distributor') 
    mids=list(set(outlets_allowed_forvan[van_id]).intersection(ids2))
    
    if(start_node in exclusive_outlets):
        mids=list(set(mids).intersection(exclusive_outlets))
    else:
        mids=list(set(mids)-set(exclusive_outlets))
        
    closuretime_overlap=[]        
    while(True):
        prev_sequence=sequence.copy()
        prev_wgt_sequence=wgt_sequence.copy()
        prev_vol_sequence=vol_sequence.copy()
        prev_time_sequence=time_sequence.copy()
        prev_bills_cov=bills_cov.copy()
        prev_base_pack_cov=base_pack_cov.copy()
        prev_plgs=plgs.copy()
        prev_channels=channels.copy()
        #print(prev_sequence)
        prev_end_time_seq=end_time_seq
        prev_ni_to_rs_time=ni_to_rs_time
        prev_num_stores_seq=num_stores_seq
        prev_cumulative_weight_seq=cumulative_weight_seq
        prev_cumulative_volume_seq=cumulative_volume_seq
        if(len(list(set(mids)-set(closuretime_overlap)))<=0):
            #if((start_node=='AP50001205498') and (van_id=='1000')):
            print('mids zero')
            break
        dist = distance_matrix[sequence[-1]][distance_matrix[sequence[-1]].index.isin(set(mids)-set(closuretime_overlap))]
        dist = pd.Series(dist)
        if(cnt==1):
            nearest_index=start_node
            cnt=cnt+1
        else:
            nearest_index = dist.idxmin()
        #print(nearest_index)
        
        if(nearest_index in rem_weight.keys()):
            bills_=list(input_data[(input_data['primarychannel'].isin(ol_channel[nearest_index])) & (input_data['SERVICING PLG'].isin(ol_plg[nearest_index])) & (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepacks[nearest_index]))]['BILL_NUMBER'].unique())
            if((rem_weight[nearest_index]> (weight-cumulative_weight_seq)) or (len(bills_)> (bills-num_bills)) or (rem_volume[nearest_index]> (volume-cumulative_volume_seq))):
                basepacks,bills_,wgt,vol,ps,cs=find_ideal_weight_to_be_added(nearest_index,key,weight-cumulative_weight_seq,volume-cumulative_volume_seq,bills-num_bills)
                if(wgt>0):
                    cumulative_weight_seq = cumulative_weight_seq+wgt
                    wgt_sequence.append(wgt)
                    cumulative_volume_seq=cumulative_volume_seq+vol
                    vol_sequence.append(vol)
                    service_time=((wgt)*ol_service_time[nearest_index])/ol_weights[nearest_index]
                    end_time_seq=end_time_seq+ service_time
                    bills_cov.append(bills_)
                    bills_list.extend(bills_)
                    plgs.append(ps)       
                    channels.append(cs)
                    #num_bills=num_bills+len(bills_)
                    base_pack_cov.append(basepacks)
                else:
                    mids.remove(nearest_index)
                    continue
                    
            else:
               cumulative_weight_seq += rem_weight[nearest_index] 
               wgt_sequence.append(rem_weight[nearest_index])
               cumulative_volume_seq=cumulative_volume_seq+rem_volume[nearest_index]
               vol_sequence.append(rem_volume[nearest_index])
               service_time=(rem_weight[nearest_index]*ol_service_time[nearest_index])/ol_weights[nearest_index]
               end_time_seq=end_time_seq+ service_time
               bills_=list(input_data[(input_data['primarychannel'].isin(ol_channel[nearest_index])) & (input_data['SERVICING PLG'].isin(ol_plg[nearest_index])) & (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepacks[nearest_index]))]['BILL_NUMBER'].unique())
               bills_cov.append(bills_)
               plgs.append(list(input_data[(input_data['primarychannel'].isin(ol_channel[nearest_index])) & (input_data['SERVICING PLG'].isin(ol_plg[nearest_index])) & (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepacks[nearest_index]))]['SERVICING PLG'] ))       
               channels.append(list(input_data[(input_data['primarychannel'].isin(ol_channel[nearest_index])) & (input_data['SERVICING PLG'].isin(ol_plg[nearest_index])) & (input_data['PARTY_HLL_CODE']==nearest_index) & (input_data['BASEPACK CODE'].isin(rem_basepacks[nearest_index]))]['primarychannel'] ))  
               #num_bills=num_bills+len(bills_)
               bills_list.extend(bills_)
               base_pack_cov.append(rem_basepacks[nearest_index])                

        else:
            cumulative_weight_seq += ol_weights[nearest_index]
            cumulative_volume_seq +=ol_volumes[nearest_index]
            wgt_sequence.append(ol_weights[nearest_index])
            #cumulative_volume_seq=cumulative_volume_seq+ol_volumes[nearest_index]
            vol_sequence.append(ol_volumes[nearest_index])
            service_time=ol_service_time[nearest_index]
            end_time_seq=end_time_seq+ service_time
            bills_=list(outlet_bill_dict[nearest_index])
            bills_cov.append(bills_)
            #num_bills=num_bills+len(bills_)
            bills_list.extend(bills_)
            base_pack_cov.append(list(outlet_basepack_dict[nearest_index]))
            plgs.append(ol_plg[nearest_index])
            channels.append(ol_channel[nearest_index])
        
        distance = dist[nearest_index]  
        travel_time = math.ceil(distance * speed)
        st_time=end_time_seq
        end_time_seq = end_time_seq + travel_time
        fi_time=end_time_seq
        
        '''
        if(nearest_index in list(ol_closure_time.keys())):
                if(((st_time > ol_closure_time[nearest_index][0]) & (st_time < ol_closure_time[nearest_index][1])) | ((fi_time > ol_closure_time[nearest_index][0]) & (fi_time < ol_closure_time[nearest_index][1]))):
                    if((start_node=='AP50001205498') and (van_id=='1000')):
                        print('ol_closure time')
                    sequence=prev_sequence.copy()
                    wgt_sequence=prev_wgt_sequence.copy()
                    vol_sequence=prev_vol_sequence.copy()
                    time_sequence=prev_time_sequence.copy()
                    bills_cov=prev_bills_cov.copy()
                    base_pack_cov=prev_base_pack_cov.copy()
                    plgs=prev_plgs.copy()
                    channels=prev_channels.copy()
                    #print(prev_sequence)
                    end_time_seq=prev_end_time_seq
                    ni_to_rs_time=prev_ni_to_rs_time
                    num_stores_seq=prev_num_stores_seq
                    cumulative_weight_seq=prev_cumulative_weight_seq
                    cumulative_volume_seq=prev_cumulative_volume_seq
                    closuretime_overlap.append(nearest_index)
                    continue
        '''
        time_sequence.append(service_time+travel_time)
        ni_to_rs_dist = distance_matrix[nearest_index][sequence[0]]
        ni_to_rs_time = math.ceil(ni_to_rs_dist * speed)
        num_stores_seq=num_stores_seq+1
        sequence.append(nearest_index)
        num_bills=len(set(bills_list))
        mids.remove(nearest_index)
        
        for ots in closuretime_overlap:
            if(end_time_seq > ol_closure_time[ots][1]):
                closuretime_overlap.remove(ots)
        
        if((cumulative_weight_seq<=weight) and (end_time_seq+ni_to_rs_time<= max_time) and (num_stores_seq<=outs) and (cumulative_volume_seq<=volume) and (num_bills<=bills)):
            continue
        else: 
            
            break
    prev_sequence.append('Distributor')
    prev_wgt_sequence.append(0)
    prev_vol_sequence.append(0)
    prev_time_sequence.append(math.ceil(distance_matrix[prev_sequence[-2]]['Distributor'] * speed))
    prev_bills_cov.append([''])
    prev_base_pack_cov.append([''])
    prev_plgs.append([''])
    prev_channels.append([''])
    
    return prev_sequence,prev_end_time_seq+prev_ni_to_rs_time,prev_num_stores_seq,prev_cumulative_weight_seq,prev_wgt_sequence,prev_cumulative_volume_seq,prev_vol_sequence,prev_time_sequence,prev_bills_cov,prev_base_pack_cov,prev_plgs,prev_channels


def find_best_beat(beat_list,van_id,cnt=0,typ='Owned'):
    '''
    sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,key,cumulative_volume_seq,vol_sequence,time_sequence
    '''
    global outlets_allowed_forvan,van_speed_dict,multitripvan_rem_time,van_cutoff_dict,van_multitrip_dict,van_weight_dict,van_bill_dict,van_endtime_dict,van_volume_dict,van_outlet_dict,grup_of_outlets,grup_rem_weight,grup_rem_basepack,grup_ol_service_time,grup_ol_weights,grup_ol_plg,grup_ol_channel,grup_outlet_bill_dict,grup_outlet_basepack_dict

    min_beat={}
    min_cost=100000
    maxtime=van_endtime_dict[van_id]
    weight=van_weight_dict[van_id]
    bills=van_bill_dict[van_id]
    volume=van_volume_dict[van_id]
    X=pd.DataFrame()
    for beat in beat_list:
        X=pd.concat([X,pd.DataFrame({'time':[beat[1]],'wgt':[beat[3]],'bills':[beat[2]],'volume':[beat[6]]})])
    X=pd.concat([X,pd.DataFrame({'time':[0],'wgt':[0],'bills':[0],'volume':[0]})])
    X=pd.concat([X,pd.DataFrame({'time':[maxtime],'wgt':[weight],'bills':[bills],'volume':[volume]})])

    from sklearn.preprocessing import MinMaxScaler
    norm=MinMaxScaler().fit(X)
    transform_X=norm.transform(X)
    
    for i in range(len(beat_list)):
        beat=beat_list[i]
        #[sequence,end_time_seq,num_stores_seq,cumulative_weight_seq]
        #cost=0.25*(1-transform_X[i][0])+0.50*(1-transform_X[i][1])+0.25*(transform_X[i][2])
        cost=0.50*(1-transform_X[i][1])+0.5*(1-transform_X[i][3])
        if(cost<min_cost):
            min_beat['sequence']=beat[0]
            min_beat['end_time']=beat[1]
            min_beat['bills']=beat[2]
            min_beat['cum_weight']=beat[3]
            if(cnt>0 and typ=='Rented'):
                min_beat['van_id']=van_id+'_'+str(cnt)+'_'+typ
            else:
                min_beat['van_id']=van_id
            min_beat['del_type']=beat[5]
            min_beat['wgt_sequence']=beat[4]
            min_beat['cum_volume']=beat[6]
            min_beat['vol_sequence']=beat[7]
            min_beat['time_sequence']=beat[8]
            min_beat['bills_cov']=beat[9]
            min_beat['base_pack_cov']=beat[10]
            min_beat['plg']=beat[11]
            min_beat['channel']=beat[12]
            min_cost=cost
    return min_beat

def add_to_output(beat):
    global output_df
    output_df=pd.concat([output_df,pd.DataFrame({'path':beat['sequence'],'endtime':[beat['end_time']]*len(beat['sequence']),'num_bills':[beat['bills']]*len(beat['sequence']),'cum_weight':[beat['cum_weight']]*len(beat['sequence']),'van_id':[beat['van_id']]*len(beat['sequence']),'weights':beat['wgt_sequence'],'cum_volume':[beat['cum_volume']]*len(beat['sequence']),'volumes':beat['vol_sequence'],'del_type':[beat['del_type']]*len(beat['sequence']),'time':beat['time_sequence'],'bill_numbers':beat['bills_cov'],'Basepack':beat['base_pack_cov'],'plg':beat['plg'],'channel':beat['channel']})])

distance_matrix=pd.DataFrame()

grup_outlets_allowed_for_bike={}

iteration=0
output_stack=[]
common_outs_accross_grups={}
grups_common_outs={}
owned_bikes_to_fill=[]
rented_bikes_to_fill=[]
owned_van_order_tofill=[]
rental_van_order_tofill=[]

van_volume_dict={}
van_weight_dict={}
van_cost_dict={}
van_bill_dict={}
van_endtime_dict={}
van_multitrip_dict={}
van_cutoff_dict={}
van_outlet_dict={}
van_plg_mapping={}
van_speed_dict={}
van_trip_dict={}
van_fixedrate_dict={}
van_baserate_dict={}
van_perkmrate_dict={}
van_perhourrate_dict={}
outlets_allowed_forvan={}

grup_of_outlets={}

grup_ol_weights={}
grup_ol_service_time={}
grup_ol_volume={}
grup_ol_plg={}
grup_ol_channel={}
grup_rem_weight={}
grup_rem_volume={}
grup_rem_basepack={}
grup_high_demand_outs={}
grups={}    
grup_outlet_bill_dict={}
grup_outlet_basepack_dict={}
grup_olplg_weights={}
grup_olplg_service_time={}
grup_olplg_volume={}
grup_olplg_bill_dict={}
grup_olplg_basepack_dict={}
input_data=pd.DataFrame()
outlets_allowed_for_bike = {}
bike_beats = []
exclusive_outlets=[]
multitripvan_rem_time={}
ol_closure_time={}
party_pack='no'
sku_master=pd.DataFrame()
output_df=pd.DataFrame()
outs_allowed_forvan_copy={}
grup_outlets_allowed_for_bike={}

def main(date,rscode,transfilename,masterfilename):
	try:
    
		global grup_outlets_allowed_for_bike,output_df,outs_allowed_forvan_copy,distance_matrix,grup_outlets_allowed_for_bike,iteration,output_stack,common_outs_accross_grups,grups_common_outs,owned_bikes_to_fill,rented_bikes_to_fill,owned_van_order_tofill,rental_van_order_tofill,van_volume_dict,van_weight_dict,van_cost_dict,van_bill_dict,van_endtime_dict,van_multitrip_dict,van_cutoff_dict,van_outlet_dict,van_plg_mapping,van_speed_dict,van_trip_dict,van_fixedrate_dict,van_baserate_dict,van_perkmrate_dict,van_perhourrate_dict,outlets_allowed_forvan,grup_of_outlets,grup_ol_weights,grup_ol_service_time,grup_ol_volume,grup_ol_plg,grup_ol_channel,grup_rem_weight,grup_rem_volume,grup_rem_basepack,grup_high_demand_outs,grups,grup_outlet_bill_dict,grup_outlet_basepack_dict,grup_olplg_weights,grup_olplg_service_time,grup_olplg_volume,grup_olplg_bill_dict,grup_olplg_basepack_dict,input_data,outlets_allowed_for_bike,bike_beats,exclusive_outlets,multitripvan_rem_time,ol_closure_time,party_pack,sku_master
		import datetime
		
		Input=pd.read_excel(masterfilename,sheet_name='hllcode_master_constraint')
		orders_data= pd.read_excel(transfilename,sheet_name='transactional_data')
		orders_data=orders_data[orders_data['bill_date'].astype(str)==date].reset_index(drop=True)
		
		Input['outlet_latitude'].fillna(0,inplace=True)
		lat0=Input[Input['outlet_latitude']==0]
		Input=Input[~Input['partyhll_code'].isin(lat0['partyhll_code'].unique())]
		Input=Input[Input['partyhll_code'].isin(orders_data['partyhll_code'].unique())]
		if(len(Input)<=0):
		     return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
		lat_long_data = Input[['partyhll_code','outlet_latitude','outlet_longitude']].drop_duplicates()
		lat_long_data.dropna(inplace = True)
		rs_lat=Input['rs_latitude'].unique()[0]
		rs_long=Input['rs_longitude'].unique()[0]
		lat_long_data = lat_long_data.append(pd.DataFrame([['Distributor',rs_lat,rs_long]], columns=lat_long_data.columns))
		lat_long_data.rename(columns = {'partyhll_code':'PartyHLLCode','outlet_latitude' : 'Latitude', 'outlet_longitude' : 'Longitude'}, inplace = True)
		lat_long_path = ''
		lat_long_data.to_excel(lat_long_path + str(rscode) + '_lat_long.xlsx',sheet_name = 'Sheet1', index = False)

		path=''
		
		path_to_odl_graphs = path+'/ODL_graphs'
		
		if(path == ''):
			input_file_path = os.getcwd()
			print(input_file_path)
			input_file_path = input_file_path 
		else:
			input_file_path = path
		print(input_file_path)
		
		def run_odl(RSCODE, input_file_path, path_to_odl_graphs):
			output_file_name = str(RSCODE) + '_Distances.csv'
			print(output_file_name)
			tree = ET.parse(input_file_path + '/runmatrix.odlx')
			root = tree.getroot()
			root.find('Instruction').find('Config').find('travelMatrixExporterConfig').find('exportFilename').text = str(output_file_name)
			tree.write(input_file_path + '/runmatrix_'+str(RSCODE)+'.odlx')
			subprocess.call(['java', '-Xmx8G','-cp','com.opendoorlogistics.connect.jar:com.opendoorlogistics.studio.jar','com.opendoorlogistics.connect.CommandLine','-inputdir',input_file_path + '/' ,'-ix',str(RSCODE) + '_lat_long.xlsx','-r','runmatrix_'+str(RSCODE)+'.odlx'])
		   
		
		
		try:
			distance_matrix = pd.read_csv(rscode + '_Distances.csv', sep = '\t')
			   
		except:
			run_odl(rscode,input_file_path, path_to_odl_graphs)
			distance_matrix = pd.read_csv(rscode + '_Distances.csv', sep = '\t')
		# if(not(all([1 if o in distance_matrix.index else 0 for o in list(Input['partyhll_code'].unique())]))):
		#     run_odl(rscode,input_file_path, path_to_odl_graphs)
		#     distance_matrix = pd.read_csv(rscode + '_Distances_1.csv', sep = '\t')
				 
		distance_matrix = get_distances_matrix_format(distance_matrix)
		distance_matrix.index = distance_matrix.index.astype(str)
		distance_matrix.columns = distance_matrix.columns.astype(str)
		
		
		#orders_data=orders_data[orders_data['bill_date'].astype(str)=='2019-03-06'].reset_index(drop=True)
		
		van_details = pd.read_excel(masterfilename, sheet_name = 'vehicle_master')
		rs_master = pd.read_excel(masterfilename, sheet_name = 'rental_vehicle_master')
		sku_master = pd.read_excel(masterfilename,sheet_name='product_master')
		outlet_data = pd.read_excel(masterfilename,sheet_name='hllcode_master_constraint')
		outlet_master = pd.read_excel(masterfilename,sheet_name='outlet_master')
		service_time_details = pd.read_excel(masterfilename,sheet_name='service_time')
		party_packing_details = pd.read_excel(masterfilename,sheet_name='party_packing_details')
		plg_clubbing = pd.read_excel(masterfilename,sheet_name='clubbing_details')
		
		if(len(orders_data)<=0):
		     return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
		orders_data=orders_data[~orders_data['servicing_plg'].isna()].reset_index(drop=True)
		
		
		party_pack=party_packing_details['party_packing'].unique()[0]
		sku_master['SKU Code']=sku_master['basepack_code'].astype(str)
		
		
		orders_data=orders_data[orders_data['partyhll_code'].isin(list(distance_matrix.columns))]
		outlet_data=outlet_data[outlet_data['partyhll_code'].isin(list(orders_data['partyhll_code'].unique()))]
		outlet_data.drop_duplicates(subset =['partyhll_code'], keep = 'first', inplace = True)
		outlet_master=outlet_master[outlet_master['partyhll_code'].isin(list(orders_data['partyhll_code'].unique()))]
		outlet_master=outlet_master.sort_values(by='area_name').reset_index(drop=True)
		outlet_master.drop_duplicates(subset =['partyhll_code'], keep = 'first', inplace = True)
		cols_to_use = outlet_master.columns.difference(outlet_data.columns)
		outlet_data=pd.merge(outlet_data, outlet_master[list(cols_to_use)+['partyhll_code']], on = 'partyhll_code', how = 'left')
		
		outlet_data['area_name']=outlet_data['area_name'].astype(str).str.lower()
		outlet_data['area_name']=outlet_data['area_name'].replace(['nan'],'All')
		
		#outlet_data['area_name'].fillna('All',inplace=True)
		orders_data=orders_data[orders_data['partyhll_code'].isin(list(outlet_data['partyhll_code'].unique()))]
		orders_data = pd.merge(orders_data, outlet_data, left_on = 'partyhll_code', right_on = 'partyhll_code', how = 'left')
		party_packing_details['crate_type']=party_packing_details['crate_type'].astype(str)
		party_packing_details['crate_type']=party_packing_details['crate_type'].str.lower()
		party_packing_details['crate_type']=party_packing_details['crate_type'].str[:7]
		
		ord_data_copy=orders_data.copy(deep=True)
		if(party_pack=='yes'):    
			orders_data['basepack_code']=orders_data['basepack_code'].astype(str)
			orders_data['crate_no'] = orders_data['crate_no'].astype(str).str.lower()
			orders_data.loc[orders_data['basepack_code']=='-','basepack_code']=orders_data['crate_no']    
			orders_data['crate_no']=orders_data['crate_no'].str[:7]
			len_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_length'])))
			wid_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_width'])))
			h_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_height'])))
			wt_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_weight'])))
			#orders_data['new_weight']=orders_data['crate_no'].map(wt_dict)
			orders_data['length']=orders_data['crate_no'].map(len_dict)
			orders_data['width']=orders_data['crate_no'].map(wid_dict)
			orders_data['height']=orders_data['crate_no'].map(h_dict)
			orders_data['length']=orders_data['length']*30
			orders_data['width']=orders_data['width']*30
			orders_data['height']=orders_data['height']*30
			orders_data['weight']=orders_data['weight'].astype(float)
			orders_data_v1=orders_data[orders_data['crate_no']=='nan'].reset_index(drop=True)
			orders_data_v2=orders_data[orders_data['crate_no']!='nan'].reset_index(drop=True)
			orders_data_v2['new_weight']=orders_data_v2['crate_no'].map(wt_dict)
			orders_data_v1['new_weight']=0
			orders_data_v2['new_weight']=orders_data_v2['new_weight']/1000
			orders_data_v2['multi_fact']=orders_data_v2['weight']/orders_data_v2['new_weight']
			orders_data_v1.loc[orders_data_v1['crate_no'].isna(),'crate_no']=orders_data_v1['basepack_code']
			orders_data_v1['multi_fact']=1
			sku_master['basepack_code']=sku_master['basepack_code'].astype(str)
			sku_len_dict=dict(list(zip(sku_master['basepack_code'],sku_master['unit_length'])))
			sku_w_dict=dict(list(zip(sku_master['basepack_code'],sku_master['unit_breadth'])))
			sku_h_dict=dict(list(zip(sku_master['basepack_code'],sku_master['unit_height'])))
			orders_data_v1['length']=orders_data_v1['basepack_code'].map(sku_len_dict)
			orders_data_v1['width']=orders_data_v1['basepack_code'].map(sku_w_dict)
			orders_data_v1['height']=orders_data_v1['basepack_code'].map(sku_h_dict)
			orders_data=pd.concat([orders_data_v1,orders_data_v2]).reset_index(drop=True)
			orders_data['multi_fact']=orders_data['multi_fact'].fillna(0)
			orders_data['multi_fact']=np.where(orders_data['multi_fact']<=1,1,orders_data['multi_fact'])
			orders_data['multi_fact']=orders_data['multi_fact'].apply(np.floor)
			orders_data['multi_fact']=1
			
		
		#orders_data=orders_data[orders_data['self_pickup'].astype(str)=='no'].copy(deep=True)
		orders_data=orders_data[(orders_data['self_pickup'].isna()) | (orders_data['self_pickup'].astype(str)=='no')].copy(deep=True)
		orders_data['outlet_type']=orders_data['outlet_type'].replace(['rest of retail'],'retail')
		#orders_data['primarychannel'] = orders_data['outlet_type'].str.lower()
		orders_data['primarychannel'] = orders_data['outlet_type']
		
		
		#orders_data['opc']=orders_data['partyhll_code']+'_'+orders_data['servicing_plg'].astype(str)+'_'+orders_data['primarychannel']
		#if(party_pack=='yes'):
		#    opc_wgt_map=orders_data.groupby(['opc'])['weight'].sum().to_dict()
		#else:
		#    opc_wgt_map=orders_data.groupby(['opc'])['net_sales_weight_kgs'].sum().to_dict()
		#orders_data['opc_wgt']=orders_data['opc'].map(opc_wgt_map)
		#orders_data=orders_data[orders_data['opc_wgt']>0.10]
		
		haver_lat_long=orders_data[['partyhll_code','outlet_latitude', 'outlet_longitude']].drop_duplicates().reset_index(drop=True)
		haver_lat_long.loc[haver_lat_long.shape[0],['partyhll_code','outlet_latitude', 'outlet_longitude']]=['Distributor',orders_data['rs_latitude'][0],orders_data['rs_longitude'][0]]
		dist_mat_v1=np.zeros([len(haver_lat_long),len(haver_lat_long)])
		
		
		for i in range(0,len(haver_lat_long)):
			for j in range(i+1,len(haver_lat_long)):
				
				lat1=haver_lat_long['outlet_latitude'][i]
				lon1=haver_lat_long['outlet_longitude'][i]
				lat2=haver_lat_long['outlet_latitude'][j]
				lon2=haver_lat_long['outlet_longitude'][j]
				
				lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
				dlon=lon2-lon1
				dlat=lat2-lat1
				a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
				c = 2 * asin(sqrt(a))
				r = 6371
				d=c*r*1.3
				#print(d)
				dist_mat_v1[i][j]=d
				dist_mat_v1[j][i]=d                
		
		haver_distance_df=pd.DataFrame(dist_mat_v1,index=haver_lat_long['partyhll_code'],columns=haver_lat_long['partyhll_code'])
		dist_mat=distance_matrix.loc[distance_matrix.index.isin(list(haver_distance_df.index)),distance_matrix.index.isin(list(haver_distance_df.columns))]
		def have_odl():    
			for i in dist_mat.columns:
				
				for j in dist_mat.index:
					odl_dist=dist_mat[i][j]
					haversine_dist=haver_distance_df[i][j]
					if odl_dist>haversine_dist:
						dist_mat[i][j]=haversine_dist
						dist_mat[j][i]=haversine_dist
			return dist_mat
		
		dist_mat_v2=have_odl()
		distance_matrix=dist_mat_v2.copy(deep=True)
		
		
		plg_clubbing['area']=plg_clubbing['area'].astype(str).str.lower()
		di={'all':'All'}
		plg_clubbing=plg_clubbing.replace({"area": di}).copy(deep=True)
		plg_clubbing.fillna('Nil',inplace=True)
		di={'All':','.join(list(set([c for cs in plg_clubbing['channel'] for c in cs.split(',') if c!='All'] )))}
		if(len(di['All'])<=0):
			di['All']=','.join(orders_data['primarychannel'].unique())
		plg_clubbing=plg_clubbing.replace({"channel": di}).copy(deep=True)
		for col in ['channel','area']:
			plg_clubbing[col]=plg_clubbing[col].apply(lambda x:x.split(','))
		for col in ['clubbing_1','clubbing_2','clubbing_3','clubbing_4','clubbing_5','clubbing_6','clubbing_7']:
			plg_clubbing[col]=plg_clubbing[col].apply(lambda x:x.split(',')) 
			plg_clubbing_=pd.DataFrame()
		for i,r in plg_clubbing.iterrows():
			#print(r)
			if(r['clubbing_1']==['All']):
				r['groups']=[]
			else:
				r['groups']=[r[c] for c in ['clubbing_1','clubbing_2','clubbing_3','clubbing_4','clubbing_5','clubbing_6'] if r[c]!=['Nil']]
			plg_clubbing_=pd.concat([plg_clubbing_,pd.DataFrame(r).T])
		
		
		orders_data['bill_date']= orders_data['bill_date'].max()
		if(list(orders_data['delivery_expectation_from_bill_date'].unique())[0]=='N+1'):
			orders_data['expected_del_date']= orders_data['bill_date'] + pd.Timedelta(days=1)
		else:
			orders_data['expected_del_date']= orders_data['bill_date'] + pd.Timedelta(days=2)
		new = orders_data['holiday_calendar'].astype(str).str.split("-", n = 1, expand = True) 
		if new.shape[1]==1:
			orders_data["Month"]= new[0] 
			orders_data["Day"]= new[0]
		else:
			orders_data["Month"]= new[0] 
			orders_data["Day"]= new[1]
			
		orders_data=orders_data[~((orders_data['expected_del_date'].dt.day==orders_data["Day"].astype(float)) & (orders_data['expected_del_date'].dt.month==orders_data["Month"].astype(float))) ]
		orders_data=orders_data[~(orders_data['expected_del_date'].dt.day==orders_data['monthly_holiday'].astype(float))]
		
		
		daydict={'M':0,'T':1,'W':2,'TH':3,'F':4,'S':5,'SU':6}
		orders_data['outlet_weekly_holiday']= orders_data['outlet_weekly_holiday'].map(daydict).astype(int)
		
		#orders_data=orders_data[~((orders_data['expected_del_date'].dt.weekday==orders_data['outlet_weekly_holiday']))]
		
		
		
		exclusive_outlets=orders_data[orders_data['exclusivity']=='yes']['partyhll_code'].unique()
		
		#orders_data.columns
		input_data=orders_data.copy(deep=True)
		input_data['Category'] = 'all'
		ids=list(input_data['partyhll_code'].unique())
		#input_data['SERVICING PLG'].unique()
		
		if party_pack=='yes':
			input_data['PARTY_HLL_CODE']=input_data['partyhll_code']
			input_data['BASEPACK CODE']=input_data['basepack_code']
			input_data['SERVICING PLG']=input_data['servicing_plg']
			input_data['BILL_NUMBER']=input_data['bill_number']
			input_data['NET_SALES_WEIGHT_IN_KGS']=input_data['weight']
			input_data['NET_SALES_QTY']=0
			di={'DETS-S':'DETS','DF-S':'D+F', 'PP-S':'PP', 'PPB-S':'PP-B','HUL-W':'HUL'}
			input_data=input_data.replace({"SERVICING PLG": di}).copy(deep=True)
			input_data=input_data[~input_data["SERVICING PLG"].isna()].copy(deep=True)
			input_data['BASEPACK CODE']=input_data['BASEPACK CODE'].astype(str)
		else:
			
			input_data['PARTY_HLL_CODE']=input_data['partyhll_code']
			input_data['BASEPACK CODE']=input_data['basepack_code']
			input_data['SERVICING PLG']=input_data['servicing_plg']
			input_data['BILL_NUMBER']=input_data['bill_number']
			input_data['NET_SALES_WEIGHT_IN_KGS']=input_data['net_sales_weight_kgs']
			input_data['NET_SALES_QTY']=input_data['net_sales_qty']
			di={'DETS-S':'DETS','DF-S':'D+F', 'PP-S':'PP', 'PPB-S':'PP-B','HUL-W':'HUL'}
			input_data=input_data.replace({"SERVICING PLG": di}).copy(deep=True)
			input_data=input_data[~input_data["SERVICING PLG"].isna()].copy(deep=True)
			input_data['BASEPACK CODE']=input_data['BASEPACK CODE'].astype(str)
		
		#os.chdir(r'D:\Dynamic Routing\\')
		
		lat_long=outlet_data.loc[~outlet_data['outlet_latitude'].isna(),['partyhll_code','outlet_latitude','outlet_longitude']].drop_duplicates().reset_index(drop=True)
		
		lat_long.loc[lat_long.shape[0],['partyhll_code','outlet_latitude','outlet_longitude']]=['Distributor',outlet_data['rs_latitude'][0],outlet_data['rs_longitude'][0]]
		
		
		
		#os.chdir(r'D:\Dynamic Routing\\')
		#lat_long=pd.read_excel(str(rscode) + '_lat_long.xlsx')
		#lat_long.drop_duplicates(subset =['PartyHLLCode'], keep = 'first', inplace = True)
		#lat_long.set_index('PartyHLLCode',inplace=True) 
		#lat_long=lat_long[lat_long.index.isin(['Distributor'] + list(ids))]
		#lat_long=lat_long.T
		#lat_long=lat_long[['Distributor'] + list(ids)]
		#lat_long=lat_long.T
		
		
		iteration=0
		output_stack=[]
		common_outs_accross_grups={}
		grups_common_outs={}
		owned_bikes_to_fill=[]
		rented_bikes_to_fill=[]
		owned_van_order_tofill=[]
		rental_van_order_tofill=[]
		
		
		van_details['van_id']=van_details['vehicle_name']
		
		van_details['multi_trips_cutoff_time']=van_details['multi_trips_cutoff_time'].fillna(0)
		van_details['multi_trips_cutoff_time']=np.where(van_details['multi_trips_cutoff_time'].astype(str).str.lower().isin(['NA','na']), 0, van_details['multi_trips_cutoff_time'])
		van_details['multi_trips_cutoff']=van_details['multi_trips_cutoff_time'].astype(str).str.split(':').apply(lambda x: float(x[0]))
		van_details['multi_trips_cutoff']=van_details['multi_trips_cutoff'].astype(int)
		van_details['multi_trips_cutoff']=(van_details['multi_trips_cutoff']-9)*60
		van_details['second_trip_cost']=np.where(van_details['second_trip_cost'].astype(str).str.lower().isin(['NA','na']), 1, van_details['second_trip_cost'])
		van_details['second_trip_cost']=van_details['second_trip_cost'].fillna(1)
		van_details['second_trip_cost2']=(van_details['second_trip_cost']-1)*van_details['cost_per_day_rupees']
		
		multitrip_df=van_details[van_details['multi_trips'].str.lower()=='yes']
		non_multitrip_df=van_details[van_details['multi_trips'].str.lower()=='no']
		skip_trips={}
		new_multitrip_df=pd.DataFrame()
		for i,r in multitrip_df.iterrows():
			if(r['multi_trips_cutoff']>0):
				n_r=r.copy(deep=True)
				n_r_=r.copy(deep=True)
				r['van_id']=r['van_id']+'_1'
				n_r['van_id']=n_r['van_id']+'_2'
				r['operating_hours_minutes']=r['multi_trips_cutoff']
				n_r['operating_hours_minutes']=480-n_r['multi_trips_cutoff']-45
				n_r['cost_per_day_rupees']=r['second_trip_cost2']
				new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(r).T])
				new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(n_r).T])
				if(n_r_['vehicle_capacity_kgs']!=30):
					n_r_['van_id']=n_r_['van_id']+'_3'
					new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(n_r_).T])
					skip_trips[n_r_['van_id']]=True
				skip_trips[n_r['van_id']]=False
				skip_trips[r['van_id']]=False
			else:
				#bike
				if((r['count_of_multi_trips']>2) and (r['vehicle_capacity_kgs']!=30)):
					n_r=r.copy(deep=True)
					r['van_id']=r['van_id']+'_1'
					r['operating_hours_minutes']=480
					new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(r).T])
					for i in range(1,int(r['count_of_multi_trips'])):
						n_r_=n_r.copy(deep=True)
						n_r_['van_id']=n_r_['van_id']+'_'+str(i+1)
						#n_r_['operating_hours_minutes']=480
						n_r_['cost_per_day_rupees']=n_r['second_trip_cost2']
						new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(n_r_).T])
						skip_trips[n_r_['van_id']]=False
				else:
					n_r=r.copy(deep=True)
					r['van_id']=r['van_id']+'_1'
					r['operating_hours_minutes']=480
					new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(r).T])
					for i in range(1,int(r['count_of_multi_trips'])):
						n_r_=n_r.copy(deep=True)
						n_r_['van_id']=n_r_['van_id']+'_'+str(i+1)
						#n_r_['operating_hours_minutes']=480
						n_r_['cost_per_day_rupees']=0
						new_multitrip_df=pd.concat([new_multitrip_df,pd.DataFrame(n_r_).T])
						skip_trips[n_r_['van_id']]=False
						
		van_details_prev=van_details.copy(deep=True)
		van_details=pd.concat([new_multitrip_df,non_multitrip_df]).copy(deep=True)
		
		
		van_details['cut_off']=np.where(van_details['multi_trips_cutoff']<0, 'no', 'yes')
		van_details['no_trips']=van_details['count_of_multi_trips'].fillna(1)
		van_details['max_capacity']=(van_details['vehicle_capacity_kgs'])*(van_details['weight_factor'])
		
		#sequential
		van_details_prev['no_trips']=van_details_prev['count_of_multi_trips'].fillna(1)
		van_details_prev['trips']=np.where(van_details_prev['no_trips']>1,2,1)
		van_details_prev['delivery_quotient']=(van_details_prev['cost_per_day_rupees']+van_details_prev['second_trip_cost2'])/(van_details_prev['operating_hours_minutes']*van_details_prev['trips']*van_details_prev['weight_factor']*van_details_prev['vehicle_capacity_kgs'])
		van_details_prev['max_capacity']=(van_details_prev['vehicle_capacity_kgs'])*(van_details_prev['weight_factor'])
		
		for van_id in list(van_details_prev[~(van_details_prev['order_type_exclusivity'].isna()) & (van_details_prev['vehicle_capacity_kgs']!=30)].sort_values(by=['vehicle_capacity_kgs','delivery_quotient'],ascending=[False,True])['van_id'].unique()):
			if(van_id not in owned_van_order_tofill):
				owned_van_order_tofill.append(van_id)
		for van_id in list(van_details_prev[~(van_details_prev['plg_vehicle_mapping'].isna()) & (van_details_prev['vehicle_capacity_kgs']!=30)].sort_values(by=['vehicle_capacity_kgs','delivery_quotient'],ascending=[False,True])['van_id'].unique()):
			if(van_id not in owned_van_order_tofill):
				owned_van_order_tofill.append(van_id)
		for van_id in list(van_details_prev[~(van_details_prev['area_assignment'].isna()) & (van_details_prev['vehicle_capacity_kgs']!=30)].sort_values(by=['vehicle_capacity_kgs','delivery_quotient'],ascending=[False,True])['van_id'].unique()):
			if(van_id not in owned_van_order_tofill):
				owned_van_order_tofill.append(van_id)
		
		owned_van_order_tofill.extend(list(van_details_prev[(van_details_prev['area_assignment'].isna()) & (van_details_prev['order_type_exclusivity'].isna()) & (van_details_prev['plg_vehicle_mapping'].isna()) & (van_details_prev['vehicle_capacity_kgs']!=30) & (van_details_prev['rental_type'].isin(['Leased - Long term','Own']))].sort_values(by=['vehicle_capacity_kgs','delivery_quotient'],ascending=[False,True])['van_id']))
		#owned_van_order_tofill=list(set(owned_van_order_tofill))
		vans=list(van_details['van_id'].unique())
		owned_van_order_tofill_new=[]
		for v in owned_van_order_tofill:
			for v2 in vans:
				if v==v2.split('_')[0]:
					owned_van_order_tofill_new.append(v2)
		
		for van_id in list(van_details_prev[~(van_details_prev['order_type_exclusivity'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30) & (van_details_prev['rental_type'].isin(['Leased - Long term','Own']))]['van_id'].unique()):
			owned_bikes_to_fill.append(van_id)
		for van_id in list(van_details_prev[~(van_details_prev['plg_vehicle_mapping'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30) & (van_details_prev['rental_type'].isin(['Leased - Long term','Own']))]['van_id'].unique()):
			owned_bikes_to_fill.append(van_id)
		for van_id in list(van_details_prev[~(van_details_prev['area_assignment'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30) & (van_details_prev['rental_type'].isin(['Leased - Long term','Own']))]['van_id'].unique()):
			owned_bikes_to_fill.append(van_id)   
		owned_bikes_to_fill.extend(list(van_details_prev[(van_details_prev['area_assignment'].isna()) & (van_details_prev['order_type_exclusivity'].isna()) & (van_details_prev['plg_vehicle_mapping'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30)& (van_details_prev['rental_type'].isin(['Leased - Long term','Own']))]['van_id']))
		vans=list(van_details['van_id'].unique())
		owned_bikes_to_fill_new=[]
		for v in owned_bikes_to_fill:
			for v2 in vans:
				if v in v2:
					owned_bikes_to_fill_new.append(v2)
		
		
		for van_id in list(van_details_prev[~(van_details_prev['order_type_exclusivity'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30) & (van_details_prev['rental_type']=='Rented - Short term')]['van_id'].unique()):
			rented_bikes_to_fill.append(van_id)
		for van_id in list(van_details_prev[~(van_details_prev['plg_vehicle_mapping'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30) & (van_details_prev['rental_type']=='Rented - Short term')]['van_id'].unique()):
			rented_bikes_to_fill.append(van_id)
		for van_id in list(van_details_prev[~(van_details_prev['area_assignment'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30) & (van_details_prev['rental_type']=='Rented - Short term')]['van_id'].unique()):
			rented_bikes_to_fill.append(van_id)   
		rented_bikes_to_fill.extend(list(van_details_prev[(van_details_prev['area_assignment'].isna()) & (van_details_prev['order_type_exclusivity'].isna()) & (van_details_prev['plg_vehicle_mapping'].isna()) & (van_details_prev['vehicle_capacity_kgs']==30)& (van_details_prev['rental_type']=='Rented - Short term')]['van_id']))
		#vans=list(van_details['van_id'].unique())
		rented_bikes_to_fill_new=[]
		for v in rented_bikes_to_fill:
			for v2 in vans:
				if v in v2:
					rented_bikes_to_fill_new.append(v2)
		
		#This is under the assumption that, for instance any 3 tonne cannot be replaced by 2 or more vehicles.
		#owned_van_order_tofill=list(van_details[(van_details['vehicle_tonnage_kgs']!=30) & (van_details['rental_type'].isin(['Leased - Long term','Own']))].sort_values(by=['max_capacity','count_of_multi_trips', 'operating_hours_minutes','cost_per_day_rupees'],ascending=[False, False,False,True])['van_id'])
		#form all type of beats using remaining outlets and then decide which vehicle suits the best
		#here i think the cot function shud change-I think i shud consider the cost of the vehicle as well
		#can i use the other idea here-make beats of all types starting from all the outlets and pass it to solver to reduce the cost and all the outlets has to be visited only once
		rs_master['max_capacity']=rs_master['weight_factor']*rs_master['rental_vehicle_capacity_kgs']
		rental_van_order_tofill=list(rs_master.sort_values(by=['max_capacity','rental_costs_per_day_rupees'],ascending=[False,False])['rental_vehicle_capacity_kgs'].astype(str).unique())
		rs_master['van_id']=rs_master['rental_vehicle_capacity_kgs'].astype(str)
		
		
		
		owned_van_order_tofill=owned_van_order_tofill_new.copy()
		owned_bikes_to_fill=owned_bikes_to_fill_new.copy()
		rented_bikes_to_fill=rented_bikes_to_fill_new.copy()
		
		van_details['volume']=(van_details['vehicle_dimensions_length']*30.5)*(van_details['vehicle_dimensions_breadth']*30.5)*(van_details['vehicle_dimensions_height']*30.5).astype(int)
		van_details['max_volume']=van_details['volume']*van_details['volume_factor']
		rs_master['volume']=(rs_master['vehicle_dimensions_length']*30.5)*(rs_master['vehicle_dimensions_breadth']*30.5)*(rs_master['vehicle_dimensions_height']*30.5).astype(int)
		rs_master['max_volume']=rs_master['volume']*rs_master['volume_factor']
		rs_master['vehicle_speed']=15
		van_details['max_bills']=van_details['max_bills']*van_details['Bills_Factors']
		#rs_master['max_bills']=rs_master['max_bills']*rs_master['Bills_Factors']
		van_details['max_outlets']=van_details['max_outlets']*van_details['outlets_Factors']
		#rs_master['max_outlets']=rs_master['max_outlets']*rs_master['outlets_Factors']
		van_volume_dict=dict(list(zip(van_details['van_id'],van_details['max_volume'])))
		van_volume_dict.update(dict(list(zip(rs_master['van_id'],rs_master['max_volume']))))
		van_weight_dict=dict(list(zip(van_details['van_id'],van_details['max_capacity'])))
		van_weight_dict.update(dict(list(zip(rs_master['van_id'],rs_master['max_capacity']))))
		van_cost_dict=dict(list(zip(van_details['van_id'],van_details['cost_per_day_rupees'])))
		van_cost_dict.update(dict(list(zip(rs_master['van_id'],rs_master['rental_costs_per_day_rupees']))))
		van_bill_dict=dict(list(zip(van_details['van_id'],van_details['max_bills'])))
		van_bill_dict.update(dict(list(zip(rs_master['van_id'],rs_master['max_bills']))))
		van_endtime_dict=dict(list(zip(van_details['van_id'],van_details['operating_hours_minutes'])))
		van_endtime_dict.update(dict(list(zip(rs_master['van_id'],rs_master['operating_hours_minutes']))))
		van_multitrip_dict=dict(list(zip(van_details['van_id'],van_details['multi_trips'])))
		van_cutoff_dict=dict(list(zip(van_details['van_id'],van_details['cut_off'])))
		van_outlet_dict=dict(list(zip(van_details['van_id'],van_details['max_outlets'])))
		van_outlet_dict.update(dict(list(zip(rs_master['van_id'],rs_master['max_outlets']))))
		van_plg_mapping=dict(list(zip(van_details['van_id'],van_details['plg_vehicle_mapping'])))
		van_speed_dict=dict(list(zip(van_details['van_id'],van_details['vehicle_speed']/60)))
		van_speed_dict.update(dict(list(zip(rs_master['van_id'],rs_master['vehicle_speed']/60))))
		van_trip_dict=dict(list(zip(van_details['van_id'],van_details['no_trips'].astype(int))))
		van_fixedrate_dict=dict(list(zip(van_details['van_id'],van_details['fixed_rate'])))
		van_fixedrate_dict.update(dict(list(zip(rs_master['van_id'],rs_master['fixed_rate']))))
		van_details['base_rate_rupees']=van_details['base_rate_rupees'].fillna(0)
		rs_master['base_rate_rupees']=rs_master['base_rate_rupees'].fillna(0)
		van_details['per_km_rate_rupees']=van_details['per_km_rate_rupees'].fillna(0)
		rs_master['per_km_rate_rupees']=rs_master['per_km_rate_rupees'].fillna(0)
		van_details['per_hour_rate_rupees']=van_details['per_hour_rate_rupees'].fillna(0)
		rs_master['per_hour_rate_rupees']=rs_master['per_hour_rate_rupees'].fillna(0)
		van_baserate_dict=dict(list(zip(van_details['van_id'],van_details['base_rate_rupees'])))
		van_baserate_dict.update(dict(list(zip(rs_master['van_id'],rs_master['base_rate_rupees']))))
		van_perkmrate_dict=dict(list(zip(van_details['van_id'],van_details['per_km_rate_rupees'])))
		van_perkmrate_dict.update(dict(list(zip(rs_master['van_id'],rs_master['per_km_rate_rupees']))))
		van_perhourrate_dict=dict(list(zip(van_details['van_id'],van_details['per_hour_rate_rupees'])))
		van_perhourrate_dict.update(dict(list(zip(rs_master['van_id'],rs_master['per_hour_rate_rupees']))))
		
		new_dict=van_endtime_dict.copy()
		new_dict={k:0 for k in new_dict}
		
		van_endtime_dict_copy={}
		for k in van_endtime_dict.keys():
			van_endtime_dict_copy[k]=van_endtime_dict[k]
			
		van_details_prev['cut_off']=np.where(van_details_prev['multi_trips_cutoff']<0, 'no', 'yes')
		
		orddict={'Quantum':'QT','Shikhar':'SH','MB':'MB'}
		van_details['order_type_exclusivity']= van_details['order_type_exclusivity'].map(orddict)   
		
		ol_closure_time = {}
		ol_master = outlet_data.copy(deep = True)
		j = 0
		print(len(ol_master))
		print(ol_master['delivery_before_time'].fillna(0).unique())
		for i in ol_master['delivery_before_time'].fillna(datetime.time(9, 0)).map(lambda x: (x.hour - 9) * 60 + x.minute):
			print(i)
			if(i != 0):
				ol_closure_time[ol_master['partyhll_code'][j]] = [i, 480]        
				j+=1
			else:
				j+=1
		j = 0
		for i in ol_master['delivery_after_time'].fillna(datetime.time(9, 0)).map(lambda x: (x.hour - 9) * 60 + x.minute):
			if(i != 0):
				ol_closure_time[ol_master['partyhll_code'][j]] = [0,i]        
				j+=1
			else:
				j+=1
		j = 0
		for i in zip(ol_master['outlet_closure_from_time'].fillna(datetime.time(9, 0)).map(lambda x: (x.hour - 9) * 60 + x.minute), ol_master['outlet_closure_to_time'].fillna(datetime.time(9, 0)).map(lambda x: (x.hour - 9) * 60 + x.minute)):
			if(i[0] != 0):
				ol_closure_time[ol_master['partyhll_code'][j]] = [i[0], i[1]]
				j+=1
			else:
				j+=1
		
		flag=False 
		flag2=True  
		checked=False 
		shikar=False
		checked2=False
		outlets_allowed_forvan={}
		
		while(iteration<1):
		
			iteration=iteration+1
			
			multitripvan_rem_time=van_details_prev[(van_details_prev['multi_trips']=='yes') & (van_details_prev['cut_off']=='no')].groupby(['van_id'])['operating_hours_minutes'].sum()
			
			for k in van_endtime_dict_copy.keys():
				van_endtime_dict[k]=van_endtime_dict_copy[k]
				
			if(not(flag)):
				for k in skip_trips.keys():
					if((int(k.split('_')[-1])==3) and (van_cutoff_dict[k]=='yes')):
						skip_trips[k]=True
					elif(skip_trips[k]):
						skip_trips[k]=False
			
			#flag=False        
			for opc in grups_common_outs.keys():
				grups_common_outs[opc]=list(set(grups_common_outs[opc]))
			for g in common_outs_accross_grups.keys():
				common_outs_accross_grups[g]=list(set(common_outs_accross_grups[g]))
						
			grup_of_outlets={}
			grup_ol_weights={}
			grup_ol_service_time={}
			grup_ol_volume={}
			grup_ol_plg={}
			grup_ol_channel={}
			grup_rem_weight={}
			grup_rem_volume={}
			grup_rem_basepack={}
			grup_high_demand_outs={}
			grups={}    
			grup_outlet_bill_dict={}
			grup_outlet_basepack_dict={}
			grup_olplg_weights={}
			grup_olplg_service_time={}
			grup_olplg_volume={}
			grup_olplg_bill_dict={}
			grup_olplg_basepack_dict={}
			
			if(party_pack=='yes'):        
				input_data['NET_SALES_WEIGHT_IN_KGS']=input_data['weight']
				input_data['volume']=input_data['length']*input_data['width']*input_data['height']*input_data['multi_fact']
			else:
				sku_master['volume']=sku_master['unit_length']*sku_master['unit_breadth']*sku_master['unit_height']
				sku_master['volume'].fillna(0,inplace=True)
				vol_map=dict(zip(sku_master['SKU Code'],sku_master['volume']))
				input_data['volume']=input_data['BASEPACK CODE'].map(vol_map)
				input_data['volume']=input_data['volume']*input_data['NET_SALES_QTY']
				
			final_input_df=pd.DataFrame()    
			
			for op in grups_common_outs.keys():
				o=op.split('_')[0]
				p=op.split('_')[1]
				data=input_data[(input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==o)].copy(deep=True)
				data['BASEPACK CODE']=data['BASEPACK CODE'].astype(str)
				grup_olplg_bill_dict[op]=set(list(data['BILL_NUMBER']))
				grup_olplg_basepack_dict[op]=set(list(data['BASEPACK CODE']))
				grup_olplg_weights[op]=data['NET_SALES_WEIGHT_IN_KGS'].sum()
				grup_olplg_volume[op]=data['volume'].sum()            
				#ADD SERVICE TIME.....................................
				service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
				grup_olplg_service_time[op]=int(service_time_details[(grup_olplg_weights[op]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_olplg_weights[op]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
			
				
			
			for i,r in plg_clubbing_[plg_clubbing_['type'].isin(['default','exception'])].iterrows():
				if(len(r['groups'])<=0):
					k='|'.join(r['area'])+'_'+'|'.join(r['channel'])
					sub_df=input_data[(input_data['primarychannel'].isin(r['channel'])) & (input_data['area_name'].isin(r['area'])) ].copy(deep=True)
					if (k in common_outs_accross_grups.keys()):
						for item in common_outs_accross_grups[k]:
							o=item.split('_')[0]
							p=item.split('_')[1]
							c=item.split('_')[2]
							if(len(sub_df[(sub_df['primarychannel']==c) & (sub_df['area_name'].isin(r['area'])) & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==o)])<1):
								sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c) & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==o)]])
					if(len(sub_df)>0):
						
						grups[k]=r['groups']
						sub_df['comb']=sub_df['PARTY_HLL_CODE']
						grup_of_outlets[k]=list(sub_df['comb'].unique())
						sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str)
						grup_outlet_bill_dict[k]=sub_df.groupby(['comb'])['BILL_NUMBER'].apply(set)
						grup_outlet_basepack_dict[k]=sub_df.groupby(['comb'])['BASEPACK CODE'].apply(set)
						final_input_df=pd.concat([final_input_df,sub_df])
						grup_ol_plg[k]=sub_df.groupby(['comb'])['SERVICING PLG'].apply(list)
						grup_ol_channel[k]=sub_df.groupby(['comb'])['primarychannel'].apply(list)
						
						grup_ol_weights[k]=sub_df.groupby('comb')['NET_SALES_WEIGHT_IN_KGS'].sum()
						grup_ol_weights[k]['Distributor'] = 0
							
						grup_ol_volume[k]=sub_df.groupby(['comb'])['volume'].sum()
						grup_ol_volume[k]['Distributor'] = 0
						
						ol_service_time={}
						service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
						for o in sub_df['comb'].unique():
							ol_service_time[o]=int(service_time_details[(grup_ol_weights[k][o]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_ol_weights[k][o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
						ol_service_time['Distributor']=0     
						grup_ol_service_time[k]=ol_service_time
						
						grup_high_demand_outs[k]=list(grup_ol_weights[k][grup_ol_weights[k]>1000].index)
						grup_rem_weight[k]={}
						grup_rem_volume[k]={}
						#grup_rem_bills[k]={}
						grup_rem_basepack[k]={}
						for h in grup_high_demand_outs[k]:
							grup_rem_weight[k][h]=grup_ol_weights[k][h]
							grup_rem_volume[k][h]=grup_ol_volume[k][h]
							#grup_rem_bills[k][h]=grup_outlet_bill_dict[k][h]
							grup_rem_basepack[k][h]=grup_outlet_basepack_dict[k][h]
				else:
					k='|'.join(r['area'])+'_'+'|'.join(r['channel'])
					for plg in r['groups']:
						#print(plg)
						sub_df=input_data[(input_data['primarychannel'].isin(r['channel'])) & (input_data['area_name'].isin(r['area'])) & (input_data['SERVICING PLG'].isin(plg)) ].copy(deep=True)
						ky=k+'_'+'|'.join(plg)
						if (ky in common_outs_accross_grups.keys()):
							for item in common_outs_accross_grups[ky]:
								o=item.split('_')[0]
								p=item.split('_')[1]
								c=item.split('_')[2]
								if(len(sub_df[(sub_df['primarychannel']==c) & (sub_df['area_name'].isin(r['area'])) & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==o)])<1):
									sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c)  & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==o)]])
						if(len(sub_df)>0):
							grups[ky]=r['groups']
							sub_df['comb']=sub_df['PARTY_HLL_CODE']
							grup_of_outlets[k+'_'+'|'.join(plg)]=list(sub_df['comb'].unique())
							final_input_df=pd.concat([final_input_df,sub_df])
							sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str)
							grup_outlet_bill_dict[ky]=sub_df.groupby(['comb'])['BILL_NUMBER'].apply(set)
							grup_outlet_basepack_dict[ky]=sub_df.groupby(['comb'])['BASEPACK CODE'].apply(set)
							grup_ol_plg[ky]=sub_df.groupby(['comb'])['SERVICING PLG'].apply(list)
							grup_ol_channel[ky]=sub_df.groupby(['comb'])['primarychannel'].apply(list)
							
							grup_ol_weights[ky]=sub_df.groupby('comb')['NET_SALES_WEIGHT_IN_KGS'].sum()
							grup_ol_weights[ky]['Distributor'] = 0
		
							grup_ol_volume[ky]=sub_df.groupby(['comb'])['volume'].sum()
							grup_ol_volume[ky]['Distributor'] = 0
							
							ol_service_time={}
							service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
							for o in sub_df['comb'].unique():
								ol_service_time[o]=int(service_time_details[(grup_ol_weights[ky][o]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_ol_weights[ky][o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
							ol_service_time['Distributor']=0    
							grup_ol_service_time[ky]=ol_service_time          
							
							grup_high_demand_outs[ky]=list(grup_ol_weights[ky][grup_ol_weights[ky]>1000].index)
							grup_rem_weight[ky]={}
							grup_rem_volume[ky]={}
							#grup_rem_bills[ky]={}
							grup_rem_basepack[ky]={}
							for h in grup_high_demand_outs[ky]:
								grup_rem_weight[ky][h]=grup_ol_weights[ky][h]
								grup_rem_volume[ky][h]=grup_ol_volume[ky][h]
								#grup_rem_bills[ky][h]=grup_outlet_bill_dict[ky][h]
								grup_rem_basepack[ky][h]=grup_outlet_basepack_dict[ky][h]
					
			grup_rem_weight_initial = copy.deepcopy(grup_rem_weight)
					
			#for each van wat are the outlets that are allowed according to plg_vehicle mapping and lane restrictions combined
			if (len(common_outs_accross_grups)<1 and len(outlets_allowed_forvan)<1):
				outlet_data['lane_restriction'].fillna(0,inplace=True)
				di2={}
				for r in outlet_data['lane_restriction'].unique():
					if r!=0:
						di2[r]=int(r[1])*1000
				di2[0]=10*1000
				outlet_data['lane_restriction']=outlet_data['lane_restriction'].map(di2)
				di3={}
				di3=dict(list(zip(outlet_data['partyhll_code'],outlet_data['lane_restriction'])))
				input_data['lane_restrictions']=input_data['PARTY_HLL_CODE'].map(di3)
				
				
				van_tonnage_dict=dict(list(zip(van_details['van_id'],van_details['vehicle_capacity_kgs'])))
				van_tonnage_dict.update(dict(list(zip(rs_master['van_id'],rs_master['rental_vehicle_capacity_kgs']))))
		#        van_areassign_dict=dict(list(zip(van_details['van_id'],van_details['area_assignment'].str.lower())))
				van_areassign_dict=dict(list(zip(van_details['van_id'],van_details['area_assignment'])))
				van_orderexclu_dict=dict(list(zip(van_details['van_id'],van_details['order_type_exclusivity'])))
				
				for van_id in owned_van_order_tofill:
					outlets_allowed_forvan[van_id]=[]
					constraints={}
					if(str(van_plg_mapping[van_id])!='nan'):
						constraints['SERVICING PLG']=van_plg_mapping[van_id]
					if(str(van_areassign_dict[van_id])!='nan'):
						constraints['area_name']=van_areassign_dict[van_id]
					if(str(van_orderexclu_dict[van_id])!='nan'):
						constraints['order_type']=van_orderexclu_dict[van_id]
					if(len(constraints)>0):    
						for k in constraints.keys():
							outlets_selected=list(input_data[(input_data[k]==constraints[k]) & (van_tonnage_dict[van_id]<input_data['lane_restrictions'])]['PARTY_HLL_CODE'].unique())
							if(len(outlets_allowed_forvan[van_id])>0):
								outlets_allowed_forvan[van_id]=list(set(outlets_allowed_forvan[van_id]).intersection(set(outlets_selected)))
							else:
								outlets_allowed_forvan[van_id]=outlets_selected
						if(len(outlets_allowed_forvan[van_id])<=1):
							outlets_allowed_forvan[van_id]=list(input_data[(van_tonnage_dict[van_id]<input_data['lane_restrictions'])]['PARTY_HLL_CODE'].unique()) 
					else:
					   outlets_allowed_forvan[van_id]=list(input_data[(van_tonnage_dict[van_id]<input_data['lane_restrictions'])]['PARTY_HLL_CODE'].unique()) 
				
				for van_id in rental_van_order_tofill:
					outlets_allowed_forvan[van_id]=list(input_data[(van_tonnage_dict[van_id]<input_data['lane_restrictions'])]['PARTY_HLL_CODE'].unique())
			
			#service_window_flag = True
			outlets_allowed_for_bike = {}
			bikes = owned_bikes_to_fill + rented_bikes_to_fill
			bike_ols=[]
			grup_outlets_allowed_for_bike={}
			for van_id in bikes: 
				print(van_id)
				grup_outlets_allowed_for_bike[van_id]={}
				max_distance_for_bike = int(list(van_details[van_details['van_id']==van_id]['bike_max_distance'])[0])
				max_distance_for_bike=12
				outlets_allowed_for_bike[van_id]=[]
				constraints={}
				if(str(van_plg_mapping[van_id])!='nan'):
					constraints['SERVICING PLG']=van_plg_mapping[van_id]
				if(str(van_areassign_dict[van_id])!='nan'):
					constraints['area_name']=van_areassign_dict[van_id]
				if(str(van_orderexclu_dict[van_id])!='nan'):
					constraints['order_type']=van_orderexclu_dict[van_id]
				if(len(constraints)>0):    
					for k in constraints.keys():
						outlets_selected=list(input_data[(input_data[k]==constraints[k]) & (input_data['PARTY_HLL_CODE'].isin(distance_matrix['Distributor'][distance_matrix['Distributor'] <= max_distance_for_bike].index))]['PARTY_HLL_CODE'].unique())
						if(len(outlets_allowed_for_bike[van_id])>0):
							outlets_allowed_for_bike[van_id]=list(set(outlets_allowed_forvan[van_id]).intersection(set(outlets_selected)))
						else:
							outlets_allowed_for_bike[van_id]=outlets_selected
					if(len(outlets_allowed_for_bike[van_id])<=1):
						outlets_allowed_for_bike[van_id]=list(input_data[(input_data['PARTY_HLL_CODE'].isin(distance_matrix['Distributor'][distance_matrix['Distributor'] <= max_distance_for_bike].index))]['PARTY_HLL_CODE'].unique()) 
				else:
				   outlets_allowed_for_bike[van_id]=list(input_data[(input_data['PARTY_HLL_CODE'].isin(distance_matrix['Distributor'][distance_matrix['Distributor'] <= max_distance_for_bike].index))]['PARTY_HLL_CODE'].unique()) 
		
				for k in grup_of_outlets.keys():
					bike_max_weight_per_bill=int(list(van_details[van_details['van_id']==van_id]['bike_max_weight_per_bill'])[0])
					grup_outlets_allowed_for_bike[van_id][k] = list(grup_ol_weights[k][grup_ol_weights[k] <= bike_max_weight_per_bill].index)
					for ol in list(grup_ol_weights[k][((grup_ol_weights[k] > bike_max_weight_per_bill) & (grup_ol_weights[k] < 30))].index):
						bill_weights = input_data[input_data['PARTY_HLL_CODE'] == ol].groupby(['BILL_NUMBER'])['NET_SALES_WEIGHT_IN_KGS'].sum()
						if(len(bill_weights > bike_max_weight_per_bill) == 0):
							grup_outlets_allowed_for_bike[van_id][k].append(ol)
				
			
			bike_beats = []
			
			outlets_allowed_for_bike_copy=outlets_allowed_for_bike.copy()
		
			if(flag2):
				def underutilized_bike(beat,van_id):
				   if(beat['cum_weight']<0.68*van_weight_dict[van_id]):
					   return True
				   return False
			   
				bike_beat_list = []
				bike_beats = []
				ids=[]
				for k in grup_of_outlets.keys():
					ids.extend(grup_of_outlets[k])
			
				ids_copy=list(set(ids)).copy()
				
				for van_id in bikes:
					bike_beat_list=[]
					shikar=False
					# len(set(list(grup_of_outlets.values())[0] + list(grup_of_outlets.values())[1] + list(grup_of_outlets.values())[2]))
					#for start_id in outlets_allowed_for_bike[van_id]:
					if(str(van_orderexclu_dict[van_id])!='nan'):
						sub_df=input_data[(input_data['PARTY_HLL_CODE'].isin(ids_copy)) &(input_data['PARTY_HLL_CODE'].isin(outlets_allowed_for_bike[van_id])) & (input_data['order_type']==van_orderexclu_dict[van_id])].copy(deep=True)
						sub_df['comb']=sub_df['PARTY_HLL_CODE']
						sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str)
						outlet_bill_dict=sub_df.groupby(['comb'])['BILL_NUMBER'].apply(set)
						outlet_basepack_dict=sub_df.groupby(['comb'])['BASEPACK CODE'].apply(set)
						ol_plg=sub_df.groupby(['comb'])['SERVICING PLG'].apply(list)
						ol_channel=sub_df.groupby(['comb'])['primarychannel'].apply(list)
						ol_weights=sub_df.groupby('comb')['NET_SALES_WEIGHT_IN_KGS'].sum()
						ol_weights['Distributor'] = 0    
						ol_volume=sub_df.groupby(['comb'])['volume'].sum()
						ol_volume['Distributor'] = 0
						ol_service_time={}
						service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
						for o in sub_df['comb'].unique():
							ol_service_time[o]=int(service_time_details[(ol_weights[o]>=service_time_details['load_range_from_kgs'].astype(float)) & (ol_weights[o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
						ol_service_time['Distributor']=0     
						bike_max_weight_per_bill=int(van_details[van_details['van_id']==van_id]['bike_max_weight_per_bill'])
						shikar_outlets_allowed_for_bike = list(ol_weights[ol_weights <= bike_max_weight_per_bill].index)
						for ol in list(ol_weights[(ol_weights > bike_max_weight_per_bill) & (ol_weights < 30)].index):
							bill_weights = input_data[input_data['PARTY_HLL_CODE'] == ol].groupby(['BILL_NUMBER'])['NET_SALES_WEIGHT_IN_KGS'].sum()
							if(len(bill_weights > bike_max_weight_per_bill) == 0):
								shikar_outlets_allowed_for_bike.append(ol)
						for id in set(shikar_outlets_allowed_for_bike)-{'Distributor'}:
							sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel=form_shikar_bike_beats(van_orderexclu_dict[van_id],id,van_id,list(set(shikar_outlets_allowed_for_bike)-{'Distributor'}),outlet_bill_dict,outlet_basepack_dict,ol_plg,ol_channel,ol_weights,ol_volume,ol_service_time)
							bike_beat_list.append([sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,van_orderexclu_dict[van_id],cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel])
						shikar=True
					else:
						shikar_outlets_allowed_for_bike=[]
						for key in grup_of_outlets.keys():
							for start_id in list(set(grup_of_outlets[key]).intersection(set(outlets_allowed_for_bike[van_id])).intersection(set(grup_outlets_allowed_for_bike[van_id][key]))):
								sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel = form_bike_beats(key, start_id, van_id)
								bike_beat_list.append([sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,key,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel])
							else:
								break
					if ((len(bike_beat_list)>0) and (max([b[3] for b in bike_beat_list])>0)):
							bike_beat = find_best_bike_beat(bike_beat_list, van_id,list(set(shikar_outlets_allowed_for_bike)-{'Distributor'}))
							beat=bike_beat
						#if(not(underutilized_bike(bike_beat,van_id))):
							bike_beats.append(bike_beat)
							if((van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') ):
								multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]= multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]-(beat['end_time']+45)
							
							ky=beat['del_type']
							if shikar:
								shikar=False
								outlets_allowed_for_bike[van_id]=list(set(outlets_allowed_for_bike[van_id])-set(beat['sequence']))
								for i in range(1,len(beat['sequence'])):
								  for g in grup_of_outlets.keys():
									  o=beat['sequence'][i]
									  if o in grup_of_outlets[g]:
												grup_of_outlets[g].remove(o)
												del grup_ol_weights[g][o]
												del grup_ol_volume[g][o]
												del grup_ol_service_time[g][o]
												del grup_outlet_bill_dict[g][o]
												del grup_outlet_basepack_dict[g][o]
												del grup_ol_plg[g][o]
												del grup_ol_channel[g][o]
							else:
								grup_of_outlets[ky]=list(set(grup_of_outlets[ky])-set(beat['sequence']))
								
								if ky in common_outs_accross_grups.keys():
									if(len(set(beat['sequence']).intersection(set([op.split('_')[0] for op in grups_common_outs.keys()])))>=1):
										for op in grups_common_outs.keys():
											if(op.split('_')[0] in beat['sequence']):
												o=op.split('_')[0]
												ps=[]
												cs=[]
												for i in range(0,len(beat['sequence'])):
													if(beat['sequence'][i]==o):
														ps=beat['plg'][i]
														cs=beat['channel'][i]
														break
												if ((op.split('_')[1] in ps) and (op.split('_')[2] in cs)):
													for g in grups_common_outs[op]:
														if((g!=ky)):
															if(not(o in grup_rem_weight[g].keys())):
																grup_ol_weights[g][o]=grup_ol_weights[g][o]-grup_olplg_weights[op]
																if(grup_ol_weights[g][o]<=0.1):
																	grup_of_outlets[g].remove(o)
																	del grup_ol_weights[g][o]
																	del grup_ol_volume[g][o]
																	del grup_ol_service_time[g][o]
																	del grup_outlet_bill_dict[g][o]
																	del grup_outlet_basepack_dict[g][o]
																	del grup_ol_plg[g][o]
																	del grup_ol_channel[g][o]
																else:
																	grup_ol_volume[g][o]=grup_ol_volume[g][o]-grup_olplg_volume[op]
																	service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
																	grup_ol_service_time[g][o]=int(service_time_details[(grup_ol_weights[g][o]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_ol_weights[g][o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
																	grup_outlet_bill_dict[g][o]=set(grup_outlet_bill_dict[g][o])-set(grup_olplg_bill_dict[op])   
																	grup_outlet_basepack_dict[g][o]=set(grup_outlet_basepack_dict[g][o])-set(grup_olplg_basepack_dict[op])
																	grup_ol_plg[g][o]=[p for p in grup_ol_plg[g][o] if p!=op.split('_')[1]]
																	grup_ol_channel[g][o]=[p for p in grup_ol_channel[g][o] if p!=op.split('_')[2]]
							ids=[]
							for k in grup_of_outlets.keys():
								ids.extend(grup_of_outlets[k])
							ids_copy=list(set(ids)).copy()
																	
		
			ids=[]
			for k in grup_of_outlets.keys():
				ids.extend(grup_of_outlets[k])
		
			ids_copy=list(set(ids)).copy()
					
			
			'''
			TODO
			'''
			flag2=False
			beat_list=[]
			beat_list2=[]
			cnt=0
			
			input_data['outlet_bp']=input_data['PARTY_HLL_CODE']+'_'+input_data['BASEPACK CODE'] 
			high_bp_demand_outs=input_data.groupby(['outlet_bp'])['NET_SALES_WEIGHT_IN_KGS'].sum()[input_data.groupby(['outlet_bp'])['NET_SALES_WEIGHT_IN_KGS'].sum()>1000].to_dict()
			vans=list(van_details['vehicle_capacity_kgs'].astype(float).unique())+list(rs_master['rental_vehicle_capacity_kgs'].astype(float).unique())
			vans.sort()
			min_cap_van_high_bp_demand_outs={}
			for ho in high_bp_demand_outs.keys():
				for vw in vans:
					if(high_bp_demand_outs[ho]>vw):
						continue
					else:
					   min_cap_van_high_bp_demand_outs[ho]=vw
					   break
			
		
			outs_allowed_forvan_copy=outlets_allowed_forvan.copy()    
			for van_id in owned_van_order_tofill:
				print('---------------------------Owned--------------------')  
				shikar=False    
				for k in grup_rem_weight.keys():
							if('AP50001805656' in grup_rem_weight[k]):
								print(k,grup_rem_weight[k]['AP50001805656'])
								
				idkey_vis=[]
				if((van_id in skip_trips.keys()) and (skip_trips[van_id])):
					print('skip',van_id)
					continue
				if((van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') and (multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]<=0)):
					continue
				print(van_id)
				beat_list=[]
				high_demand_outs=[]
				if(str(van_orderexclu_dict[van_id])!='nan'):
					print(shikar,'entered shikar')
					print(van_id,van_orderexclu_dict[van_id])
					sub_df=input_data[(input_data['PARTY_HLL_CODE'].isin(ids_copy)) & (input_data['PARTY_HLL_CODE'].isin(outlets_allowed_forvan[van_id])) & (input_data['order_type'].str.lower() == van_orderexclu_dict[van_id].lower())].copy(deep=True)
					sub_df['comb']=sub_df['PARTY_HLL_CODE']
					sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str)
					outlet_bill_dict=sub_df.groupby(['comb'])['BILL_NUMBER'].apply(set)
					outlet_basepack_dict=sub_df.groupby(['comb'])['BASEPACK CODE'].apply(set)
					ol_plg=sub_df.groupby(['comb'])['SERVICING PLG'].apply(list)
					ol_channel=sub_df.groupby(['comb'])['primarychannel'].apply(list)
					ol_weights=sub_df.groupby('comb')['NET_SALES_WEIGHT_IN_KGS'].sum()
					ol_weights['Distributor'] = 0    
					ol_volume=sub_df.groupby(['comb'])['volume'].sum()
					ol_volume['Distributor'] = 0
					ol_service_time={}
					service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
					for o in sub_df['comb'].unique():
						ol_service_time[o]=int(service_time_details[(ol_weights[o]>=service_time_details['load_range_from_kgs'].astype(float)) & (ol_weights[o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
					ol_service_time['Distributor']=0     
					high_demand_outs=list(ol_weights[ol_weights>1000].index)
					rem_weight={}
					rem_volume={}
					rem_basepack={}
					for h in high_demand_outs:
						rem_weight[h]=ol_weights[h]
						rem_volume[h]=ol_volume[h]
						rem_basepack[h]=outlet_basepack_dict[h]
					for id in sub_df['PARTY_HLL_CODE'].unique():
						sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel=form_shikar_beats(van_orderexclu_dict[van_id],id,van_id,list(sub_df['PARTY_HLL_CODE'].unique()),outlet_bill_dict,outlet_basepack_dict,ol_plg,ol_channel,ol_weights,ol_volume,ol_service_time,high_demand_outs,rem_weight,rem_volume,rem_basepack)
						beat_list.append([sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,van_orderexclu_dict[van_id],cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel])
					shikar=True
				
				elif(len(outlets_allowed_forvan[van_id])>0):
					for id in ids_copy:
						for key in grup_of_outlets.keys():
							if(id in outlets_allowed_forvan[van_id]):
								if(id in grup_of_outlets[key]):
									#print(id,key)
									idkey_vis.append((id,key))
									print(grup_of_outlets.keys())
									sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel=form_normal_beats(key,id,van_id)
									beat_list.append([sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,key,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel])
									#if(('HUL-444305D-P42835' in sequence ) and (key=='rest of retail_PP| PP-A| PP-B')):
									#    print('gadbad')
							else:
								break
							
				if((len(beat_list)>0) and (max([b[3] for b in beat_list])>0)):
					
					beat_list_clone2=[]
					beat_list_clone_flag2=0
					for obp in min_cap_van_high_bp_demand_outs.keys():
						for k in grup_rem_weight.keys():
							o=obp.split('_')[0]
							bp=obp.split('_')[1]
							if((o in grup_rem_weight[k].keys()) and (bp in grup_rem_basepack[k][o]) and (grup_rem_weight[k][o]>1000) and (van_tonnage_dict[van_id]==min_cap_van_high_bp_demand_outs[obp])):
								for b in beat_list:
									if((o in b[0]) and (bp in [item for sublist in b[10] for item in sublist])):
										beat_list_clone2.append(b)
					
					if(len(beat_list_clone2)>0):
						print('beat_list_clone')
						beat_list_clone_flag2=1
						beat_list=beat_list_clone2.copy()
						
					beat_list_clone_flag=0
					if(beat_list_clone_flag2==0):
						rem_high_demand_outs=[]
						if(van_multitrip_dict[van_id]=='yes'):
							for k in grup_rem_weight.keys():
								for ot in grup_rem_weight[k].keys():
									if(grup_rem_weight[k][ot]>0.10) and (((grup_rem_weight[k][ot]+0.10)>=grup_rem_weight_initial[k][ot]) or (grup_rem_weight[k][ot]>=1000)):
										rem_high_demand_outs.append(ot)
										
						beat_list_clone=[]
						beat_list_clone_flag=0
						rem_high_demand_outs=list(set(rem_high_demand_outs))
						for ot in rem_high_demand_outs:
							for b in beat_list:
								if(ot in b[0]):
									beat_list_clone.append(b)
						
						if(len(beat_list_clone)>0):
							print('beat_list_clone')
							beat_list_clone_flag=1
							beat_list=beat_list_clone.copy()
					
					
								
					beat=find_best_beat(beat_list,van_id)
					
					
					
					'''
					if (van_multitrip_dict[van_id]=='yes'):
						if((int(van_id.split('_')[-1])==1) and (van_cutoff_dict[van_id]=='yes')):
							if(beat['cum_weight']>=0.68*van_weight_dict[van_id]):
								skip_trips['_'.join(van_id.split('_')[:-1])+'_'+str(3)]=True
							else:
								skip_trips['_'.join(van_id.split('_')[:-1])+'_'+str(2)]=True
								continued
						elif((int(van_id.split('_')[-1])==1) and (van_cutoff_dict[van_id]=='no')):
							if(beat['cum_weight']<0.68*van_weight_dict[van_id]):
								for t in range(int(van_id.split('_')[-1]),van_trip_dict[van_id]+1):
									skip_trips['_'.join(van_id.split('_')[:-1])+'_'+str(t)]=True
								
						if((int(van_id.split('_')[-1])==2) and (van_cutoff_dict[van_id]=='yes')):
							if(beat['cum_weight']>=0.68*van_weight_dict[van_id]):
								skip_trips['_'.join(van_id.split('_')[:-1])+'_'+str(3)]=True
							else:
								#if(len(beat['sequence'])>5):
								continue
						elif((int(van_id.split('_')[-1])>1) and (van_cutoff_dict[van_id]=='no')):
							if(beat['cum_weight']<0.68*van_weight_dict[van_id]):
								for t in range(int(van_id.split('_')[-1]),van_trip_dict[van_id]+1):
									skip_trips['_'.join(van_id.split('_')[:-1])+'_'+str(t)]=True
								#if(len(beat['sequence'])>5):
								continue
					'''                
					if((van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='no') ):
						multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]= multitripvan_rem_time['_'.join(van_id.split('_')[:-1])]-(beat['end_time']+45)
					
					
					if((van_multitrip_dict[van_id]=='yes') and (van_cutoff_dict[van_id]=='yes')):
						
						if (int(van_id.split('_')[1])==1) and (new_dict[van_id]<1):
							print(True)
							new_dict[van_id]=new_dict[van_id]+1
							trip_1_aval_time=van_endtime_dict[van_id]-beat['end_time']
							new_vehicle_id=van_id.split('_')[0]+'_'+str(2)
							print(beat['end_time'],van_endtime_dict[van_id]-beat['end_time'],van_endtime_dict[new_vehicle_id],van_endtime_dict[new_vehicle_id]+trip_1_aval_time)
							avail_time=van_endtime_dict[new_vehicle_id]
							van_endtime_dict[new_vehicle_id]=avail_time+trip_1_aval_time
					
					
					beat_list2.append(beat)
					if(beat_list_clone_flag==1):
						print(set(beat['sequence']).intersection(set(rem_high_demand_outs)))
		#            if('AP50001805656' in beat['sequence']):
		#                for o in range(len(beat['sequence'])):
		#                    if(beat['sequence'][o]=='AP50001805656'):
		#                       print(beat['del_type'],beat['sequence'][o],beat['wgt_sequence'][o]) 
					#add_to_output(beat)
					ky=beat['del_type']
					if shikar:
						print('entered shikar')
						shikar=False
						outlets_allowed_forvan[van_id]=list(set(outlets_allowed_forvan[van_id])-set(beat['sequence']))
						
						for i in range(1,len(beat['sequence'])):
    						  for g in grup_of_outlets.keys():
    							  o=beat['sequence'][i]
    							  if o in grup_of_outlets[g]:
    								  if o in  high_demand_outs:
        									bp_included_beat=beat['base_pack_cov'][i]
        									wgt_included_beat=beat['wgt_sequuence'][i]
        									vol_included_beat=beat['vol_sequence'][i]
        									plg_included_beat=beat['plg'][i]
        									channel_included_beat=beat['channel'][i]
        									sub_df=input_data.copy(deep=True)
        									sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str) 
        									sub_df=sub_df[(sub_df['PARTY_HLL_CODE']==o) & (sub_df['BASEPACK CODE'].isin(list(grup_rem_basepack[g][o])))].copy(deep=True)
        									sub_df=sub_df[sub_df['BASEPACK CODE'].isin(bp_included_beat)].copy(deep=True)
        									if o in grup_rem_weight[g].keys():
        										grup_rem_weight[g][o]=grup_rem_weight[g][o]-sub_df['NET_SALES_WEIGHT_IN_KGS'].sum()
        										if(grup_rem_weight[g][o]<=0.1):
        											grup_of_outlets[g].remove(o)
        											del grup_ol_weights[g][o]
        											del grup_ol_volume[g][o]
        											del grup_ol_service_time[g][o]
        											del grup_outlet_bill_dict[g][o]
        											del grup_outlet_basepack_dict[g][o]
        											del grup_ol_plg[g][o]
        											del grup_ol_channel[g][o]
        											del grup_rem_weight[g][o]
        											del grup_rem_basepack[g][o]
        											del grup_rem_volume[g][o]
        										else:
        										   grup_rem_volume[g][o]= grup_rem_volume[g][o]-sub_df['volume'].sum()
        										   grup_rem_basepack[g][o]=set(grup_rem_basepack[g][o])-set(list(sub_df['BASEPACK CODE']))
        									else:
        										grup_ol_weights[g][o]=grup_ol_weights[g][o]-sub_df['NET_SALES_WEIGHT_IN_KGS'].sum()
        										if(grup_ol_weights[g][o]<=0.1):
        											grup_of_outlets[g].remove(o)
        											del grup_ol_weights[g][o]
        											del grup_ol_volume[g][o]
        											del grup_ol_service_time[g][o]
        											del grup_outlet_bill_dict[g][o]
        											del grup_outlet_basepack_dict[g][o]
        											del grup_ol_plg[g][o]
        											del grup_ol_channel[g][o]
        										else:
        											grup_ol_volume[g][o]=grup_ol_volume[g][o]-sub_df['volume'].sum()
        											service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
        											grup_ol_service_time[g][o]=int(service_time_details[(grup_ol_weights[g][o]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_ol_weights[g][o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
        											grup_outlet_bill_dict[g][o]=set(grup_outlet_bill_dict[g][o])-set(list(sub_df['BILL_NUMBER']))   
        											grup_outlet_basepack_dict[g][o]=set(grup_outlet_basepack_dict[g][o])-set(list(sub_df['BASEPACK CODE']))
        											grup_ol_plg[g][o]=list(set(grup_ol_plg[g][o])-set(plg_included_beat))
        											grup_ol_channel[g][o]=list(set(grup_ol_plg[g][o])-set(channel_included_beat))
    								  else:
        									if o in grup_rem_weight[g].keys():
        										grup_of_outlets[g].remove(o)
        										del grup_ol_weights[g][o]
        										del grup_ol_volume[g][o]
        										del grup_ol_service_time[g][o]
        										del grup_outlet_bill_dict[g][o]
        										del grup_outlet_basepack_dict[g][o]
        										del grup_ol_plg[g][o]
        										del grup_ol_channel[g][o]
        										del grup_rem_weight[g][o]
        										del grup_rem_basepack[g][o]
        										del grup_rem_volume[g][o]
        									else:
        										grup_of_outlets[g].remove(o)
        										del grup_ol_weights[g][o]
        										del grup_ol_volume[g][o]
        										del grup_ol_service_time[g][o]
        										del grup_outlet_bill_dict[g][o]
        										del grup_outlet_basepack_dict[g][o]
        										del grup_ol_plg[g][o]
        										del grup_ol_channel[g][o]
										
					else:
						grup_of_outlets[ky]=list(set(grup_of_outlets[ky])-set(beat['sequence']))
						if ky in common_outs_accross_grups.keys():
							if(len(set(beat['sequence']).intersection(set([op.split('_')[0] for op in grups_common_outs.keys()])))>=1):
								for op in grups_common_outs.keys():
									if(op.split('_')[0] in beat['sequence']):
										o=op.split('_')[0]
										ps=[]
										cs=[]
										for i in range(0,len(beat['sequence'])):
											if(beat['sequence'][i]==o):
												ps=beat['plg'][i]
												cs=beat['channel'][i]
												break
										if ((op.split('_')[1] in ps) and (op.split('_')[2] in cs)):
											for g in grups_common_outs[op]:
												if((g!=ky)):
													if(not(o in grup_rem_weight[g].keys())):
														grup_ol_weights[g][o]=grup_ol_weights[g][o]-grup_olplg_weights[op]
														if(grup_ol_weights[g][o]<=0.1):
															grup_of_outlets[g].remove(o)
															del grup_ol_weights[g][o]
															del grup_ol_volume[g][o]
															del grup_ol_service_time[g][o]
															del grup_outlet_bill_dict[g][o]
															del grup_outlet_basepack_dict[g][o]
															del grup_ol_plg[g][o]
															del grup_ol_channel[g][o]
														else:
															grup_ol_volume[g][o]=grup_ol_volume[g][o]-grup_olplg_volume[op]
															service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
															grup_ol_service_time[g][o]=int(service_time_details[(grup_ol_weights[g][o]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_ol_weights[g][o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
															grup_outlet_bill_dict[g][o]=set(grup_outlet_bill_dict[g][o])-set(grup_olplg_bill_dict[op])   
															grup_outlet_basepack_dict[g][o]=set(grup_outlet_basepack_dict[g][o])-set(grup_olplg_basepack_dict[op])
															grup_ol_plg[g][o]=[p for p in grup_ol_plg[g][o] if p!=op.split('_')[1]]
															grup_ol_channel[g][o]=[p for p in grup_ol_channel[g][o] if p!=op.split('_')[2]]
													else:
														
														for i in range(len(beat['sequence'])):
															if (beat['sequence'][i]==o):
																bp_included_beat=beat['base_pack_cov'][i]
																break
														#print(set(bp_included_beat))
														'''
														l=g.split('_')    
														if(len(l)>2):
															sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['SERVICING PLG'].isin(l[2].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)        
														else:
															sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)
														
														if (g in common_outs_accross_grups.keys()):
															if(o in [op.split('_')[0] for op in common_outs_accross_grups[g]]):
																for opc in common_outs_accross_grups[g]:
																	if(o in opc.split('_')):
																		print(opc)
																	p=opc.split('_')[1]
																	c=opc.split('_')[2]
																if(len(sub_df[(sub_df['primarychannel']==c)  & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==o)])<1):
																	sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c)  & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==o)].copy(deep=True)])
														'''
														sub_df=input_data.copy(deep=True)
														sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str) 
														sub_df=sub_df[(sub_df['PARTY_HLL_CODE']==o) & (sub_df['BASEPACK CODE'].isin(list(grup_rem_basepack[g][o]))) & (sub_df['SERVICING PLG']==op.split('_')[1])].copy(deep=True)
														sub_df=sub_df[sub_df['BASEPACK CODE'].isin(bp_included_beat)].copy(deep=True)
			
														grup_rem_weight[g][o]=grup_rem_weight[g][o]-sub_df['NET_SALES_WEIGHT_IN_KGS'].sum()
														if(grup_rem_weight[g][o]<=0.1):
															grup_of_outlets[g].remove(o)
															del grup_ol_weights[g][o]
															del grup_ol_volume[g][o]
															del grup_ol_service_time[g][o]
															del grup_outlet_bill_dict[g][o]
															del grup_outlet_basepack_dict[g][o]
															del grup_ol_plg[g][o]
															del grup_ol_channel[g][o]
															del grup_rem_weight[g][o]
															del grup_rem_basepack[g][o]
															del grup_rem_volume[g][o]
														else:
														   grup_rem_volume[g][o]= grup_rem_volume[g][o]-sub_df['volume'].sum()
														   grup_rem_basepack[g][o]=set(grup_rem_basepack[g][o])-set(list(sub_df['BASEPACK CODE']))
							   
						for h in grup_rem_weight[ky].keys():
							if((h not in grup_of_outlets[ky]) and (h in beat['sequence'])):
								wgt_included_beat=sum([beat['wgt_sequence'][i] for i in range(len(beat['sequence'])) if beat['sequence'][i]==h])
								vol_included_beat=sum([beat['vol_sequence'][i] for i in range(len(beat['sequence'])) if beat['sequence'][i]==h])
								bp_included_beat=[]
								for i in range(len(beat['sequence'])):
									if (beat['sequence'][i]==h):
										bp_included_beat=beat['base_pack_cov'][i]
										ps=beat['plg'][i]
										cs=beat['channel'][i]
										break
								l=ky.split('_')  
								#print('original key')
								#print(set(bp_included_beat))
								if(len(l)>2):
									sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['SERVICING PLG'].isin(l[2].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)        
								else:
									sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)
								
								if (ky in common_outs_accross_grups.keys()):
									if(h in [op.split('_')[0] for op in common_outs_accross_grups[ky]]):
										for opc in common_outs_accross_grups[ky]:
											p=opc.split('_')[1]
											c=opc.split('_')[2]
										if(len(sub_df[(sub_df['primarychannel']==c)  & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==h)])<1):
											sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c)  & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==h)].copy(deep=True)])
							
								sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str) 
								sub_df=sub_df[(sub_df['PARTY_HLL_CODE']==h) & (sub_df['BASEPACK CODE'].isin(list(grup_rem_basepack[ky][h])))].copy(deep=True)
								sub_df_rev=sub_df[sub_df['BASEPACK CODE'].isin(bp_included_beat)].copy(deep=True)
								sub_df=sub_df[~sub_df['BASEPACK CODE'].isin(bp_included_beat)].copy(deep=True)
								#grup_rem_bills[ky][h]=set(sub_df['BILL_NUMBER'].unique())
								grup_rem_basepack[ky][h]=set(sub_df['BASEPACK CODE'].unique())
								#print('wgt_included_beat')
								#print(wgt_included_beat)
								r_w=grup_rem_weight[ky][h]-wgt_included_beat
								r_v=grup_rem_volume[ky][h]-vol_included_beat
								#print('weight_rem')
								#print(r_w)
								if((r_w>0) and (r_v>0)):
									grup_of_outlets[ky].append(h)
								grup_rem_weight[ky][h]=r_w
								grup_rem_volume[ky][h]=r_v
			
							if((h not in grup_of_outlets[ky]) and (grup_rem_weight[ky][h]>0)):
								grup_of_outlets[ky].append(h)
			
					ids=[]
					for k in grup_of_outlets.keys():
						ids.extend(grup_of_outlets[k])
					ids_copy=list(set(ids)).copy()
			
			print('MULTITRIPS CHECK---------------------------------------')               
			if(not(checked)):
				print('entered')
				multitripvans_w_cutoff=[]
				multitripvans_wo_cutoff=[]
				non_multitrip_van=[]
				for b in beat_list2:
					if((van_multitrip_dict[b['van_id']]=='yes') and (van_cutoff_dict[b['van_id']]=='yes') and (b['van_id'].split('_')[1]!='3')):
						multitripvans_w_cutoff.append(b)
					elif((van_multitrip_dict[b['van_id']]=='yes') and (van_cutoff_dict[b['van_id']]=='no')):
						multitripvans_wo_cutoff.append(b)
					elif((van_multitrip_dict[b['van_id']]=='no')):
						non_multitrip_van.append(b)
				
				for b in beat_list2[::-1]:
					if(b['cum_weight']<0.50*van_weight_dict[b['van_id']]) and (b['end_time']<van_endtime_dict[b['van_id']]):
						print([b['van_id'] for b in multitripvans_w_cutoff])
						print([b['van_id'] for b in multitripvans_wo_cutoff])
						for b1 in range(0,len(multitripvans_w_cutoff)):
							if(multitripvans_w_cutoff[b1]['van_id'].split('_')[1]=='2'):
								continue
							bt1=multitripvans_w_cutoff[b1]
							if((b1+1)<len(multitripvans_w_cutoff)):
								bt2=multitripvans_w_cutoff[b1+1]
							else:
								continue
							if(bt1['van_id'].split('_')[0]!=bt2['van_id'].split('_')[0]):
								continue
							if(bt1['del_type']==b['del_type']):
								if((bt1['cum_weight']<0.50*van_weight_dict[bt2['van_id']]) and (bt2['cum_weight']<0.50*van_weight_dict[bt1['van_id']])):
									skip_trips[bt2['van_id']]=True
									skip_trips[bt1['van_id']]=True
									skip_trips['_'.join(bt1['van_id'].split('_')[:-1])+'_3']=False
									flag=True
								elif(bt2['cum_weight']<0.50*van_weight_dict[bt2['van_id']]):
									skip_trips[bt2['van_id']]=True
									flag=True
							if(flag):
								break
						if(flag):
							break
						for b2 in multitripvans_wo_cutoff[::-1]:
							if((int(b2['van_id'].split('_')[-1])>1) and (b2['del_type']==b['del_type'])  and (b2['cum_weight']<0.50*van_weight_dict[b2['van_id']])):
								skip_trips[b2['van_id']]=True
								flag=True
							if(flag):
								break
					if(flag):
						break
					
			print(flag,checked)
			if((flag) and (not(checked))):
				checked=True
				iteration=iteration-1
				continue
			
			print('BIKE BEATS CHECK---------------------------------------')
			if((len(bike_beats)<=0) and (len(bikes)>0) and (not(checked2))):
				print('entered')
				if(len(ids_copy)>0):
				   for k in grup_ol_weights.keys():
					   if((grup_ol_weights[k][grup_ol_weights[k].index.isin(list(set(ids_copy)-set(list(grup_rem_weight[k].keys()))))].sum()<30*len(bikes)) and (grup_ol_weights[k][grup_ol_weights[k].index.isin(list(set(ids_copy)-set(list(grup_rem_weight[k].keys()))))].sum()>0)):
						   flag2=True
						   break
				else:
					if(len(owned_bikes_to_fill)>0):
						mylist = [True if b['cum_weight']< 30*len(owned_bikes_to_fill) else False for b in beat_list2]
						if(any(mylist)):
							flag2=True
			
			print(flag2,checked2)
			if((flag2) and (not(checked2))):
				checked2=True
				iteration=iteration-1
				continue
							
		#add fixed rate,mutitrips,plgmapping,exclusivity,area
			def check_assign_underuts_owned(b2,count):
					b3=[]
					owned_vans_to_consider=[]
					eligible_vans_foreach_beat={}
					for v in owned_van_order_tofill:
						if(van_multitrip_dict[v]=='no'):
							if((str(van_plg_mapping[v])=='nan') and (str(van_areassign_dict[v])=='nan') and (str(van_orderexclu_dict[v])=='nan')):
								owned_vans_to_consider.append(v)
					
					b2index=0 
					mts={}
					beat_cost_dict={}
					for b in b2:
						beat_cost_dict[b2index]={}
						eligible_vans_foreach_beat[b2index]=[b['van_id']]
						if(van_fixedrate_dict[b['van_id']]=='yes'):
							beat_cost_dict[b2index][b['van_id']]=van_cost_dict[b['van_id']]
						else:
							if(van_perkmrate_dict[b['van_id']]==0):
							   beat_cost_dict[b2index][b['van_id']]=van_baserate_dict[b['van_id']]+(b['end_time']/60)*van_perhourrate_dict[b['van_id']]
							else:
							   beat_cost_dict[b2index][b['van_id']]=van_baserate_dict[b['van_id']]+(b['end_time']*(1/van_speed_dict[b['van_id']]))*van_perkmrate_dict[b['van_id']] 
						
						if((str(van_plg_mapping[b['van_id']])!='nan') or (str(van_areassign_dict[b['van_id']])!='nan') or (str(van_orderexclu_dict[b['van_id']])!='nan')):
							b2index=b2index+1 
							continue
						for v in owned_vans_to_consider:
							if((b['cum_weight']<=van_weight_dict[v]) and (b['end_time']<=van_endtime_dict[v]) and (b['bills']<=van_outlet_dict[v]) and (b['cum_volume']<=van_volume_dict[v]) and (sum([len(bl) for bl in b['bills_cov']])<=van_bill_dict[v]) and (v!=b['van_id'])):
								eligible_vans_foreach_beat[b2index].append(v)
								if(van_fixedrate_dict[v]=='yes'):
									beat_cost_dict[b2index][v]=van_cost_dict[v]
								else:
									if(van_perkmrate_dict[v]==0):
									   beat_cost_dict[b2index][v]=van_baserate_dict[v]+(b['end_time']/60)*van_perhourrate_dict[v]
									else:
									   beat_cost_dict[b2index][v]=van_baserate_dict[v]+(b['end_time']*(1/van_speed_dict[v]))*van_perkmrate_dict[v] 
		
						if((van_multitrip_dict[b['van_id']]=='yes') and (int(b['van_id'].split('_')[-1])==1)):
							mts[(b2index,b['van_id'])]=[]
							k=(b2index,b['van_id'])
						elif((van_multitrip_dict[b['van_id']]=='yes') and (int(b['van_id'].split('_')[-1])>1)):
							if('_'.join(b['van_id'].split('_')[:-1])+'_1' in [k[1] for k in mts.keys()]):
								mts[k].append((b2index,b['van_id']))
						b2index=b2index+1  
					
					bv_var= pulp.LpVariable.dicts("beat van ",((bi,v) for bi in eligible_vans_foreach_beat.keys() for v in eligible_vans_foreach_beat[bi]),lowBound=0,upBound=1,cat='Binary')
					model1 = pulp.LpProblem("cost", pulp.LpMinimize)
					model1 += pulp.lpSum([bv_var[(bi,v)]*beat_cost_dict[bi][v] for bi in eligible_vans_foreach_beat.keys() for v in eligible_vans_foreach_beat[bi]])    
				
					for bi in eligible_vans_foreach_beat.keys():
						model1 += pulp.lpSum([bv_var[(bi,v)] for v in eligible_vans_foreach_beat[bi]])==1
					
					for ov in owned_van_order_tofill:
						model1 += pulp.lpSum([bv_var[(bi,v)] for bi in eligible_vans_foreach_beat.keys() for v in eligible_vans_foreach_beat[bi] if ov==v])<=1
						
					for k in mts.keys():
						for b in mts[k]:
							model1+=pulp.lpSum(bv_var[k])>=bv_var[b]*1
							
					result=model1.solve(pulp.PULP_CBC_CMD(maxSeconds=100))  
				  
					if(result==1):
						for bi in eligible_vans_foreach_beat.keys():
							for v in eligible_vans_foreach_beat[bi]:
								if(bv_var[bi,v].varValue==1):
									b2[bi]['van_id']=v
									b3.append(b2[bi])
								
					return b3
						
			for b in beat_list2:
				print(b['van_id'],b['cum_weight'],van_weight_dict[b['van_id']])
				
			print('REARRANGE---------------------------------------')
			owned_van_trips=len(set(['_'.join(v.split('_')[:-1]) if(van_multitrip_dict[v]=='yes') else v for v in owned_van_order_tofill]))
			used_owned_van_trips=len(set(['_'.join(b['van_id'].split('_')[:-1]) if(van_multitrip_dict[b['van_id']]=='yes') else b['van_id'] for b in beat_list2]))
			if(used_owned_van_trips<owned_van_trips):
				beat_list2=check_assign_underuts_owned(beat_list2,owned_van_trips-len(beat_list2))
				
			for b in beat_list2:
				print(b['van_id'],b['cum_weight'],van_weight_dict[b['van_id']])
				
			beat_list=[]
			cnt=0
			uu=False
			least_wgt_uu_beat=None
			
			#define underutilised and you shud also write about using 3t until its underutilised
			def underutilized(b,b_type):
				
				utilised=b['cum_weight']
				if(b_type=='Rented'):
					total=van_weight_dict[b['van_id'].split('_')[0]]
					apt_van=b['van_id'].split('_')[0]
				else:
					total=van_weight_dict[b['van_id']]
					apt_van=b['van_id']
				if(b_type=='Owned'):
					for i in range(len(owned_van_order_tofill)):
						if(total>van_weight_dict[owned_van_order_tofill[i]]):
							apt_van=owned_van_order_tofill[i]
							break
				else:
					for i in range(len(rental_van_order_tofill)):
						if(total>van_weight_dict[rental_van_order_tofill[i]]):
							apt_van=rental_van_order_tofill[i]
							break
			
				if((utilised<0.68*total) and (abs(total-utilised)>=abs(utilised-van_weight_dict[apt_van]))):
					return True
				'''
				if((apt_van==b['van_id']) and (b_type=='Owned')):
					for i in range(len(owned_van_order_tofill)):
						if((total==van_weight_dict[owned_van_order_tofill[i]]) and (van_endtime_dict[owned_van_order_tofill[i]]<van_endtime_dict[b['van_id']]) and (van_endtime_dict[owned_van_order_tofill[i]]>b['end_time'])):
							apt_van=owned_van_order_tofill[i]
					if(apt_van!=b['van_id']):
						return True
				'''
				return False
			
			'''
			for b in beat_list2:
			   if(underutilized(b,'Owned')): 
				   uu=True
				   least_wgt_uu_beat=b
			   
			if(uu and (len(ids_copy)>0)):
				for r in range(len(rental_van_order_tofill)):
				  if(van_weight_dict[rental_van_order_tofill[r]]<van_weight_dict[least_wgt_uu_beat['van_id']]):
					  #print('breAKKKK')    
					  rental_van_order_tofill=rental_van_order_tofill[r:]
					  break
				  if(r==len(rental_van_order_tofill)-1):
					  rental_van_order_tofill=[rental_van_order_tofill[-1]]
					  break
			'''  
			b3copy=[]  
			curb={}          
			def form_rental_beats(ids_copy,cnt,i,beat_list3,rem_weight={},ren_typ='RENTAL'):
				#cnt=0
				#i=0 
				print(len(ids_copy))
		#        for k in grup_ol_weights.keys():
		#            if('AP50001805656' in grup_ol_weights[k]):
		#                print(k,grup_ol_weights[k]['AP50001805656'])
				global rental_van_order_tofill
				ids=ids_copy.copy()          
				while(len(ids_copy)>0):
						print(len(ids_copy))
						if(len(ids_copy)<5):
							print(ids_copy)
		#                for k in grup_ol_weights.keys():
		#                    if('AP50001805656' in grup_ol_weights[k]):
		#                        print(k,grup_ol_weights[k]['AP50001805656'])
						print('------------------------------------')
						if(i<len(rental_van_order_tofill)):
							van_id=rental_van_order_tofill[i]
						cnt=cnt+1
						beat_list=[]
						if(len(ids_copy)==1):
							for k in grup_rem_weight.keys():
								if(ids_copy[0] in grup_rem_weight[k]):
									if(grup_rem_weight[k][ids_copy[0]]>van_weight_dict[van_id]):
										#print(i)
										j=i
										for i in range(j,-1,-1):
											#print(i,ids_copy[0])
											if(van_weight_dict[rental_van_order_tofill[i]]>=grup_rem_weight[k][ids_copy[0]]):
												sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel=form_normal_beats(k,ids_copy[0],rental_van_order_tofill[i])
												beat_list.append([sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,k,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel])  
												ids_copy=[]
												van_id=rental_van_order_tofill[i]
												break
										break
						for id in ids_copy:
							for key in grup_of_outlets.keys():
		#                        if len(van_id.split('_'))==3:
		#                            van_id=van_id.split('_')[0]
		#                        
								if(id in outlets_allowed_forvan[van_id]):
									if(id in grup_of_outlets[key]):
										#print(id)
										sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel=form_normal_beats(key,id,van_id)
										beat_list.append([sequence,end_time_seq,num_stores_seq,cumulative_weight_seq,wgt_sequence,key,cumulative_volume_seq,vol_sequence,time_sequence,bills_cov,base_pack_cov,plg,channel])
										#print(sequence,end_time_seq,cumulative_weight_seq)
								else:
									break
						if(len(beat_list)>0) :  
							
							beat_list_clone2=[]
							beat_list_clone_flag2=0
							for obp in min_cap_van_high_bp_demand_outs.keys():
								for k in grup_rem_weight.keys():
									o=obp.split('_')[0]
									bp=obp.split('_')[1]
									if((o in grup_rem_weight[k].keys()) and (bp in grup_rem_basepack[k][o]) and (grup_rem_weight[k][o]>1000) and (van_tonnage_dict[van_id]==min_cap_van_high_bp_demand_outs[obp])):
										for b in beat_list:
											if((o in b[0]) and (bp in [item for sublist in b[10] for item in sublist])):
												beat_list_clone2.append(b)
							
							if(len(beat_list_clone2)>0):
								print('beat_list_clone')
								beat_list_clone_flag2=1
								beat_list=beat_list_clone2.copy()
								
							beat=find_best_beat(beat_list,van_id,cnt,'Rented')
							print(beat['sequence'],beat['cum_weight'],van_id)
							if((not(underutilized(beat,'Rented'))) or (i>=len(rental_van_order_tofill)-1) or (len(ids_copy)==0) or (beat_list_clone_flag2==1)):
								print('not underutilised')
								if(set(beat['sequence'])=={'Distributor'}):
									print('Posiibility of Infine loop hence terminating')
									break
								beat_list3.append(beat)
								#add_to_output(beat)
		#                        if('AP50001805656' in beat['sequence']):
		#                            for i in range(0,len(beat['sequence'])):
		#                                if(beat['sequence'][i]=='AP50001805656'):
		#                                    print('AP50001805656',beat['wgt_sequence'][i])
								ky=beat['del_type']
								print(len(grup_of_outlets[ky]),len(beat['sequence']))
								grup_of_outlets[ky]=list(set(grup_of_outlets[ky])-set(beat['sequence']))
								print(len(grup_of_outlets[ky]))
								if(ren_typ=='RENTAL'):
									if ky in common_outs_accross_grups.keys():
										if(len(set(beat['sequence']).intersection(set([op.split('_')[0] for op in grups_common_outs.keys()])))>=1):
											for op in grups_common_outs.keys():
												if(op.split('_')[0] in beat['sequence']):
													o=op.split('_')[0]
													ps=[]
													cs=[]
													for i in range(0,len(beat['sequence'])):
														if(beat['sequence'][i]==o):
															ps=beat['plg'][i]
															cs=beat['channel'][i]
															break
													if ((op.split('_')[1] in ps) and (op.split('_')[2] in cs)):
														for g in grups_common_outs[op]:
															if((g!=ky)):
																if(not(o in grup_rem_weight[g].keys())):
																	#print(o,op,ky,g)
																	#curb=beat
																	#b3copy=beat_list3.copy()
																	grup_ol_weights[g][o]=grup_ol_weights[g][o]-grup_olplg_weights[op]
																	if(grup_ol_weights[g][o]<=0.1):
																		grup_of_outlets[g].remove(o)
																		del grup_ol_weights[g][o]
																		del grup_ol_volume[g][o]
																		del grup_ol_service_time[g][o]
																		del grup_outlet_bill_dict[g][o]
																		del grup_outlet_basepack_dict[g][o]
																		del grup_ol_plg[g][o]
																		del grup_ol_channel[g][o]
																	else:
																		grup_ol_volume[g][o]=grup_ol_volume[g][o]-grup_olplg_volume[op]
																		service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
																		grup_ol_service_time[g][o]=int(service_time_details[(grup_ol_weights[g][o]>=service_time_details['load_range_from_kgs'].astype(float)) & (grup_ol_weights[g][o]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
																		grup_outlet_bill_dict[g][o]=set(grup_outlet_bill_dict[g][o])-set(grup_olplg_bill_dict[op])   
																		grup_outlet_basepack_dict[g][o]=set(grup_outlet_basepack_dict[g][o])-set(grup_olplg_basepack_dict[op])
																		grup_ol_plg[g][o]=[p for p in grup_ol_plg[g][o] if p!=op.split('_')[1]]
																		grup_ol_channel[g][o]=[p for p in grup_ol_channel[g][o] if p!=op.split('_')[2]]
																else:
																	#print(o,op,ky,g)
																	#curb=beat
																	#b3copy=beat_list3.copy()
																	for i in range(len(beat['sequence'])):
																		if (beat['sequence'][i]==o):
																			bp_included_beat=beat['base_pack_cov'][i]
																			break
																	#print(set(bp_included_beat))
																	'''
																	l=g.split('_')    
																	if(len(l)>2):
																		sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['SERVICING PLG'].isin(l[2].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)        
																	else:
																		sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)
																	
																	if (g in common_outs_accross_grups.keys()):
																		if(o in [op.split('_')[0] for op in common_outs_accross_grups[g]]):
																			for opc in common_outs_accross_grups[g]:
																				if(o in opc.split('_')):
																					print(opc)
																				p=opc.split('_')[1]
																				c=opc.split('_')[2]
																			if(len(sub_df[(sub_df['primarychannel']==c)  & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==o)])<1):
																				sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c)  & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==o)].copy(deep=True)])
																	'''
																	sub_df=input_data.copy(deep=True)
																	sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str) 
																	sub_df=sub_df[(sub_df['PARTY_HLL_CODE']==o) & (sub_df['BASEPACK CODE'].isin(list(grup_rem_basepack[g][o]))) & (sub_df['SERVICING PLG']==op.split('_')[1])].copy(deep=True)
																	sub_df=sub_df[sub_df['BASEPACK CODE'].isin(bp_included_beat)].copy(deep=True)
						
																	grup_rem_weight[g][o]=grup_rem_weight[g][o]-sub_df['NET_SALES_WEIGHT_IN_KGS'].sum()
																	if(grup_rem_weight[g][o]<=0.1):
																		grup_of_outlets[g].remove(o)
																		del grup_ol_weights[g][o]
																		del grup_ol_volume[g][o]
																		del grup_ol_service_time[g][o]
																		del grup_outlet_bill_dict[g][o]
																		del grup_outlet_basepack_dict[g][o]
																		del grup_ol_plg[g][o]
																		del grup_ol_channel[g][o]
																		del grup_rem_weight[g][o]
																		del grup_rem_basepack[g][o]
																		del grup_rem_volume[g][o]
																	else:
																	   grup_rem_volume[g][o]= grup_rem_volume[g][o]-sub_df['volume'].sum()
																	   grup_rem_basepack[g][o]=set(grup_rem_basepack[g][o])-set(list(sub_df['BASEPACK CODE']))    
								
								for h in grup_rem_weight[ky].keys():
									if((h not in grup_of_outlets[ky]) and (h in beat['sequence'])):
										wgt_included_beat=sum([beat['wgt_sequence'][i] for i in range(len(beat['sequence'])) if beat['sequence'][i]==h])
										vol_included_beat=sum([beat['vol_sequence'][i] for i in range(len(beat['sequence'])) if beat['sequence'][i]==h])
										bp_included_beat=[]
										for i in range(len(beat['sequence'])):
											if (beat['sequence'][i]==h):
												bp_included_beat=beat['base_pack_cov'][i]
												break
										l=ky.split('_')    
										if(len(l)>2):
											sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['SERVICING PLG'].isin(l[2].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)        
										else:
											sub_df=input_data[(input_data['primarychannel'].isin(l[1].split('|'))) & (input_data['area_name'].isin(l[0].split('|')))].copy(deep=True)
										
										if (ky in common_outs_accross_grups.keys()):
											if(h in [op.split('_')[0] for op in common_outs_accross_grups[ky]]):
												for opc in common_outs_accross_grups[ky]:
													p=opc.split('_')[1]
													c=opc.split('_')[2]
												if(len(sub_df[(sub_df['primarychannel']==c)  & (sub_df['SERVICING PLG']==p) & (sub_df['PARTY_HLL_CODE']==h)])<1):
													sub_df=pd.concat([sub_df,input_data[(input_data['primarychannel']==c)  & (input_data['SERVICING PLG']==p) & (input_data['PARTY_HLL_CODE']==h)].copy(deep=True)])
									
										sub_df['BASEPACK CODE']=sub_df['BASEPACK CODE'].astype(str) 
										sub_df=sub_df[(sub_df['PARTY_HLL_CODE']==h) & (sub_df['BASEPACK CODE'].isin(list(grup_rem_basepack[ky][h])))].copy(deep=True)
										sub_df=sub_df[~sub_df['BASEPACK CODE'].isin(bp_included_beat)].copy(deep=True)
										#grup_rem_bills[ky][h]=set(sub_df['BILL_NUMBER'].unique())
										grup_rem_basepack[ky][h]=set(sub_df['BASEPACK CODE'].unique())
										#print('wgt_included_beat')
										#print(wgt_included_beat)
										r_w=grup_rem_weight[ky][h]-wgt_included_beat
										r_v=grup_rem_volume[ky][h]-vol_included_beat
										#print('weight_rem')
										#print(r_w)
										if((r_w>0) and (r_v>0)):
											grup_of_outlets[ky].append(h)
										grup_rem_weight[ky][h]=r_w
										grup_rem_volume[ky][h]=r_v
									if((h not in grup_of_outlets[ky]) and (grup_rem_weight[ky][h]>0)):
										grup_of_outlets[ky].append(h)
								ids=[]
								for k in grup_of_outlets.keys():
									ids.extend(grup_of_outlets[k])
								ids_copy=list(set(ids)).copy()
							else:
								print('under uts')
								cnt=cnt-1
								i=i+1
								continue
						else:
							print('beats empty')
							cnt=cnt-1
							i=i+1
							continue
				return beat_list3,i
			print('---------------------RENTAL---------------------------------')
			j=0
			rental_beats=[]
			stack=[]
			found=False
			to_remove=[]
			print(grup_rem_weight)
			print(len(ids_copy))
			rental_beats,van=form_rental_beats(ids_copy,0,0,[],'RENTAL')
			stack.append(rental_beats)
			print('---------------------RENTAL22222---------------------------------')
			if((len(rental_beats)>0) and (van>0)):
				prev_bt=rental_beats[-1]
				if(underutilized(rental_beats[-1],'Rented')):
					if((van-1>=0)):
						found=False
						to_remove=[]
						for bt in rental_beats[::-1]:
							vanid_sliced=bt['van_id'].split('_')
							to_remove.append(bt['van_id'])
							if(vanid_sliced[0] in rental_van_order_tofill[0:van]):
								found=True
								#to_remove.append(bt['van_id'])
								cnt=int(vanid_sliced[1])
								index=rental_van_order_tofill.index(vanid_sliced[0])+1
								if(index==len(rental_van_order_tofill)):
									index=index-1
								btlst3=[bt for bt in rental_beats if not(bt['van_id'] in to_remove)]
								ids_copyy=[]
								rem_wgt_dict={}
								for beat in rental_beats:
									if(beat['van_id'] in to_remove):
										ky=beat['del_type']
										for h in grup_high_demand_outs[ky]:
											grup_rem_weight[ky][h]=0
											grup_rem_volume[ky][h]=0    
											#grup_rem_bills[ky][h]={}
											grup_rem_basepack[ky][h]={}
										for i in range(len(beat['sequence'])):
											if (beat['sequence'][i] in grup_high_demand_outs[ky]):
												grup_rem_weight[ky][beat['sequence'][i]]+=beat['wgt_sequence'][i] 
												grup_rem_volume[ky][beat['sequence'][i]]+=beat['vol_sequence'][i] 
												#grup_rem_bills[ky][beat['sequence'][i]]=set(beat['bills_cov'])
												grup_rem_basepack[ky][beat['sequence'][i]]=set(beat['base_pack_cov'][i])
										ids_copyy.extend(list(set(beat['sequence'])-{'Distributor'}))
										grup_of_outlets[ky].extend(list(set(beat['sequence'])-{'Distributor'}))
								break
							prev_bt=bt
						if((found) and (len(to_remove)>0)):
							rental_beats,van=form_rental_beats(ids_copyy,cnt,index,btlst3,rem_wgt_dict,'RENTAL2')
							stack.append(rental_beats)
			   
							
			min_cost_rental_beats=[]
			beat_list4=[]
			cost=1000000000
			for s in stack:
				 s_cost=0
				 for b in s:
					 s_cost+=van_cost_dict[b['van_id'].split('_')[0]]
				 if(s_cost<cost):
					 beat_list4=s.copy()
					 cost=s_cost
								   
			
			print('--------------------------beat_list4 len---------------------------------------')
			print(len(beat_list4))
			print('--------------------------beat_list4 len---------------------------------------')
			output_stack.append([])
			for b in bike_beats:
				output_stack[iteration-1].append(b)
			for b in beat_list2:
				output_stack[iteration-1].append(b)
			for b in beat_list4:
				print('add rental beats')
				output_stack[iteration-1].append(b)
		
		
			cnt1=0
			cnt2=0
			if (iteration<=1):
				for b in beat_list2:
					cnt1=cnt1+1
					if(str(van_plg_mapping[b['van_id']])!='nan'):
						continue
					if(str(van_areassign_dict[b['van_id']])!='nan'):
						continue
					if(str(van_orderexclu_dict[b['van_id']])!='nan'):
						continue
					if(underutilized(b,'Owned')): 
						cnt1=cnt1-1
						key=b['del_type']
						area=key.split('_')[0]
						channel=key.split('_')[1].split('|')
						if len(key.split('_'))==3:
							plgs=key.split('_')[2].split('|')
						else:
							plgs=['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']
						if(area=='All'):
							typ='default_underutilized'
						else:
							typ='exception_underutilized'
						for ot in set(b['sequence'])-{'Distributor'}:
							_plg_clubbing_=plg_clubbing_.copy(deep=True)
							_plg_clubbing_['area'] = _plg_clubbing_['area'].apply(lambda x: ','.join(x))
							plg_clubbing_new=_plg_clubbing_[(_plg_clubbing_['area']==area) & (_plg_clubbing_['type']==typ)].copy(deep=True) 
							#channel_new=plg_clubbing_new['channel']
							channels_new=list(plg_clubbing_new['channel'])
		
							cs=grup_ol_channel[key][ot]
							ps=grup_ol_plg[key][ot]
							for y in channels_new:
								if all(x in y for x in cs):
									new_channel=y
							plg_clubbing_new['channel'] = plg_clubbing_new['channel'].apply(lambda x: ','.join(x))
							plg_clubbing_new2=plg_clubbing_new[plg_clubbing_new['channel']==','.join(y)].copy(deep=True)
							c_toadd=list(set(new_channel).difference(set(cs)))
							plgclub=plg_clubbing_new2['groups']
							if(len(plgclub)<=1):
								plgclub=[['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']]
							plggrup=[]      
							for g in grups.keys():
								if(g!=key) and (g.split('_')[0]==area):
									gcs=g.split('_')[1].split('|')
									if(len(g.split('_'))==3):   
										gps=g.split('_')[2].split('|')
									else:
										gps=['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']
									if len(set(c_toadd).intersection(set(gcs)))>=1:
										for p in set(ps):
											#p_toadd=[p2 for club in plgclub if p in club for p2 in club if p2!=p]
											for club in plgclub:
												if p in club:
													plggrup=club
											if len(set(plggrup).intersection(set(gps)))>=1:
												if g in common_outs_accross_grups.keys():
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												else:
													common_outs_accross_grups[g]=[]
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												if ot+'_'+p+'_'+cs[0] in grups_common_outs.keys():
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
												else:
													grups_common_outs[ot+'_'+p+'_'+cs[0]]=[]
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
													
											
									elif len(set(new_channel).intersection(set(gcs)))>=1:
										for p in set(ps):
											#p_toadd=[p2 for club in plgclub if p in club for p2 in club if p2!=p]
											for club in plgclub:
												if p in club:
													plggrup=club
											if ((not(p in gps)) and (len(set(plggrup).intersection(set(gps)))>=1)):
												if g in common_outs_accross_grups.keys():
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												else:
													common_outs_accross_grups[g]=[]
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												if ot+'_'+p+'_'+cs[0] in grups_common_outs.keys():
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
												else:
													grups_common_outs[ot+'_'+p+'_'+cs[0]]=[]
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
							for p in set(ps):
								if key in common_outs_accross_grups.keys():
									common_outs_accross_grups[key].append(ot+'_'+p+'_'+cs[0])
								else:
									common_outs_accross_grups[key]=[]
									common_outs_accross_grups[key].append(ot+'_'+p+'_'+cs[0])
								if ot+'_'+p+'_'+cs[0] in grups_common_outs.keys():
									grups_common_outs[ot+'_'+p+'_'+cs[0]].append(key)
								else:
									grups_common_outs[ot+'_'+p+'_'+cs[0]]=[]
									grups_common_outs[ot+'_'+p+'_'+cs[0]].append(key)
				
				for b in beat_list4:
					cnt2=cnt2+1
					if(underutilized(b,'Rented')): 
						print('underuts--------in rental beats---hence change clubbing')
						cnt2=cnt2-1
						key=b['del_type']
						area=key.split('_')[0]
						channel=key.split('_')[1].split('|')
						if len(key.split('_'))==3:
							plgs=key.split('_')[2].split('|')
						else:
							plgs=['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']
						if(area=='All'):
							typ='default_underutilized'
						else:
							typ='exception_underutilized'
		
						for ot in set(b['sequence'])-{'Distributor'}:
							_plg_clubbing_=plg_clubbing_.copy(deep=True)
							_plg_clubbing_['area'] = _plg_clubbing_['area'].apply(lambda x: ','.join(x))
							plg_clubbing_new=_plg_clubbing_[(_plg_clubbing_['area']==area) & (_plg_clubbing_['type']==typ)].copy(deep=True) 
							#channel_new=plg_clubbing_new['channel']
							channels_new=list(plg_clubbing_new['channel'])
		
							cs=grup_ol_channel[key][ot]
							ps=grup_ol_plg[key][ot]
							for y in channels_new:
								if all(x in y for x in cs):
									new_channel=y
							plg_clubbing_new['channel'] = plg_clubbing_new['channel'].apply(lambda x: ','.join(x))
							plg_clubbing_new2=plg_clubbing_new[plg_clubbing_new['channel']==','.join(y)].copy(deep=True)
							c_toadd=list(set(new_channel).difference(set(cs)))
							plgclub=plg_clubbing_new2['groups']
							if(len(plgclub)<=1):
								plgclub=[['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']]
								   
							for g in grups.keys():
								if(g!=key) and (g.split('_')[0]==area):
									gcs=g.split('_')[1].split('|')
									if(len(g.split('_'))==3):   
										gps=g.split('_')[2].split('|')
									else:
										gps=['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']
									if len(set(c_toadd).intersection(set(gcs)))>=1:
										for p in set(ps):
											#p_toadd=[p2 for club in plgclub if p in club for p2 in club if p2!=p]
											for club in plgclub:
												if p in club:
													plggrup=club
											if len(set(plggrup).intersection(set(gps)))>=1:
												#print(b['del_type'],g,ot+'_'+p+'_'+cs[0])
												if g in common_outs_accross_grups.keys():
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												else:
													common_outs_accross_grups[g]=[]
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												if ot+'_'+p+'_'+cs[0] in grups_common_outs.keys():
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
												else:
													grups_common_outs[ot+'_'+p+'_'+cs[0]]=[]
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
											
									elif len(set(new_channel).intersection(set(gcs)))>=1:
										for p in set(ps):
											#p_toadd=[p2 for club in plgclub if p in club for p2 in club if p2!=p]
											for club in plgclub:
												if p in club:
													plggrup=club
											if ((not(p in gps)) and (len(set(plggrup).intersection(set(gps)))>=1)):
												#print(b['del_type'],g,ot+'_'+p+'_'+cs[0])
												if g in common_outs_accross_grups.keys():
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												else:
													common_outs_accross_grups[g]=[]
													common_outs_accross_grups[g].append(ot+'_'+p+'_'+cs[0])
												if ot+'_'+p+'_'+cs[0] in grups_common_outs.keys():
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
												else:
													grups_common_outs[ot+'_'+p+'_'+cs[0]]=[]
													grups_common_outs[ot+'_'+p+'_'+cs[0]].append(g)
											
							for p in set(ps):
								if key in common_outs_accross_grups.keys():
									common_outs_accross_grups[key].append(ot+'_'+p+'_'+cs[0])
								else:
									common_outs_accross_grups[key]=[]
									common_outs_accross_grups[key].append(ot+'_'+p+'_'+cs[0])
								if ot+'_'+p+'_'+cs[0] in grups_common_outs.keys():
									grups_common_outs[ot+'_'+p+'_'+cs[0]].append(key)
								else:
									grups_common_outs[ot+'_'+p+'_'+cs[0]]=[]
									grups_common_outs[ot+'_'+p+'_'+cs[0]].append(key)
			
		
			flag=False 
			flag2=False  
			checked=False 
			checked2=False                                
			if((iteration<=1) and (cnt1==len(beat_list2)) and (cnt2==len(beat_list4))):
				break
		
		outlets_allowed_forvan=outs_allowed_forvan_copy.copy() 
		for k in outlets_allowed_for_bike_copy.keys():
			outlets_allowed_forvan[k]=[]
			outlets_allowed_forvan[k].extend(outlets_allowed_for_bike_copy[k])
		for s in output_stack:
			for b in s:
			  print(b['van_id'],b['cum_weight'],b['sequence'])
			print('-----------')  
						
		tot_cost=10000000
		count=0
		#print(outlets_allowed_forvan['TATA ACE SHIKHAR'])
		for s in output_stack:
			 s_cost=0
			 for b in s:
				 if(b['van_id'].split('_')[-1]=='Rented'):
					 s_cost+=van_cost_dict[b['van_id'].split('_')[0]] 
				 else:
					 s_cost+=van_cost_dict[b['van_id']]
					 
			 if(s_cost<tot_cost):
				 selected_beats_index=count
				 beat_list5=s.copy()
				 tot_cost=s_cost
			 count=count+1
		
		output_df=pd.DataFrame() 
		
		for b in beat_list5:
			add_to_output(b)
			
		lat_long['path']=lat_long.index    
		print(output_df.shape)
		print(output_df.columns)
		output_df2 = pd.merge(output_df, lat_long, left_on = 'path', right_on = 'partyhll_code', how = 'left')    
		output_df2.to_csv(str(rscode)+'_1v9.csv')
		
		rs_lat=input_data['rs_latitude'].unique()[0]
		rs_long=input_data['rs_longitude'].unique()[0]
		
		
		edf=output_df2[output_df2['van_id'].isin(list(output_df2[output_df2['path_x'].isin(exclusive_outlets)]['van_id'].unique())+bikes)].copy(deep=True)
		tdf=output_df2[~output_df2['van_id'].isin(list(output_df2[output_df2['path_x'].isin(exclusive_outlets)]['van_id'].unique())+bikes)].copy(deep=True)
		
		new_tdf=pd.DataFrame()
		  
		
		for del_type in set(tdf['del_type'].unique()):
			
			df=tdf[tdf['del_type']==del_type].reset_index(drop=True)
			df_inittt=df.copy(deep=True)
			beats=df['van_id'].unique()
			df['day']='day'
			
			def resequence(clusterdf):
				#print('resequence')
				w_dict=clusterdf.groupby(['path_x'])['weights'].sum().to_dict()
				v_dict=clusterdf.groupby(['path_x'])['volumes'].sum().to_dict()
				b_dict=clusterdf.groupby(['path_x'])['bill_numbers'].apply(list).to_dict()
				for k in b_dict.keys():
					l=[]
					l.extend(b_dict[k])
					b_dict[k]=l            
					
				for k in b_dict.keys():
					l=[]
					for sub in b_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9_]+', ',', str(sub))).split(',')) 
					b_dict[k]=[i for i in l if i!='']
					
				bp_dict=clusterdf.groupby(['path_x'])['Basepack'].apply(list).to_dict()
				for k in bp_dict.keys():
					l=[]
					l.extend(bp_dict[k])
					bp_dict[k]=l
				
				for k in bp_dict.keys():
					l=[]
					for sub in bp_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9_]+', ',', str(sub))).split(',')) 
					bp_dict[k]=[i for i in l if i!=''] 
				
				c_dict=clusterdf.groupby(['path_x'])['channel'].apply(list).to_dict()    
				p_dict=clusterdf.groupby(['path_x'])['plg'].apply(list).to_dict()
				
				for k in c_dict.keys():
					l=[]
					l.extend(c_dict[k])
					c_dict[k]=l
				
				for k in c_dict.keys():
					l=[]
					for sub in c_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9]+', ',', str(sub))).split(',')) 
					c_dict[k]=[i for i in l if i!=''] 
				
				for k in p_dict.keys():
					l=[]
					l.extend(p_dict[k])
					p_dict[k]=l
				
				for k in p_dict.keys():
					l=[]
					for sub in p_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(sub))).split(',')) 
					p_dict[k]=[i for i in l if i!=''] 
					
				#b_dict=dict(zip(clusterdf['path_x'],clusterdf['bill_numbers']))
				#bp_dict=dict(zip(clusterdf['path_x'],clusterdf['Basepack']))
				lat_dict=dict(zip(clusterdf['path_x'],clusterdf['outlet_latitude']))
				long_dict=dict(zip(clusterdf['path_x'],clusterdf['outlet_longitude']))
				lat_dict['Distributor']=rs_lat
				long_dict['Distributor']=rs_long
				pathseq=['Distributor']
				wgtseq=[0]
				volseq=[0]
				timeseq=[0]
				billseq=[[]]
				bpseq=[[]]
				chseq=[[]]
				plgseq=[[]]
				deltyp=list(clusterdf['del_type'].unique())[0]
				numbills=0
				cum_weight=0
				cum_volume=0
				van_id=list(clusterdf['van_id'].unique())[0]
				van_id_init=van_id
				day=list(clusterdf['day'].unique())[0]
				latseq=[lat_dict['Distributor']]
				longseq=[long_dict['Distributor']]
				mids=list(set(w_dict.keys())-{'Distributor'})
				cnt=0
				#print('resequence while start')
				while(cnt!=len(mids)):
					cnt=cnt+1
					end_time_seq=0
					dist = distance_matrix[pathseq[-1]][distance_matrix[pathseq[-1]].index.isin(list(set(mids)-set(pathseq)))]
					if(len(dist)==0):
						break
					dist = pd.Series(dist)
					nearest_index = dist.idxmin()
					pathseq.append(nearest_index)
					wgtseq.append(w_dict[nearest_index])
					volseq.append(v_dict[nearest_index])
					distance = dist[nearest_index]
					if (van_id.split('_')[-1]=='Rented'):
						van_id=van_id.split('_')[0]
					
					travel_time = math.ceil(distance * (1/van_speed_dict[van_id]))
					end_time_seq = end_time_seq + travel_time
					service_time=int(service_time_details[(w_dict[nearest_index]>=service_time_details['load_range_from_kgs'].astype(float)) & (w_dict[nearest_index]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
					end_time_seq=end_time_seq+ service_time
					timeseq.append(travel_time+service_time)
					billseq.append((b_dict[nearest_index]))
					bpseq.append((bp_dict[nearest_index]))
					chseq.append((c_dict[nearest_index]))
					plgseq.append((p_dict[nearest_index]))
					numbills=numbills+len(b_dict[nearest_index])
					cum_weight=cum_weight+w_dict[nearest_index]
					cum_volume=cum_volume+v_dict[nearest_index]
					latseq.append(lat_dict[nearest_index])
					longseq.append(long_dict[nearest_index])
				#print('resequence while end')    
				df_=pd.DataFrame()
				pathseq.append('Distributor')
				wgtseq.append(0)
				volseq.append(0)
				if van_id.split('_')[-1]=='Rented':
					timeseq.append(math.ceil(distance_matrix[pathseq[-2]]['Distributor'] * (1/van_speed_dict[van_id.split('_')[0]])))
				else:        
					timeseq.append(math.ceil(distance_matrix[pathseq[-2]]['Distributor'] * (1/van_speed_dict[van_id])))
				billseq.append([''])
				bpseq.append([''])
				chseq.append([''])
				plgseq.append([''])
				latseq.append(lat_dict['Distributor'])
				longseq.append(long_dict['Distributor'])
				
				return pd.concat([df_,pd.DataFrame({'path_x':pathseq, 'endtime':[sum(timeseq)]*len(pathseq), 'num_bills':[numbills]*len(pathseq), 'cum_weight':[cum_weight]*len(pathseq), 'van_id':[van_id_init]*len(pathseq), 'weights':wgtseq,'cum_volume':[cum_volume]*len(pathseq), 'volumes':volseq, 'del_type':deltyp,'time':timeseq, 'bill_numbers':billseq, 'Basepack':bpseq,'partyhll_code':pathseq, 'outlet_latitude':latseq, 'outlet_longitude':longseq, 'day':[day]*len(pathseq),'plg':plgseq,'channel':chseq})])
			
			
			def underutilized(bt,btyp):
				if(btyp=='Rented'):
					b=bt.split('_')[0]
				else:
					b=bt
				if(df[df['van_id']==bt]['weights'].sum()<0.68*van_weight_dict[b]):
					return True
				else:
					return False
			
			'''
			
			non_constrained_beats_underuts=False
			non_constrained_beats=[]
			for b in beats:
				flag=False
				c=0
				for lr in list(set(input_data['lane_restrictions'].unique())-{10000}):
					if(van_tonnage_dict[b]<lr):
						c=c+1
					if(c==len(list(set(input_data['lane_restrictions'].unique())-{10000}))):
						print(b)
						flag=True
				if((str(van_plg_mapping[b])=='nan') and (str(van_areassign_dict[b])=='nan') and (str(van_orderexclu_dict[b])=='nan') and (flag)):
					non_constrained_beats.append(b)
					
			for bt in non_constrained_beats:   
				if(bt.split('_')[-1]=='Rented'):
					if(underutilized(bt,'Rented')):
						non_constrained_beats_underuts=True
				else:
					if(underutilized(bt,'Owned')):
						non_constrained_beats_underuts=True
				
			
			'''
			
		
			def find_max_overlap_beats(beats,df): 
				
				max_overlap_percent=0
				max_overlap_bt1='nil'
				max_overlap_bt2='nil'
				overlap_outlets={}
				if(len(beats)==1):
					return max_overlap_bt1,max_overlap_bt2,{}
				for bt in beats:
					if((list(df[df['van_id']!=bt]['path_x'].unique())==['Distributor'])): 
						return max_overlap_bt1,max_overlap_bt2,{}
					if((list(df[df['van_id']==bt]['path_x'].unique())==['Distributor'])):
						continue
					overlap_outs_cnt=0
					overlap_clus=[]
					overlap_outs=[]
					for o in set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}:
						other_outs=list(set(df[df['van_id']!=bt]['path_x'].unique())-{'Distributor',o})
						beat_outs=list(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor',o})
						nbr_outside_beat=pd.Series(distance_matrix[o][distance_matrix[o].index.isin(other_outs)]).idxmin()
						if(len(beat_outs)>0):
							nbr_within_beat=pd.Series(distance_matrix[o][distance_matrix[o].index.isin(beat_outs)]).idxmin()
							if(distance_matrix[o][nbr_within_beat]>distance_matrix[o][nbr_outside_beat]):
								overlap_outs_cnt=overlap_outs_cnt+1
								overlap_outs.append(o)
								overlap_clus.append(df[df['path_x']==nbr_outside_beat]['van_id'].unique()[0])
						else:
							nbt=df[df['path_x']==nbr_outside_beat]['van_id'].unique()[0]
							if(nbt.split('_')[-1]=='Rented'):
								if(underutilized(nbt,'Rented')):
									overlap_outs_cnt=overlap_outs_cnt+1
									overlap_outs.append(o)
									# if there are more than one beat choose near by one
									overlap_clus.append(nbt)
							else:
								if(underutilized(nbt,'Owned')):
									overlap_outs_cnt=overlap_outs_cnt+1
									overlap_outs.append(o)
									# if there are more than one beat choose near by one
									overlap_clus.append(nbt)
							
					if(overlap_outs_cnt>0):        
						overlap_percent=((overlap_outs_cnt)/len(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}))*100
					else:
						overlap_percent=0
					if(overlap_percent>max_overlap_percent):
						overlap_outlets={}
						max_overlap_percent=overlap_percent
						max_overlap_bt1=bt
						from collections import Counter 
						def most_frequent(List): 
							occurence_count = Counter(List) 
							return occurence_count.most_common(1)[0][0] 	
						max_overlap_bt2=most_frequent(overlap_clus)
						for i in range(len(overlap_clus)):
							if(overlap_clus[i] in overlap_outlets.keys()):
								overlap_outlets[overlap_clus[i]].append(overlap_outs[i])
							else:
								overlap_outlets[overlap_clus[i]]=[]
								overlap_outlets[overlap_clus[i]].append(overlap_outs[i])
						
		
				return   max_overlap_bt1,max_overlap_bt2,overlap_outlets  
							
			 
			def check_if_both_comb_possible(bt1,bt2,comb_df,bt1_overlap_outs):
				
				c1=0
				if(bt1.split('_')[-1]=='Rented'):
					bt1=bt1.split('_')[0]
				if(bt2.split('_')[-1]=='Rented'):
					bt2=bt2.split('_')[0]
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
		#        if((len(set(outs).intersection(set(exclusive_outlets)))!=len(outs)) or (len(set(outs).intersection(set(exclusive_outlets)))!=0)):
		#            return False
				for o in outs:
					if(o in outlets_allowed_forvan[bt1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[bt2]):
						c2=c2+1
						
				if((c1==len(outs)) or (c2==len(outs))):
				#if((c2==len(outs))):
					return True
					
				else:
					return False
						
		
			def find_apt_van(comb_df,bt1,bt2):
				#if there is any ngbr beat in vicinity go for larger beat else go for smaller beat
				#usually this smaller beat wud be fully filled and this exercise wud be a waste of time
				#sometimes smaller beat wudnt make sense if in original beat it was fully utilised
				
				if(bt1.split('_')[-1]=='Rented'):
					b1=bt1.split('_')[0]
					b1typ='Rented'
				else:
					b1=bt1
					b1typ='Owned'
				if(bt2.split('_')[-1]=='Rented'):
					b2=bt2.split('_')[0]
					b2typ='Rented'
				else:
					b2=bt2
					b2typ='Owned'
				if((van_weight_dict[b1]<van_weight_dict[b2])):
					smaller_beat=bt1
					smaller_beat_typ=b1typ
					larger_beat=bt2
					larger_beat_typ=b2typ
				elif((van_weight_dict[b2]<van_weight_dict[b1])):
					smaller_beat=bt2
					smaller_beat_typ=b2typ
					larger_beat=bt1
					larger_beat_typ=b1typ
				else:
					if((van_endtime_dict[b1]<van_endtime_dict[b2])):
						smaller_beat=bt1
						smaller_beat_typ=b1typ
						larger_beat=bt2
						larger_beat_typ=b2typ
					else:
						smaller_beat=bt2
						smaller_beat_typ=b2typ
						larger_beat=bt1
						larger_beat_typ=b1typ
					
				c1=0
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
				for o in outs:
					if(o in outlets_allowed_forvan[b1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[b2]):
						c2=c2+1
						
				if((c1==len(outs)) and (c2!=len(outs))):
					return bt1,bt2
				elif((c2==len(outs)) and (c1!=len(outs))):
					return bt2,bt1
				else:
					if((larger_beat.split('_')[-1]=='Rented') and (smaller_beat.split('_')[-1]!='Rented')):
						return smaller_beat,larger_beat
					return larger_beat,smaller_beat
				'''
				if(not(underutilized(smaller_beat,smaller_beat_typ))):
					return larger_beat,smaller_beat
				else:
					if(not(underutilized(larger_beat,larger_beat_typ))):
						return smaller_beat,larger_beat
					else:
						return larger_beat,smaller_beat
				'''
			
			def find_apt_van_others(comb_df,othbeats):
				othbeats_2=[]
				for b in othbeats:
					if(b.split('_')[-1]=='Rented'):
						othbeats_2.append(b.split('_')[0])
					else:
						othbeats_2.append(b)
				
				candidate_vans=[]  
				ind=0
				for beat in othbeats_2:
					c1=0
					outs=set(comb_df['path_x'].unique())-{'Distributor'}
					for o in outs:
						if(o in outlets_allowed_forvan[beat]):
							c1=c1+1
					if(c1==len(outs)):
						candidate_vans.append(othbeats[ind])
					ind=ind+1
				mindiff=10000000
				min_utsper=100
				ind=0
				cand_van='nil'
				for v in candidate_vans:
				   v2=othbeats_2[othbeats.index(v)]
				   diff=abs(van_weight_dict[v2]-comb_df['weights'].sum()) 
				   if(diff<=mindiff):
					   utsper=(df[df['van_id']==v]['weights'].sum()/van_weight_dict[v2])*100
					   if(utsper<min_utsper):
						   cand_van=v
				   ind=ind+1
				return cand_van
			
			def make_three_beats(van_name,bt1,bt2,comb_df) :
				comb_df['van_id']=van_name
				if(bt1.split('_')[-1]=='Rented'):
					b1=bt1.split('_')[0]
					b1typ='Rented'
				else:
					b1=bt1
					b1typ='Owned'
				if(bt2.split('_')[-1]=='Rented'):
					b2=bt2.split('_')[0]
					b2typ='Rented'
				else:
					b2=bt2
					b2typ='Owned'
				target_wgt=df[df['van_id']==van_name]['weights'].sum()
				s=0
				sub_df=pd.DataFrame()
				for i,r in df[df['van_id']==van_name].iterrows():
					s=s+r['weights']
					if(s<van_weight_dict[b1]):
						r['van_id']=bt1
					else:
						r['van_id']=bt2
					sub_df=pd.concat([sub_df,pd.DataFrame(r).T])
				return pd.concat([sub_df,comb_df])
			
			def all_beats_complied(df):
				if(len(new_tdf)>0):
					rdf=df[~df['del_type'].isin(new_tdf['del_type'].unique())]
					df=pd.concat([rdf,new_tdf])
				df=pd.concat([df,edf])
				all_beats=df['van_id'].unique()
				bcnt=0
				for b in all_beats:
					odf=df[df['van_id']!=b]
					workdf_cluster=df[df['van_id']==b]
					van1=workdf_cluster['van_id'].unique()[0] 
					if (van1.split('_')[-1]=='Rented'):
						van1=van1.split('_')[0]
					van1endtime=0
					tot_endtime=0
					if(van1 in van_multitrip_dict.keys()):
						if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
							if(van_cutoff_dict[van1]=='yes'):
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
							else:
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
							odf['van_id2']=odf['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
							for v in odf['van_id'].unique():
								if(len(odf[(odf['van_id']==v) & (odf['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
									tot_endtime=tot_endtime+odf[(odf['van_id']==v)]['endtime'].unique()[0]
					
						else:
							van1endtime=van_endtime_dict[van1]
					else:
						van1endtime=van_endtime_dict[van1]
						
					van1wgt=workdf_cluster['cum_weight'].unique()[0]
					van1tm=workdf_cluster['endtime'].unique()[0]
					van1vol=workdf_cluster['cum_volume'].unique()[0]
					van1nbills=workdf_cluster['num_bills'].unique()[0]
					if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and ((van1tm+tot_endtime<=van1endtime))):
						bcnt=bcnt+1
				return (bcnt==len(all_beats))
			
			def find_non_compliance_beat(df):
				if(len(new_tdf)>0):
					rdf=df[~df['del_type'].isin(new_tdf['del_type'].unique())]
					df=pd.concat([rdf,new_tdf])
				df=pd.concat([df,edf])
				all_beats=df['van_id'].unique()
				for b in all_beats:
					odf=df[df['van_id']!=b]
					workdf_cluster=df[df['van_id']==b]
					van1=workdf_cluster['van_id'].unique()[0] 
					if (van1.split('_')[-1]=='Rented'):
						van1=van1.split('_')[0]
					van1wgt=workdf_cluster['cum_weight'].unique()[0]
					van1tm=workdf_cluster['endtime'].unique()[0]
					van1vol=workdf_cluster['cum_volume'].unique()[0]
					van1nbills=workdf_cluster['num_bills'].unique()[0]
					van1endtime=0
					tot_endtime=0
					if(van1 in van_multitrip_dict.keys()):
						if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
							if(van_cutoff_dict[van1]=='yes'):
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
							else:
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
							odf['van_id2']=odf['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
							for v in odf['van_id'].unique():
								if(len(odf[(odf['van_id']==v) & (odf['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
									tot_endtime=tot_endtime+odf[(odf['van_id']==v)]['endtime'].unique()[0]
					
						else:
							van1endtime=van_endtime_dict[van1]
					else:
						van1endtime=van_endtime_dict[van1]
						
					if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and ((van1tm+tot_endtime<=van1endtime))):
						continue
					else:
						if(van1wgt>van_weight_dict[van1]):
							print('can cons breached')
						if((van1vol>van_volume_dict[van1])):
							print('vol cons breached')
						if((van1nbills>van_bill_dict[van1])):
							print(van1nbills,van_bill_dict[van1])
							print('bills cons breached')
						if((van1tm>van_endtime_dict[van1])):
							print('endtime breached')
						return b
				return 'nil'
			
			def find_nearest_cluster(mindistout,bt,df,non_cluster_outs,cluster_outs):
				#print(len(non_cluster_outs))
				if(len(set(non_cluster_outs)-set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor',mindistout})<=0):
					return 'nil'
				outcluster_mindistout=pd.Series(distance_matrix[mindistout][distance_matrix[mindistout].index.isin(set(non_cluster_outs)-set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor',mindistout})]).idxmin()
				return list(set(df[df['path_x']==outcluster_mindistout]['van_id'].unique())-{bt})[0]
			
			def find_low_cost_empty_van(df):
				empty_vans=[]
				van_path=df.groupby(['van_id'])['path_x'].apply(list)
				for i in van_path.index:
					if(list(set(van_path[i]))==['Distributor']):
						empty_vans.append(i)
				mincost=1000000
				mincostvan='nil'
				for v in empty_vans:
					if((len(v.split('_'))>1) and (v.split('_')[-1]!='Rented')):
						if(van_cost_dict['_'.join(v.split('_')[:-1])+'_'+'1']<mincost):
							mincost=van_cost_dict['_'.join(v.split('_')[:-1])+'_'+'1']
							mincostvan=v
					elif((len(v.split('_'))>1) and (v.split('_')[-1]=='Rented')):
						if(van_cost_dict[v.split('_')[0]]<mincost):
							mincost=van_cost_dict[v.split('_')[0]]
							mincostvan=v
					else:
						if(van_cost_dict[v]<mincost):
							mincost=van_cost_dict[v]
							mincostvan=v
				return mincostvan   
			
			def check_limits_complied2(workdf_cluster,datfr):
				if(len(new_tdf)>0):
					rdf=datfr[~datfr['del_type'].isin(new_tdf['del_type'].unique())]
					datfr=pd.concat([rdf,new_tdf])
				datfr=pd.concat([datfr,edf])
				if(len(workdf_cluster)<=0):
					return True
				van1=workdf_cluster['van_id'].unique()[0] 
				datfr=datfr[~((datfr['van_id'].isin(workdf_cluster['van_id'].unique())) & (datfr['path_x'].isin(workdf_cluster['path_x'].unique())))].copy(deep=True)
				workdf_cluster=resequence(workdf_cluster)
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van1endtime=0
				tot_endtime=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								tot_endtime=tot_endtime+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
					
				if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
					return True
				else:
					return False     
			
			#if(non_constrained_beats_underuts):
			bt1=''
			bt2=''
			any_beats_subj_constraints_and_underuts=False
			subj_cons=False
			underuts=False
			for b in df['van_id'].unique():
				if(b.split('_')[-1]=='Rented'):
					b=b.split('_')[0]
					btyp='Rented'
				else:
					btyp='Owned'
				lr=False
				for l in set(input_data['lane_restrictions'].unique())-{10000}:
					if(van_weight_dict[b]>l):
					  lr=True
					  break
				if(btyp=='Owned'):
					if((str(van_plg_mapping[b])!='nan') or (str(van_areassign_dict[b])!='nan') or (str(van_orderexclu_dict[b])!='nan') or (lr)):
						subj_cons=True
				if(underutilized(b,btyp)):
					underuts=True
				if(underuts and subj_cons):
					any_beats_subj_constraints_and_underuts=True
					break
			
			al_vis_beat_pairs=[]    
			#if there is underutilisation then do this
			if(any_beats_subj_constraints_and_underuts):
				while((bt1!='nil') and (bt2!='nil')):
		
					print(bt1,bt2,df['weights'].sum())
					df_init_init=df.copy(deep=True)
					bt1,bt2,overlap_outs=find_max_overlap_beats(set(beats)-set(al_vis_beat_pairs),df)
					vis=[]
					
					'''
					while(((bt1,bt2) in al_vis_beat_pairs) or ((bt2,bt1) in al_vis_beat_pairs)):            
						bt1,bt2,overlap_outs=find_max_overlap_beats(list(set(beats)-{bt1,bt2}))
						if(len(al_vis_beat_pairs)==len(beats)*len(beats)-1):
							bt1='nil'
							break
					'''
					if(set(al_vis_beat_pairs)==len(df['van_id'].unique())):
						break
					#if they are not nil
					if((bt1=='nil') or (bt2=='nil')):
						continue
					l=[]
					for k in overlap_outs.keys():
						l.extend(overlap_outs[k])
						
					remianing_outs=set(df[df['van_id']==bt1]['path_x'])-{'Distributor'}-set(l)
					al_vis_beat_pairs.append(bt1)
					mixedall=0
					obeats=[]
					print(bt1)
					print(overlap_outs)
						
					for k in overlap_outs.keys():
						#if(outlets_mixing_allowed(df[df['van_id']==k],df[(df['van_id']==bt1) & (df['path_x'].isin(overlap_outs[k]))])):
							#mixedall=mixedall+1
						comb_df=resequence(pd.concat([df[df['van_id'].isin([k])],df[(df['van_id'].isin([bt1])) & (df['path_x'].isin(overlap_outs[k]))]]))            
						if(check_if_both_comb_possible(bt1,k,comb_df,overlap_outs[k])):
							#van_name,other_beat=find_apt_van(comb_df,bt1,k)
							#van_name=k
							#other_beat=bt1
							#obeats.append(other_beat)
							#print(van_name,other_beat,bt1)
							comb_df['van_id']=k
							#comb_df['del_type']=van_deltyp_dict[k]
							df=df[~df['van_id'].isin([k])]
							mixedall=mixedall+1
							df=df[~((df['van_id'].isin([bt1])) & (df['path_x'].isin(overlap_outs[k])))]
							df=pd.concat([df,comb_df])
								#df=pd.concat([df,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[other_beat], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':del_type,'time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
							
			#                else:
			#                    van_name=find_apt_van_others(comb_df,list(set(beats)-set([bt1,k])))
			#                    if(van_name!='nil'):
			#                       three_df=make_three_beats(van_name,bt1,bt2,comb_df) 
			#                       df=df[~df['van_id'].isin([bt1,bt2,van_name])]
			#                       df=pd.concat([df,three_df])
			#                       mixedall=mixedall+1
								   
						else:
							print('cannot merge-shud we roll back?',k)
		
					already_vis=[]
					
					for ob in set(obeats):
						df=pd.concat([df,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[ob], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':del_type,'del_typ2':'nil','time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
		
					
					if((mixedall==len(overlap_outs.keys())) and (set(obeats)=={bt1})):
						#treat rem_outlets also
						for ob in set(obeats):
							df=pd.concat([df,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[ob], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':del_type,'del_typ2':'nil','time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
						remianing_outs_copy=remianing_outs.copy()
						for ro in remianing_outs:
							non_cluster_outs=list(set(df[df['van_id']!=bt1]['path_x'].unique())-{'Distributor'})
							if(len(non_cluster_outs)<=0):
								continue
							clus=find_nearest_cluster(ro,bt1,df,non_cluster_outs,[])
							if(clus.split('_')[-1]=='Rented'):
								clus2=clus.split('_')[0]
							else:
								clus2=clus
							
							if(ro not in outlets_allowed_forvan[clus2]):
								print('outs not allowed for van')
								eligible_vs=[]
								for v in outlets_allowed_forvan.keys():
									if(v.split('_')[-1]=='Rented'):
										v2=v.split('_')[0]
									else:
										v2=v 
									if(ro in outlets_allowed_forvan[v2]):
										eligible_vs.append(v)
								non_cluster_outs=list(set(df[df['van_id'].isin(eligible_vs)]['path_x'].unique())-{'Distributor'})
								clus=find_nearest_cluster(ro,bt1,df,non_cluster_outs,[])
								if(clus.split('_')[-1]=='Rented'):
									clus2=clus.split('_')[0]
								else:
									clus2=clus 
								
							bt_bt=df[(df['van_id']==bt1) & (df['path_x'].isin(remianing_outs_copy))]
							#if(outlets_mixing_allowed(df[df['van_id']==clus],bt_bt[bt_bt['path_x']==ro])):
							new_bt=resequence(pd.concat([df[df['van_id']==clus],bt_bt[bt_bt['path_x']==ro]]))
							if(len(bt_bt[bt_bt['path_x']!=ro])>0):
								modi_bt=resequence(bt_bt[bt_bt['path_x']!=ro])
							else:
								modi_bt=pd.DataFrame()
								
							sub_df=pd.concat([new_bt,modi_bt])
							df=df[~((df['van_id']==bt1) & (df['path_x'].isin(remianing_outs_copy)))].copy(deep=True)
							df=pd.concat([df[~df['van_id'].isin([clus])],sub_df])
							already_vis.append(ro)
							remianing_outs_copy.remove(ro)
					
					
					df.drop_duplicates(subset =['van_id','path_x'], keep = 'first', inplace = True)
					
					already_vis=[]
					vis_beats=[]
					fl=False
					
					for v in df['van_id'].unique():
						if(len(df[df['van_id']==v])>0):
							print(v)
							print(df[df['van_id']==v]['cum_weight'].unique()[0])
							sdf=resequence(df[df['van_id']==v])
							print(sdf['cum_weight'].unique()[0])
							df=pd.concat([df[df['van_id']!=v],sdf])
					
					df_inittt=df.copy(deep=True)
					
					def find_non_compliance_beats(df):
						if(len(new_tdf)>0):
							rdf=df[~df['del_type'].isin(new_tdf['del_type'].unique())]
							df=pd.concat([rdf,new_tdf])
						df=pd.concat([df,edf])
						all_beats=df['van_id'].unique()
						non_compliance_beats=[]
						for b in all_beats:
							odf=df[df['van_id']!=b]
							workdf_cluster=df[df['van_id']==b]
							van1=workdf_cluster['van_id'].unique()[0] 
							if (van1.split('_')[-1]=='Rented'):
								van1=van1.split('_')[0]
							van1wgt=workdf_cluster['cum_weight'].unique()[0]
							van1tm=workdf_cluster['endtime'].unique()[0]
							van1vol=workdf_cluster['cum_volume'].unique()[0]
							van1nbills=workdf_cluster['num_bills'].unique()[0]
							van1endtime=0
							tot_endtime=0
							if(van1 in van_multitrip_dict.keys()):
								if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
									if(van_cutoff_dict[van1]=='yes'):
										van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
									else:
										van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
									odf['van_id2']=odf['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
									for v in odf['van_id'].unique():
										if(len(odf[(odf['van_id']==v) & (odf['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
											print(v)
											tot_endtime=tot_endtime+odf[(odf['van_id']==v)]['endtime'].unique()[0]
							
								else:
									van1endtime=van_endtime_dict[van1]
							else:
								van1endtime=van_endtime_dict[van1]
							if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
								continue
							else:
								non_compliance_beats.append(b)
						return non_compliance_beats
					
					def find_non_empty_vans(df):
						empty_vans=[]
						van_path=df.groupby(['van_id'])['path_x'].apply(list)
						for i in van_path.index:
							if(list(set(van_path[i]))==['Distributor']):
								empty_vans.append(i)
						non_empty_vans=list(set(df['van_id'].unique())-set(empty_vans)) 
						return non_empty_vans   
					
					non_compliance_beats=find_non_compliance_beats(pd.concat([tdf[tdf['del_type']!=del_type],df]))
					non_empty_vans=find_non_empty_vans(df)
					
					for btt in non_compliance_beats:
						assgn_emp_van=0
						df_init=df.copy(deep=True)
						vis_beats=[]
						non_empty_vans=find_non_empty_vans(df)
						prev_exchange_pairs=[]
						exchange_pairs=[]
						df_init2=df.copy(deep=True)
						clus=''
						prev_bt=''
						out_already_vis_in_beat={}
						for v in non_empty_vans:
							out_already_vis_in_beat[v]=[]
						while((not(all_beats_complied(pd.concat([tdf[tdf['del_type']!=del_type],df])))) and (not(len(set(vis_beats))==len(non_empty_vans)))):
							
							bt=find_non_compliance_beat(pd.concat([tdf[tdf['del_type']!=del_type],df]))
							if(bt not in out_already_vis_in_beat.keys()):
								out_already_vis_in_beat[bt]=[]
		#                    if(bt==clus):
		#                        clus_l=[]
		#                        for o in set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}:
		#                           non_cluster_outs=list(set(df[df['van_id']!=bt]['path_x'].unique())-{'Distributor'})                    
		#                           clus=find_nearest_cluster(o,bt,df,non_cluster_outs)
		#                           clus_l.append(clus)
		#                        if((len(set(clus_l))==1) and (clus_l[0]==prev_bt)):
		#                            df=df_init2.copy(deep=True)
		#                            bt=prev_bt
		#                    if(prevbt==bt):
		#                        df=df_init_init.copy(deep=True)
		#                        break
							print(bt)
							vis_beats.append(bt)
							already_vis=[]
							df_init2=df.copy(deep=True)
							#prev_exchange_pairs=exchange_pairs.copy()
							exchange_pairs=[]
							restricted_bts=[bt]
							while(not(check_limits_complied2(df[df['van_id']==bt],pd.concat([tdf[tdf['del_type']!=del_type],df])))):
								if(bt not in out_already_vis_in_beat.keys()):
										out_already_vis_in_beat[bt]=[]
								cluster_outs=list(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}-set(already_vis)-set(out_already_vis_in_beat[bt]))                
								non_cluster_outs=list(set(df[~df['van_id'].isin(restricted_bts)]['path_x'].unique())-{'Distributor'})                    
								if((len(cluster_outs)>0) and (len(non_cluster_outs)>0)):
									print('find outlet to be swapped')
									dm=distance_matrix[non_cluster_outs].T[cluster_outs].T 
									values=list(dm.min(axis=1))
									mindistout=list(dm.index)[values.index(min(values))]                            
									clus=find_nearest_cluster(mindistout,bt,df,non_cluster_outs,cluster_outs)
									if(clus=='nil'):
										already_vis.append(mindistout)
										continue
									if(clus.split('_')[-1]=='Rented'):
										clus2=clus.split('_')[0]
									else:
										clus2=clus
										
									if((mindistout in cluster_outs) and (mindistout in non_cluster_outs)):
										print('mindistout in both')
										if(not((check_limits_complied2(df[(df['van_id']==bt) & (df['path_x']!=mindistout)],pd.concat([tdf[tdf['del_type']!=del_type],df]))) and (df[(df['van_id'].isin([bt,clus])) & (df['path_x']==mindistout)]['weights'].sum()<van_weight_dict[clus2]))):
											already_vis.append(mindistout)
											continue
									if(mindistout not in outlets_allowed_forvan[clus2]):
										print('mindistout not allowed')
										already_vis.append(mindistout)
										continue
									#if(((mindistout,bt) in prev_exchange_pairs) and (clus==prev_bt)):
		#                            if((clus==prev_bt)):
		#                                print('same as prev exchanges/beat')
		#                                already_vis.append(mindistout)
		#                                continue
									
									clus_l=[]
									for o in set(df[df['van_id']==clus]['path_x'].unique())-{'Distributor'}:
									   ncos=list(set(df[df['van_id']!=clus]['path_x'].unique())-{'Distributor'})                    
									   c=find_nearest_cluster(o,clus,df,ncos,list(set(df[df['van_id']==clus]['path_x'].unique())-{'Distributor'}))
									   clus_l.append(c)
									if((len(set(clus_l))==1) and (clus_l[0]==bt)):
										if(clus.split('_')[-1]=='Rented'):
											ctyp='Rented'
										else:
											ctyp='Owned'
										if(not(check_limits_complied2(pd.concat([df[df['van_id']==clus],df[(df['van_id']==bt) & (df['path_x']==mindistout)]]),pd.concat([tdf[tdf['del_type']!=del_type],df])))):
											print('adding res beat',clus)
											restricted_bts.append(clus)
											continue
									
									print(mindistout,clus2) 
									exchange_pairs.append((mindistout,clus))
									if(clus not in out_already_vis_in_beat.keys()):
										out_already_vis_in_beat[clus]=[]
									out_already_vis_in_beat[clus].append(mindistout)
									bt_bt=df[df['van_id']==bt]
									new_bt=resequence(pd.concat([df[df['van_id']==clus],bt_bt[bt_bt['path_x']==mindistout]]))
									modi_bt=resequence(bt_bt[bt_bt['path_x']!=mindistout])
									sub_df=pd.concat([new_bt,modi_bt])
									df=pd.concat([df[~df['van_id'].isin([clus,bt])],sub_df])
									already_vis.append(mindistout)
								else:
									print('clstrout empty hence using a empty van')
									assgn_emp_van=assgn_emp_van+1
									df=df_init.copy(deep=True)
									bt=find_non_compliance_beat(pd.concat([tdf[tdf['del_type']!=del_type],df]))
									emp_van=find_low_cost_empty_van(df)
									#v_emp_van.append(emp_van)
									if(assgn_emp_van>6):
										emp_van='nil'
									if(emp_van!='nil'):
										sub_df=df[df['van_id']==emp_van]
										non_comply_beat=df[df['van_id']==bt].reset_index(drop=True)
										non_comply_beat_copy=non_comply_beat.copy(deep=True)
										
										if(emp_van.split('_')[-1]=='Rented'):
											emp_van2=emp_van.split('_')[0]
										else:
											emp_van2=emp_van
											
										for idx in reversed(non_comply_beat.index):
											if(non_comply_beat.iloc[idx]['path_x'] in outlets_allowed_forvan[emp_van2]):
												 sub_df=pd.concat([sub_df,pd.DataFrame(non_comply_beat.iloc[idx]).T])
												 non_comply_beat_copy=non_comply_beat_copy[non_comply_beat_copy.index!=idx].copy(deep=True)
												 sub_df=resequence(sub_df)
												 non_comply_beat_copy=resequence(non_comply_beat_copy)
											else:
												continue
											if(check_limits_complied2(non_comply_beat_copy,pd.concat([tdf[tdf['del_type']!=del_type],df]))):
												break
										df=df[~df['van_id'].isin([bt,emp_van])]
										df=pd.concat([df,sub_df])
										df=pd.concat([df,non_comply_beat_copy])
										df_init=df.copy(deep=True)
									else:
										print('ROLL_BACK')
										assgn_emp_van=0
										df=df_init_init.copy(deep=True)
		
							prev_bt=bt
							
						if((len(set(vis_beats))==len(non_empty_vans)) and (not(all_beats_complied(pd.concat([tdf[tdf['del_type']!=del_type],df]))))):
							print('ROLL_BACK')
							df=df_init_init.copy(deep=True)
							
		
			len(df)
			df['day']='day'
			all_outlets=list(set(df['path_x'].unique())-{'Distributor'})
			
				
			nn_df=pd.DataFrame(columns=['outlet','outlet_neighbour','nghbr_dist'])
			for o in all_outlets:
				if(len(all_outlets)==1):
					nn_df=pd.DataFrame()
					break
				eligible_ngbrs_of_outlet=[]
				for v in df[df['path_x']==o]['van_id'].unique():
					if(v.split('_')[-1]=='Rented'):
						v2=v.split('_')[0]
					else:
						v2=v
					eligible_ngbrs_of_outlet.extend(outlets_allowed_forvan[v2])
				print(v2,outlets_allowed_forvan[v2],outlets_allowed_forvan)
				eligible_ngbrs_of_outlet=list(set(all_outlets).intersection(set(eligible_ngbrs_of_outlet)))
				print(eligible_ngbrs_of_outlet)
				dist = pd.Series(distance_matrix[o][distance_matrix[o].index.isin(set(eligible_ngbrs_of_outlet)-{'Distributor',o})])    
				nearest_index = dist.idxmin()
				nn_df=pd.concat([nn_df,pd.DataFrame({'outlet':[o],'outlet_neighbour':[nearest_index],'nghbr_dist':[distance_matrix[o][nearest_index]]})])
				
			cluster_of=df.groupby(['path_x'])['van_id'].apply(list).to_dict()  
			
			if(len(all_outlets)>1):
				nn_df=nn_df.sort_values(by=['nghbr_dist'], ascending=True).copy(deep=True)
			
			
			def beatouts_exchangable(out,nout):
				return True
			
			
			
			def push_to_cluster(out,workdf_cluster,workdf_ncluster):
				workdf_cluster=pd.concat([workdf_cluster,workdf_ncluster[workdf_ncluster['path_x']==out]])
				workdf_ncluster=workdf_ncluster[~workdf_ncluster['path_x'].isin([out])]
				workdf_cluster=resequence(workdf_cluster)
				workdf_ncluster=resequence(workdf_ncluster) 
				return workdf_cluster,workdf_ncluster
			
				
			def check_limits_complied(workdf_cluster,workdf_ncluster,datfr):
				if(len(new_tdf)>0):
					rdf=datfr[~datfr['del_type'].isin(new_tdf['del_type'].unique())]
					datfr=pd.concat([rdf,new_tdf])
				datfr=pd.concat([datfr,edf])
				van1=workdf_cluster['van_id'].unique()[0]
				van2=workdf_ncluster['van_id'].unique()[0]
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				if (van2.split('_')[-1]=='Rented'):
					van2=van2.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van2wgt=workdf_ncluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van2tm=workdf_ncluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van2vol=workdf_ncluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van2nbills=workdf_ncluster['num_bills'].unique()[0]
				van1endtime=0
				van2endtime=0
				van1_tot_time=0
				van2_tot_time=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								van1_tot_time=van1_tot_time+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
				
				if(van2 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van2]=='yes') and (not((van_cutoff_dict[van2]=='yes') and (van_endtime_dict[van2]==480)))):
						if(van_cutoff_dict[van2]=='yes'):
							van2endtime=van_endtime_dict['_'.join(workdf_ncluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van2endtime=van_endtime_dict['_'.join(workdf_ncluster['van_id'].unique()[0].split('_')[:-1])+'_1']
		
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_ncluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								van2_tot_time=van2_tot_time+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]        
					else:
						van2endtime=van_endtime_dict[van2]
				else:
					van2endtime=van_endtime_dict[van2]
				
				if((van1wgt<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van2wgt<=van_weight_dict[van2]) and (van2vol<=van_volume_dict[van2]) and (van2nbills<=van_bill_dict[van2]) and (van1tm+van1_tot_time<=van1endtime) and (van2tm+van2_tot_time<=van2endtime)):
					return True
				else:
					return False
			
			def check_limits_complied2(workdf_cluster,datfr):
				if(len(new_tdf)>0):
					rdf=datfr[~datfr['del_type'].isin(new_tdf['del_type'].unique())]
					datfr=pd.concat([rdf,new_tdf])
				datfr=pd.concat([datfr,edf])
				if(len(workdf_cluster)<=0):
					return True
				van1=workdf_cluster['van_id'].unique()[0] 
				datfr=datfr[~((datfr['van_id'].isin(workdf_cluster['van_id'].unique())) & (datfr['path_x'].isin(workdf_cluster['path_x'].unique())))].copy(deep=True)
				workdf_cluster=resequence(workdf_cluster)
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van1endtime=0
				tot_endtime=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
		
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								tot_endtime=tot_endtime+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
					
				if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
					return True
				else:
					return False      
				
			nn_df=nn_df.reset_index(drop=True)  
			  
			for i,r in nn_df.iterrows():
				print('enter for')
				out=r['outlet']
				nout=r['outlet_neighbour']
				ndist=r['nghbr_dist']
				cluster_of=df.groupby(['path_x'])['van_id'].apply(list).to_dict()  
				#if in different cluster
				if(len(set(cluster_of[out]).intersection(set(cluster_of[nout])))==0):
					#print(i)
				  if(beatouts_exchangable(out,nout)):
					  df_init=df.copy(deep=True)
					  newdf=df[~df['van_id'].isin([cluster_of[out][0],cluster_of[nout][0]])]
					  workdf=df[df['van_id'].isin([cluster_of[out][0],cluster_of[nout][0]])]
					  workdf_cluster=workdf[workdf['van_id'].isin(cluster_of[out])]
					  workdf_ncluster=workdf[workdf['van_id'].isin(cluster_of[nout])]
					  tot_outs_cluster=len(set(workdf_ncluster['path_x'])-{'Distributor'})
					  workdf_ncluster=pd.concat([workdf_ncluster,workdf_cluster[workdf_cluster['path_x']==out]])
					  v=cluster_of[nout][0]
					  if(v.split('_')[-1]=='Rented'):
						  v2=v.split('_')[0]
					  else:
						  v2=v
					 
					  if(out not in outlets_allowed_forvan[v2]):
						  continue
					  workdf_cluster=workdf_cluster[~workdf_cluster['path_x'].isin([out])]
					  workdf_cluster=resequence(workdf_cluster)
					  workdf_ncluster=resequence(workdf_ncluster)
					  count=0
					  rev=False
					  already_vis=[]
					  while((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count!=tot_outs_cluster)):
    						 count=count+1
    						 print('enter while',count)
    						 cluster_outs=list(set(workdf_cluster['path_x'])-{'Distributor'})
    						 ncluster_outs=list(set(workdf_ncluster['path_x'])-{'Distributor'}-set(already_vis))
    						 dm=distance_matrix[cluster_outs].T[ncluster_outs].T 
    						 if((len(ncluster_outs)==0) or (len(cluster_outs)==0)):
    							 rev=True
    							 break
    						 values=list(dm.min(axis=1))
    						 if(min(values)>ndist):
    							 rev=True
    							 break
    						 #mindistout=list(dm.idxmin(axis=1))[values.index(min(values))]
    						 mindistout=list(dm.index)[values.index(min(values))]
    						 print('ncluster',mindistout in ncluster_outs)
    						 print('cluster',mindistout in cluster_outs)
    						 print('-----------------')
    						 v=workdf_cluster['van_id'].unique()[0]
    						 if(v.split('_')[-1]=='Rented'):
    							  v2=v.split('_')[0]
    						 else:
    							  v2=v
    						 if(not(mindistout in outlets_allowed_forvan[v2])):# and (not(outlets_mixing_allowed(workdf_cluster,workdf_ncluster[workdf_ncluster['path_x']==mindistout])))):
    							 already_vis.append(mindistout)
    							 continue
    						 wgt_to_add=float(workdf_ncluster[workdf_ncluster['path_x']==mindistout]['weights'])
    						 print(workdf_cluster['van_id'].unique()[0])
    						 print(workdf_cluster['weights'].sum())
    						 if workdf_cluster['van_id'].unique()[0].split('_')[-1]=='Rented':
    							 wgt_can_handle=(van_weight_dict[(workdf_cluster['van_id'].unique()[0]).split('_')[0]])-(workdf_cluster['weights'].sum())
    						 else:
    							 wgt_can_handle=(van_weight_dict[workdf_cluster['van_id'].unique()[0]])-(workdf_cluster['weights'].sum())
    		#                 wgt_can_handle=(van_weight_dict[workdf_cluster['van_id'].unique()[0]])-(workdf_cluster['weights'].sum())
    						 print(wgt_to_add,wgt_can_handle)
    						 if(wgt_to_add>wgt_can_handle):
    							 
    							 already_vis.append(mindistout)
    							 continue
    						 print(workdf_cluster['van_id'].unique(),workdf_ncluster['van_id'].unique(),len(workdf_cluster),len(workdf_ncluster),workdf_cluster['weights'].sum(),workdf_ncluster['weights'].sum())
    						 #workdf_cluster,workdf_ncluster=push_to_cluster(mindistout,workdf_cluster,workdf_ncluster)
    						 workdf_cluster_init=workdf_cluster.copy(deep=True)
    						 workdf_ncluster_init=workdf_ncluster.copy(deep=True)
    						 workdf_cluster,workdf_ncluster=push_to_cluster(mindistout,workdf_cluster,workdf_ncluster)
    						 print(workdf_cluster['van_id'].unique(),workdf_ncluster['van_id'].unique(),len(workdf_cluster),len(workdf_ncluster),workdf_cluster['weights'].sum(),workdf_ncluster['weights'].sum())
    						 already_vis.append(mindistout)
    						 if(not(check_limits_complied2(workdf_cluster,pd.concat([tdf[tdf['del_type']!=del_type],df_init])))):
        							workdf_cluster=workdf_cluster_init.copy(deep=True)
        							workdf_ncluster=workdf_ncluster_init.copy(deep=True) 
					  if((rev) or ((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count==tot_outs_cluster))):
						  print('unchanged')
						  df=df_init.copy(deep=True)
					  else:
						  print('changed')
						  #if(not(check_limits_complied2(pd.concat([workdf_cluster,workdf_ncluster])))):
						  df=pd.concat([newdf,workdf_cluster]) 
						  df=pd.concat([df,workdf_ncluster]) 
			#              else:
			#                  workdf_cluster=pd.concat([workdf_cluster,workdf_ncluster])
			#                  workdf_cluster=resequence(workdf_cluster)
			#                  df=pd.concat([newdf,workdf_cluster]) 
			
			len(df)
			#df.to_csv('fnw_444230_7mar_nn.csv')
			al_vis_bt=[]
			
			def find_nearest_cluster(rem_df,new_df):
			   clstrouts=list(rem_df['path_x'].unique()) #+list(new_df[new_df['van_id'].isin(rem_df['van_id'])]['path_x'].unique()) 
			   nclstrouts=list(new_df[~new_df['van_id'].isin(list(rem_df['van_id'])+al_vis_bt)]['path_x'].unique()) 
			   #dm=distance_matrix[clstrouts].T[nclstrouts].T 
			   #values=list(dm.min(axis=1))
			   mindistoutclus=[]
			   if(len(set(nclstrouts)-{'Distributor'})==0):
				   
				   return 'nil'
			   for co in clstrouts:
				   mindistout=pd.Series(distance_matrix[co][distance_matrix[co].index.isin(set(nclstrouts)-{'Distributor'})]).idxmin()
				   mindistoutclus.append(new_df[new_df['path_x']==mindistout]['van_id'].unique()[0])
			   mindistoutclus_cnt={}
			   for c in set(mindistoutclus):
				   mindistoutclus_cnt[c]=mindistoutclus.count(c)
				   
				   '''
				min_value1=min(values)
			   mindistout1=list(dm.index)[values.index(min_value1)]
			   clstrouts=list(new_df[new_df['van_id'].isin(rem_df['van_id'])]['path_x'].unique()) 
			   nclstrouts=list(new_df[~new_df['van_id'].isin(rem_df['van_id'])]['path_x'].unique()) 
			   dm=distance_matrix[clstrouts].T[nclstrouts].T 
			   values=list(dm.min(axis=1))
			   min_value2=min(values)
			   mindistout2=list(dm.index)[values.index(min_value2)]
			   if(min_value1>min_value2):
				   mindistout=mindistout2
			   else:
				   mindistout=mindistout1'''
			   
			   return max(mindistoutclus_cnt, key=mindistoutclus_cnt.get)
			
			
				
				
			def find_neighbour_outside(n_bts_outs,ot):
				dist = distance_matrix[ot][distance_matrix[ot].index.isin(list(n_bts_outs))]
				dist = pd.Series(dist)
				nearest_index = dist.idxmin() 
				return nearest_index
				
				
			def find_sim_beats(bt,df):
				return list(df[~(df['van_id']==bt)]['van_id'].unique()) 
		
			def check_if_both_comb_possible(bt1,bt2,comb_df):
				c1=0
				if(bt1.split('_')[-1]=='Rented'):
					bt1=bt1.split('_')[0]
				if(bt2.split('_')[-1]=='Rented'):
					bt2=bt2.split('_')[0]
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
		#        if((len(set(outs).intersection(set(exclusive_outlets)))!=len(outs)) or (len(set(outs).intersection(set(exclusive_outlets)))!=0)):
		#            return False
				for o in outs:
					if(o in outlets_allowed_forvan[bt1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[bt2]):
						c2=c2+1
						
				if((c1==len(outs))):
					return True
				else:
					return False       
				
			def dist_jump_abnormal(bt,sim_bts,df):
				bt_df=df[df['van_id']==bt]
				nbt_df=df[~(df['van_id']==bt)]
				bt_outs=list(bt_df['path_x'].unique())
				n_bts_outs=list(nbt_df['path_x'].unique())
				checked=[]
				flag=False
				rem_df=pd.DataFrame()
				inter_dist=[]
				intra_dist=[]
				for i in range(len(bt_outs)-1,0,-1):
					ot=bt_outs[i]
					if(ot=='Distributor'):
					  checked.append(ot)
					  continue
					not_bt=bt_outs[i-1]
					
					not_nbt=find_neighbour_outside(n_bts_outs,ot)
					print(ot,not_bt,not_nbt)
					checked.append(ot)
					intra_dist.append(distance_matrix[ot][not_bt])
					inter_dist.append(distance_matrix[ot][not_nbt])
				
				if((len(intra_dist[:-1])>0) and (max(intra_dist[:-1])>2*inter_dist[intra_dist.index(max(intra_dist[:-1]))])):
					flag=True
			#        if(len(bt_outs)-1-intra_dist.index(max(intra_dist[:-1]))>intra_dist.index(max(intra_dist[:-1]))):
			#            rem_df=bt_df[bt_df['path_x'].isin(bt_outs[1:intra_dist.index(max(intra_dist[:-1]))+1])]
			#            bt_df=bt_df[bt_df['path_x'].isin(bt_outs[intra_dist.index(max(intra_dist[:-1]))+1:]+['Distributor'])]
			#        else:
					rem_df=bt_df[bt_df['path_x'].isin(bt_outs[::-1][:intra_dist.index(max(intra_dist[:-1]))+1])]
					bt_df=bt_df[bt_df['path_x'].isin(bt_outs[::-1][intra_dist.index(max(intra_dist[:-1]))+1:])]
				print(len(rem_df),len(bt_df))
				return flag,pd.concat([nbt_df,bt_df]),rem_df     
					
				
			def outlet_shift_across_cluster(rem_df,new_df,df_init):
				cluster=find_nearest_cluster(rem_df,new_df)
				if(cluster=='nil'):
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
					 return df,dont_check_again 
				 
				al_vis_bt.append(cluster)
				workdf_cluster=new_df[new_df['van_id'].isin(list(rem_df['van_id'].unique()))]
				workdf_ncluster=new_df[new_df['van_id'].isin([cluster])]
				#create_two_beats(workdf_cluster,workdf_cluster,rem_df)
				newdf=new_df[~new_df['van_id'].isin(list(rem_df['van_id'].unique())+[cluster])]
				print(len(workdf_cluster),len(workdf_ncluster),len(rem_df))
				workdf_ncluster=pd.concat([workdf_ncluster,rem_df])
				print(len(workdf_cluster),len(workdf_ncluster),len(rem_df))
				workdf_cluster=resequence(workdf_cluster)
				workdf_ncluster=resequence(workdf_ncluster)
				if(not(check_if_both_comb_possible(cluster,rem_df['van_id'].unique()[0],workdf_ncluster))):
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
					 return df,dont_check_again 
				tot_outs_cluster=len(set(workdf_ncluster['path_x'])-{'Distributor'})
				rev=False
				already_vis=[]
				count=0
				tot_wgt_added=rem_df['weights'].sum()
				wgt_to_add=0
				rem_wgt=tot_wgt_added
				dont_check_again=False
				while((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count!=tot_outs_cluster)): 
					 print('--------------------') 
					 count=count+1
					 cluster_outs=list(set(workdf_cluster['path_x'])-{'Distributor'})
					 ncluster_outs=list(set(workdf_ncluster['path_x'])-{'Distributor'}-set(already_vis))
					 dm=distance_matrix[cluster_outs].T[ncluster_outs].T 
					 if((len(ncluster_outs)==0) or (len(cluster_outs)==0)):
						 print('rev True')
						 rev=True
						 break
					 values=list(dm.min(axis=1))
					 mindistout=list(dm.index)[values.index(min(values))]
		#             if(mindistout in rem_df['path_x'].unique()):
		#                 print('min dist out was in work cluster')
		#                 already_vis.append(mindistout)
		#                 continue
					 v=workdf_cluster['van_id'].unique()[0]
					 if(v.split('_')[-1]=='Rented'):
						  v2=v.split('_')[0]
					 else:
						  v2=v
					 if(not(mindistout in outlets_allowed_forvan[v2])):
						 print('min dist out not in outlets_allowed_forvan')
						 already_vis.append(mindistout)
						 continue
					 wgt_to_add=float(workdf_ncluster[workdf_ncluster['path_x']==mindistout]['weights'])
					 if workdf_cluster['van_id'].unique()[0].split('_')[-1]=='Rented':
						 wgt_can_handle=(van_weight_dict[(workdf_cluster['van_id'].unique()[0]).split('_')[0]])-(workdf_cluster['weights'].sum())
					 else:
						 wgt_can_handle=(van_weight_dict[workdf_cluster['van_id'].unique()[0]])-(workdf_cluster['weights'].sum())
					 if(wgt_to_add>wgt_can_handle):
						 print('wgt_to_add>wgt_can_handle')
						 already_vis.append(mindistout)
						 continue
					 workdf_cluster_init=workdf_cluster.copy(deep=True)
					 workdf_ncluster_init=workdf_ncluster.copy(deep=True)
					 workdf_cluster,workdf_ncluster=push_to_cluster(mindistout,workdf_cluster,workdf_ncluster)
					 already_vis.append(mindistout)
					 if(not(check_limits_complied2(workdf_cluster,pd.concat([tdf[tdf['del_type']!=del_type],df_init])))):
    						print('workdf_cluster not cpmplying') 
    						workdf_cluster=workdf_cluster_init.copy(deep=True)
    						workdf_ncluster=workdf_ncluster_init.copy(deep=True) 
				
				if((rev) or ((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count==tot_outs_cluster))):
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
				else:
					 print('chnaged')
					 #if(not(check_limits_complied2(resequence(pd.concat([workdf_cluster,workdf_ncluster]))))):
					 #if(not(check_limits_complied2(pd.concat([workdf_cluster,workdf_ncluster])))):
					 df=pd.concat([newdf,workdf_cluster]) 
					 df=pd.concat([df,workdf_ncluster]) 
		#             else:
		#                  print(workdf_cluster['van_id'].unique(),workdf_ncluster['van_id'].unique())
		#                  workdf_cluster=pd.concat([workdf_cluster,workdf_ncluster])
		#                  workdf_cluster=resequence(workdf_cluster)
		#                  df=pd.concat([newdf,workdf_cluster]) 
					
				return df,dont_check_again    
			
				 
						 
			df_copy=df.copy(deep=True)
			vans=owned_van_order_tofill+rental_van_order_tofill
			removed_vans=[]
			van_path=df.groupby(['van_id'])['path_x'].apply(list)
			for i in van_path.index:
				if(list(set(van_path[i]))==['Distributor']):
					removed_vans.append(i)
			al_vis_bt=[]        
			if(len(set(df_copy['van_id'].unique())-set(removed_vans))>1):
				for bt in vans[::-1]:
					if((bt in df_copy['van_id'].unique()) or (bt in [b.split('_')[0] for b in df_copy['van_id'].unique()])):
						al_vis_bt=[]
						while True:
							bt_df=df[df['van_id']==bt]
							if(set(bt_df['path_x'])=={'Distributor'}):
								break
							nbt_df=df[~(df['van_id']==bt)]
							sim_bts=find_sim_beats(bt,df)
							df_init=df.copy(deep=True)
							flag=False
							flag,new_df,rem_df=dist_jump_abnormal(bt,sim_bts,df)
							print(flag)
							if(flag):
								df,dont_check_again=outlet_shift_across_cluster(rem_df,new_df,df_init) 
								if(dont_check_again):
									break
							else:
								df=new_df.copy(deep=True)
								break
						
					
			len(df)
				
			new_tdf=pd.concat([new_tdf,df])
		
		
		old_tdf=tdf.copy(deep=True)    
		gdf=new_tdf.groupby(['van_id'])['path_x'].apply(list)
		removed_vans=[]
		for v in new_tdf['van_id'].unique():
		  if(list(set(gdf[v]))==['Distributor']):
			  new_tdf=new_tdf[new_tdf['van_id']!=v]  
			  removed_vans.append(v)
		
		
		  
		tdf=new_tdf.copy(deep=True) 
		
		def find_rental_van_replacement(van,options):
			eligible_ops=[]
			for vn in options:
				if((van_multitrip_dict[vn]=='yes') and (van_cutoff_dict[vn]=='yes') and (int(vn.split('_')[-1])==3)):
					continue
				if(vn.split('_')[-1]=='Rented'):
					vn2=vn.split('_')[0]
				else:
					vn2=vn
				workdf_cluster=tdf[tdf['van_id']==van]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				lr=False
				if(len(set(set(list(workdf_cluster['path_x']))-{'Distributor'}).intersection(outlets_allowed_forvan[vn2]))==len(set(set(list(workdf_cluster['path_x']))-{'Distributor'}))):
					lr=True
				if((van1wgt<van_weight_dict[vn2]) and (van1vol<van_volume_dict[vn2]) and (van1nbills<van_bill_dict[vn2]) and (van1tm<van_endtime_dict[vn2]) and (lr)):
					eligible_ops.append(vn)
					
			closest_van_name='nil'
			closest_diff=100000
			for op in eligible_ops:
				if(op.split('_')[-1]=='Rented'):
					op2=vn.split('_')[0]
				else:
					op2=vn
				diff=van_weight_dict[op2]-workdf_cluster['weights'].sum()
				if(diff<closest_diff):
					closest_diff=diff
					closest_van_name=op
			print(closest_van_name,van)        
			return closest_van_name
				
					
		van_cap_dict={}    
		for v in tdf['van_id'].unique():
			if(v.split('_')[-1]=='Rented'):
				v2= v.split('_')[0]
			else:
				v2=v
			van_cap_dict[v]=van_weight_dict[v2]
			
			
		for van in {k: v for k, v in sorted(van_cap_dict.items(), key=lambda item: item[1],reverse=True)}.keys():
			if(van.split('_')[-1]=='Rented'):
			   vanname=find_rental_van_replacement(van,set(owned_van_order_tofill)-set(tdf['van_id'].unique())) 
			   if(vanname!='nil'):
				   subdf=tdf[tdf['van_id']==van].copy(deep=True)
				   subdf['van_id']=vanname
				   tdf=pd.concat([tdf[tdf['van_id']!=van],subdf])
				   
		new_tdf=tdf.copy(deep=True)
		#add the empty owned vans
		for v in owned_van_order_tofill:
			if(v not in new_tdf['van_id'].unique()):
				if((van_multitrip_dict[v]=='yes') and (int(v.split('_')[-1])!=3)):
					if((int(v.split('_')[-1])!=3) and ('_'.join(v.split('_')[:-1])+'_3' not in new_tdf['van_id'].unique())):
						print(v)
						new_tdf=pd.concat([new_tdf,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[v], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':'','time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
				elif(van_multitrip_dict[v]=='no'):
					print(v)
					new_tdf=pd.concat([new_tdf,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[v], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':'','time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
			   
		#new_tdf.to_csv('svr_2ndfeb_wc_merge_cc_test_2.csv')      
		#tdf=pd.read_csv('tvishi_16march_wc_merge_cc_test.csv')
		
		tdf=new_tdf.copy(deep=True)
		
		def underutilized(bt,btyp):
				if(btyp=='Rented'):
					b=bt.split('_')[0]
				else:
					b=bt
				if(tdf[tdf['van_id']==bt]['weights'].sum()<0.68*van_weight_dict[b]):
					return True
				else:
					return False
				
		across_cluster_merge=False
		under_uts_dict={}
		under_uts_deltyp={}
		
		for vn in tdf['van_id'].unique():    
			if(vn.split('_')[-1]=='Rented'):
				vn2=vn.split('_')[0]
				btyp='Rented'
			else:
				vn2=vn
				btyp='Owned'
			if(underutilized(vn,btyp)):
				print(tdf[tdf['van_id']==vn]['weights'].sum())        
				under_uts_dict[vn]=van_weight_dict[vn2]-tdf[tdf['van_id']==vn]['weights'].sum()
				under_uts_deltyp[vn]=tdf[tdf['van_id']==vn]['del_type'].unique()[0]
				across_cluster_merge=True       
		
		under_uts_dict2={k: v for k, v in sorted(under_uts_dict.items(), key=lambda item: item[1],reverse=True)}.keys()    
		new_tdf=pd.DataFrame()
		
		subdf=tdf[tdf['del_type'].isin(list(set([str(v) for v in van_orderexclu_dict.values()])-{'nan'}))].copy(deep=True)
		tdf=tdf[~tdf['del_type'].isin(list(set([str(v) for v in van_orderexclu_dict.values()])-{'nan'}))].copy(deep=True)
		
		
		if((across_cluster_merge) and (len(tdf['del_type'].unique())>1)):  
			df=tdf.copy(deep=True)
			beats=df['van_id'].unique()
			df['day']='day'
			df['del_typ2']='nil'
			van_deltyp_dict=dict(list(zip(df['van_id'],df['del_type'])))
			def resequence(clusterdf):
				w_dict=clusterdf.groupby(['path_x'])['weights'].sum().to_dict()
				v_dict=clusterdf.groupby(['path_x'])['volumes'].sum().to_dict()
				b_dict=clusterdf.groupby(['path_x'])['bill_numbers'].apply(list).to_dict()
				for k in b_dict.keys():
					l=[]
					l.extend(b_dict[k])
					b_dict[k]=l            
					
				for k in b_dict.keys():
					l=[]
					for sub in b_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9_]+', ',', str(sub))).split(',')) 
					b_dict[k]=[i for i in l if i!='']
					
				bp_dict=clusterdf.groupby(['path_x'])['Basepack'].apply(list).to_dict()
				for k in bp_dict.keys():
					l=[]
					l.extend(bp_dict[k])
					bp_dict[k]=l
				for k in bp_dict.keys():
					l=[]
					for sub in bp_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9_]+', ',', str(sub))).split(',')) 
					bp_dict[k]=[i for i in l if i!=''] 
				c_dict=clusterdf.groupby(['path_x'])['channel'].apply(list).to_dict()    
				p_dict=clusterdf.groupby(['path_x'])['plg'].apply(list).to_dict()
				
				for k in c_dict.keys():
					l=[]
					l.extend(c_dict[k])
					c_dict[k]=l
				for k in c_dict.keys():
					l=[]
					for sub in c_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9]+', ',', str(sub))).split(',')) 
					c_dict[k]=[i for i in l if i!=''] 
				
				for k in p_dict.keys():
					l=[]
					l.extend(p_dict[k])
					p_dict[k]=l
				for k in p_dict.keys():
					l=[]
					for sub in p_dict[k]:
						#print(sub)
						l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(sub))).split(',')) 
					p_dict[k]=[i for i in l if i!=''] 
					
				#b_dict=dict(zip(clusterdf['path_x'],clusterdf['bill_numbers']))
				#bp_dict=dict(zip(clusterdf['path_x'],clusterdf['Basepack']))
				lat_dict=dict(zip(clusterdf['path_x'],clusterdf['outlet_latitude']))
				long_dict=dict(zip(clusterdf['path_x'],clusterdf['outlet_longitude']))
				lat_dict['Distributor']=input_data['rs_latitude'].unique()[0]
				long_dict['Distributor']=input_data['rs_longitude'].unique()[0]
				
				pathseq=['Distributor']
				wgtseq=[0]
				volseq=[0]
				timeseq=[0]
				billseq=[[]]
				bpseq=[[]]
				chseq=[[]]
				plgseq=[[]]
				deltyp=list(clusterdf['del_type'].unique())[0]
				deltyp2=list(clusterdf['del_typ2'].unique())[0]
				numbills=0
				cum_weight=0
				cum_volume=0
				van_id=list(clusterdf['van_id'].unique())[0]
				van_id_init=van_id
				day=list(clusterdf['day'].unique())[0]
				latseq=[lat_dict['Distributor']]
				longseq=[long_dict['Distributor']]
				mids=list(set(w_dict.keys())-{'Distributor'})
				cnt=0
				
				while(cnt!=len(mids)):
					cnt=cnt+1
					end_time_seq=0
					dist = distance_matrix[pathseq[-1]][distance_matrix[pathseq[-1]].index.isin(list(set(mids)-set(pathseq)))]
					if(len(dist)==0):
						break
					dist = pd.Series(dist)
					nearest_index = dist.idxmin()
					pathseq.append(nearest_index)
					wgtseq.append(w_dict[nearest_index])
					volseq.append(v_dict[nearest_index])
					distance = dist[nearest_index]
					if (van_id.split('_')[-1]=='Rented'):
						van_id=van_id.split('_')[0]
					
					travel_time = math.ceil(distance * (1/van_speed_dict[van_id]))
					end_time_seq = end_time_seq + travel_time
					service_time=int(service_time_details[(w_dict[nearest_index]>=service_time_details['load_range_from_kgs'].astype(float)) & (w_dict[nearest_index]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
					end_time_seq=end_time_seq+ service_time
					timeseq.append(travel_time+service_time)
					billseq.append(b_dict[nearest_index])
					bpseq.append(bp_dict[nearest_index])
					chseq.append(c_dict[nearest_index])
					plgseq.append(p_dict[nearest_index])
					numbills=numbills+len(b_dict[nearest_index])
					cum_weight=cum_weight+w_dict[nearest_index]
					cum_volume=cum_volume+v_dict[nearest_index]
					latseq.append(lat_dict[nearest_index])
					longseq.append(long_dict[nearest_index])
					
				df_=pd.DataFrame()
				pathseq.append('Distributor')
				wgtseq.append(0)
				volseq.append(0)
				if van_id.split('_')[-1]=='Rented':
					timeseq.append(math.ceil(distance_matrix[pathseq[-2]]['Distributor'] * (1/van_speed_dict[van_id.split('_')[0]])))
				else:
					timeseq.append(math.ceil(distance_matrix[pathseq[-2]]['Distributor'] * (1/van_speed_dict[van_id])))
				billseq.append([''])
				bpseq.append([''])
				chseq.append([''])
				plgseq.append([''])
				latseq.append(lat_dict['Distributor'])
				longseq.append(long_dict['Distributor'])
				
				return pd.concat([df_,pd.DataFrame({'path_x':pathseq, 'endtime':[sum(timeseq)]*len(pathseq), 'num_bills':[numbills]*len(pathseq), 'cum_weight':[cum_weight]*len(pathseq), 'van_id':[van_id_init]*len(pathseq), 'weights':wgtseq,'cum_volume':[cum_volume]*len(pathseq), 'volumes':volseq, 'del_type':deltyp,'del_typ2':deltyp2,'time':timeseq, 'bill_numbers':billseq, 'Basepack':bpseq,'partyhll_code':pathseq, 'outlet_latitude':latseq, 'outlet_longitude':longseq, 'day':[day]*len(pathseq),'plg':plgseq,'channel':chseq})])
				
			def underutilized(bt,btyp):
				if(btyp=='Rented'):
					b=bt.split('_')[0]
				else:
					b=bt
				if(df[df['van_id']==bt]['weights'].sum()<0.68*van_weight_dict[b]):
					return True
				else:
					return False
		
		
			def find_max_overlap_beats(beats,df): 
			
				max_overlap_percent=0
				max_overlap_bt1='nil'
				max_overlap_bt2='nil'
				overlap_outlets={}
				if(len(beats)==1):
					return max_overlap_bt1,max_overlap_bt2,{}
				for bt in beats:
					if((list(df[df['van_id']!=bt]['path_x'].unique())==['Distributor'])): 
						return max_overlap_bt1,max_overlap_bt2,{}
					if((list(df[df['van_id']==bt]['path_x'].unique())==['Distributor'])):
						continue
					overlap_outs_cnt=0
					overlap_clus=[]
					overlap_outs=[]
					for o in set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}:
						other_outs=list(set(df[df['van_id']!=bt]['path_x'].unique())-{'Distributor',o})
						beat_outs=list(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor',o})
						nbr_outside_beat=pd.Series(distance_matrix[o][distance_matrix[o].index.isin(other_outs)]).idxmin()
						if(len(beat_outs)>0):
							nbr_within_beat=pd.Series(distance_matrix[o][distance_matrix[o].index.isin(beat_outs)]).idxmin()
							if(distance_matrix[o][nbr_within_beat]>distance_matrix[o][nbr_outside_beat]):
								overlap_outs_cnt=overlap_outs_cnt+1
								overlap_outs.append(o)
								overlap_clus.append(df[df['path_x']==nbr_outside_beat]['van_id'].unique()[0])
						else:
							nbt=df[df['path_x']==nbr_outside_beat]['van_id'].unique()[0]
							if(nbt.split('_')[-1]=='Rented'):
								if(underutilized(nbt,'Rented')):
									overlap_outs_cnt=overlap_outs_cnt+1
									overlap_outs.append(o)
									# if there are more than one beat choose near by one
									overlap_clus.append(nbt)
							else:
								if(underutilized(nbt,'Owned')):
									overlap_outs_cnt=overlap_outs_cnt+1
									overlap_outs.append(o)
									# if there are more than one beat choose near by one
									overlap_clus.append(nbt)
							
					if(overlap_outs_cnt>0):        
						overlap_percent=((overlap_outs_cnt)/len(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}))*100
					else:
						overlap_percent=0
					if(overlap_percent>max_overlap_percent):
						overlap_outlets={}
						max_overlap_percent=overlap_percent
						max_overlap_bt1=bt
						from collections import Counter 
						def most_frequent(List): 
							occurence_count = Counter(List) 
							return occurence_count.most_common(1)[0][0] 	
						max_overlap_bt2=most_frequent(overlap_clus)
						for i in range(len(overlap_clus)):
							if(overlap_clus[i] in overlap_outlets.keys()):
								overlap_outlets[overlap_clus[i]].append(overlap_outs[i])
							else:
								overlap_outlets[overlap_clus[i]]=[]
								overlap_outlets[overlap_clus[i]].append(overlap_outs[i])
						
		
				return   max_overlap_bt1,max_overlap_bt2,overlap_outlets  
							
			def outlets_mixing_allowed(df1,df2):
				deltyp1=df1['del_type'].unique()[0]
				deltyp2=df2['del_type'].unique()[0]
				if((deltyp1.split('_')[0]=='All') and (deltyp2.split('_')[0]=='All')):
					plgclubtyp='default_underutilized'
				elif((deltyp1.split('_')[0]!='All') and (deltyp2.split('_')[0]!='All')):
					plgclubtyp='exception_underutilized'
				else:
					return False
				#c1=deltyp1.split('_')[1].split('|')
				#c2=deltyp2.split('_')[1].split('|')
				#plgclubtyp='default_underutilized'
				c1list=[]
				l=[]
				for ri in df1['channel'].astype(str):
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9]+', ',', str(ri))).split(',')) 
				c1list=[i for i in l if i!='' if i!='nan'] 
				'''
				if(not('nan' in df1['channel'].astype(str).str.lower().unique())):
					for ri in df1['channel']:
						c1list=c1list+list(set(ri)-{''})
						c1list=list(set(c1list))'''
				c2list=[]
				l=[]
				for ri in df2['channel'].astype(str):
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9]+', ',', str(ri))).split(',')) 
				c2list=[i for i in l if i!='' if i!='nan'] 
				'''
				if(not('nan' in df2['channel'].astype(str).str.lower().unique())):
					for ri in df2['channel']:
						c2list=c2list+list(set(ri)-{''})
						c2list=list(set(c2list))'''
				for i,r in plg_clubbing_[plg_clubbing_['type'].isin([plgclubtyp])].iterrows():
				   if((len(plg_clubbing_[plg_clubbing_['type'].isin([plgclubtyp])])==1) or (len(set(c2list+c1list).intersection(set(r['channel'])))==len(set(c2list+c1list)))):
						   p1list=[]
						   '''
						   if(not('nan' in df1['plg'].astype(str).str.lower().unique())):
							   for ri in df1['plg']:
									p1list=p1list+list(set(ri)-{''})
									p1list=list(set(p1list))'''
						   l=[]
						   for ri in df1['plg'].astype(str):
							#print(sub)
								  l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(ri))).split(',')) 
						   p1list=[i for i in l if i!='' if i!='nan'] 
						   p2list=[]
						   '''
						   if(not('nan' in df2['plg'].astype(str).str.lower().unique())):
							   for ri in df2['plg']:
								   p2list=p2list+list(set(ri)-{''})
								   p2list=list(set(p2list))'''
						   l=[]
						   for ri in df2['plg'].astype(str):
							#print(sub)
								  l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(ri))).split(',')) 
						   p2list=[i for i in l if i!='' if i!='nan']  
						   
						   if((len(p1list)==0) or (len(p2list)==0)):
							   return True
						   if(len(r['groups'])==0):
							   g=['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']
							   if(len(set(p2list+p1list).intersection(set(g)))==len(set(p2list+p1list))):
								   #print(g,'T')
								   return True
						   for g in r['groups']:    
							   if(len(set(p2list+p1list).intersection(set(g)))==len(set(p2list+p1list))):
								   #print(g,'T')
								   return True
				   else:
						  continue
				
				return False
				
				
				
			def check_if_both_comb_possible(bt1,bt2,comb_df,bt1_overlap_outs):
				c1=0
				if(bt1.split('_')[-1]=='Rented'):
					bt1=bt1.split('_')[0]
				if(bt2.split('_')[-1]=='Rented'):
					bt2=bt2.split('_')[0]
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
		#        if((len(set(outs).intersection(set(exclusive_outlets)))!=len(outs)) or (len(set(outs).intersection(set(exclusive_outlets)))!=0)):
		#            return False
				for o in outs:
					if(o in outlets_allowed_forvan[bt1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[bt2]):
						c2=c2+1
						
				if((c1==len(outs)) or (c2==len(outs))):
					return True
					
				else:
					return False
						
		
			def find_apt_van(comb_df,bt1,bt2):
				#if there is any ngbr beat in vicinity go for larger beat else go for smaller beat
				#usually this smaller beat wud be fully filled and this exercise wud be a waste of time
				#sometimes smaller beat wudnt make sense if in original beat it was fully utilised
				
				if(bt1.split('_')[-1]=='Rented'):
					b1=bt1.split('_')[0]
					b1typ='Rented'
				else:
					b1=bt1
					b1typ='Owned'
				if(bt2.split('_')[-1]=='Rented'):
					b2=bt2.split('_')[0]
					b2typ='Rented'
				else:
					b2=bt2
					b2typ='Owned'
				if((van_weight_dict[b1]<van_weight_dict[b2])):
					smaller_beat=bt1
					smaller_beat_typ=b1typ
					larger_beat=bt2
					larger_beat_typ=b2typ
				elif((van_weight_dict[b2]<van_weight_dict[b1])):
					smaller_beat=bt2
					smaller_beat_typ=b2typ
					larger_beat=bt1
					larger_beat_typ=b1typ
				else:
					if((van_endtime_dict[b1]<van_endtime_dict[b2])):
						smaller_beat=bt1
						smaller_beat_typ=b1typ
						larger_beat=bt2
						larger_beat_typ=b2typ
					else:
						smaller_beat=bt2
						smaller_beat_typ=b2typ
						larger_beat=bt1
						larger_beat_typ=b1typ
					
				c1=0
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
				for o in outs:
					if(o in outlets_allowed_forvan[b1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[b2]):
						c2=c2+1
						
				if((c1==len(outs)) and (c2!=len(outs))):
					return bt1,bt2
				elif((c2==len(outs)) and (c1!=len(outs))):
					return bt2,bt1
				else:
					if((larger_beat.split('_')[-1]=='Rented') and (smaller_beat.split('_')[-1]!='Rented')):
						return smaller_beat,larger_beat
					
					return larger_beat,smaller_beat
				'''
				if(not(underutilized(smaller_beat,smaller_beat_typ))):
					return larger_beat,smaller_beat
				else:
					if(not(underutilized(larger_beat,larger_beat_typ))):
						return smaller_beat,larger_beat
					else:
						return larger_beat,smaller_beat
				'''
			
			
			
			def all_beats_complied(df):
				df=pd.concat([df,subdf])
				all_beats=df['van_id'].unique()
				bcnt=0
				for b in all_beats:
					odf=df[df['van_id']!=b]
					workdf_cluster=df[df['van_id']==b]
					van1=workdf_cluster['van_id'].unique()[0] 
					if (van1.split('_')[-1]=='Rented'):
						van1=van1.split('_')[0]
					van1wgt=workdf_cluster['cum_weight'].unique()[0]
					van1tm=workdf_cluster['endtime'].unique()[0]
					van1vol=workdf_cluster['cum_volume'].unique()[0]
					van1nbills=workdf_cluster['num_bills'].unique()[0]
					van1endtime=0
					tot_endtime=0
					if(van1 in van_multitrip_dict.keys()):
						if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
							if(van_cutoff_dict[van1]=='yes'):
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
							else:
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
		
							odf['van_id2']=odf['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
							for v in odf['van_id'].unique():
								if(len(odf[(odf['van_id']==v) & (odf['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
									tot_endtime=tot_endtime+odf[(odf['van_id']==v)]['endtime'].unique()[0]
					
						else:
							van1endtime=van_endtime_dict[van1]
					else:
						van1endtime=van_endtime_dict[van1]
					if((van1wgt<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
						bcnt=bcnt+1
				return (bcnt==len(all_beats))
			
			def find_non_compliance_beat(df):
				df=pd.concat([df,subdf])
				all_beats=df['van_id'].unique()
				for b in all_beats:
					odf=df[df['van_id']!=b]
					workdf_cluster=df[df['van_id']==b]
					van1=workdf_cluster['van_id'].unique()[0] 
					if (van1.split('_')[-1]=='Rented'):
						van1=van1.split('_')[0]
					van1wgt=workdf_cluster['cum_weight'].unique()[0]
					van1tm=workdf_cluster['endtime'].unique()[0]
					van1vol=workdf_cluster['cum_volume'].unique()[0]
					van1nbills=workdf_cluster['num_bills'].unique()[0]
					van1endtime=0
					tot_endtime=0
					if(van1 in van_multitrip_dict.keys()):
						if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
							if(van_cutoff_dict[van1]=='yes'):
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
							else:
								van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
							odf['van_id2']=odf['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
							for v in odf['van_id'].unique():
								if(len(odf[(odf['van_id']==v) & (odf['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
									print(v)
									tot_endtime=tot_endtime+odf[(odf['van_id']==v)]['endtime'].unique()[0]
					
						else:
							van1endtime=van_endtime_dict[van1]
					else:
						van1endtime=van_endtime_dict[van1]
						
					if((van1wgt<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
						continue
					else:
						return b
				return 'nil'
			
			def find_nearest_cluster(mindistout,bt,df,non_cluster_outs,cluster_outs):
				#print(len(non_cluster_outs))
				outcluster_mindistout=pd.Series(distance_matrix[mindistout][distance_matrix[mindistout].index.isin(set(non_cluster_outs)-set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor',mindistout})]).idxmin()
				return list(set(df[df['path_x']==outcluster_mindistout]['van_id'].unique())-{bt})[0]
			
			def find_low_cost_empty_van(df):
				empty_vans=[]
				van_path=df.groupby(['van_id'])['path_x'].apply(list)
				for i in van_path.index:
					if(list(set(van_path[i]))==['Distributor']):
						empty_vans.append(i)
				mincost=1000000
				mincostvan='nil'
				for v in empty_vans:
					if((len(v.split('_'))>1) and (v.split('_')[-1]!='Rented')):
						if(van_cost_dict['_'.join(v.split('_')[:-1])+'_'+'1']<mincost):
							mincost=van_cost_dict['_'.join(v.split('_')[:-1])+'_'+'1']
							mincostvan=v
					elif((len(v.split('_'))>1) and (v.split('_')[-1]=='Rented')):
						if(van_cost_dict[v.split('_')[0]]<mincost):
							mincost=van_cost_dict[v.split('_')[0]]
							mincostvan=v
					else:
						if(van_cost_dict[v]<mincost):
							mincost=van_cost_dict[v]
							mincostvan=v
				return mincostvan   
			
			def check_limits_complied2(workdf_cluster,datfr):
				if(len(workdf_cluster)<=0):
					return True
				van1=workdf_cluster['van_id'].unique()[0] 
				datfr=datfr[~((datfr['van_id'].isin(workdf_cluster['van_id'].unique())) & (datfr['path_x'].isin(workdf_cluster['path_x'].unique())))].copy(deep=True)
				workdf_cluster=resequence(workdf_cluster)
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van1endtime=0
				tot_endtime=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								tot_endtime=tot_endtime+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
					
				if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
					return True
				else:
					return False    
			#if(non_constrained_beats_underuts):
			bt1=''
			bt2=''
		
			al_vis_beat_pairs=[]    
			#if there is underutilisation then do this
		
			while((bt1!='nil') and (bt2!='nil')):
				print(bt1,bt2,df['weights'].sum())
				df_init_init=df.copy(deep=True)
				bt1,bt2,overlap_outs=find_max_overlap_beats(set(beats)-set(al_vis_beat_pairs),df)
				vis=[]
				'''
				while(((bt1,bt2) in al_vis_beat_pairs) or ((bt2,bt1) in al_vis_beat_pairs)):            
					bt1,bt2,overlap_outs=find_max_overlap_beats(list(set(beats)-{bt1,bt2}))
					if(len(al_vis_beat_pairs)==len(beats)*len(beats)-1):
						bt1='nil'
						break
				'''
				if(set(al_vis_beat_pairs)==len(df['van_id'].unique())):
					break
				#if they are not nil
				if((bt1=='nil') or (bt2=='nil')):
					continue
				l=[]
				for k in overlap_outs.keys():
					l.extend(overlap_outs[k])
				remianing_outs=set(df[df['van_id']==bt1]['path_x'])-{'Distributor'}-set(l)
				al_vis_beat_pairs.append(bt1)
				mixedall=0
				obeats=[]
				
				for k in overlap_outs.keys():
					if(outlets_mixing_allowed(df[df['van_id']==k],df[(df['van_id']==bt1) & (df['path_x'].isin(overlap_outs[k]))])):
						#mixedall=mixedall+1
						comb_df=resequence(pd.concat([df[df['van_id'].isin([k])],df[(df['van_id'].isin([bt1])) & (df['path_x'].isin(overlap_outs[k]))]]))            
						if(check_if_both_comb_possible(bt1,k,comb_df,overlap_outs[k])):
							#van_name,other_beat=find_apt_van(comb_df,bt1,k)
							#obeats.append(other_beat)
							#print(van_name,other_beat,bt1)
							comb_df['van_id']=k
							comb_df['del_type']=van_deltyp_dict[k]
							df=df[~df['van_id'].isin([k])]
							mixedall=mixedall+1
							df=df[~((df['van_id'].isin([bt1])) & (df['path_x'].isin(overlap_outs[k])))]
							df=pd.concat([df,comb_df])
							#df=pd.concat([df,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[other_beat], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':del_type,'time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
						
		#                else:
		#                    van_name=find_apt_van_others(comb_df,list(set(beats)-set([bt1,k])))
		#                    if(van_name!='nil'):
		#                       three_df=make_three_beats(van_name,bt1,bt2,comb_df) 
		#                       df=df[~df['van_id'].isin([bt1,bt2,van_name])]
		#                       df=pd.concat([df,three_df])
		#                       mixedall=mixedall+1
							   
					else:
						print('cannot merge-shud we roll back?')
						#mixedall=0
						#df=df_init_init.copy(deep=True)
						#break
				
				already_vis=[]
				
				for ob in set(obeats):
						df=pd.concat([df,pd.DataFrame({'path_x':['Distributor'], 'endtime':[0], 'num_bills':[0], 'cum_weight':[0], 'van_id':[ob], 'weights':[0],'cum_volume':[0], 'volumes':[0], 'del_type':van_deltyp_dict[ob],'del_typ2':'nil','time':[0], 'bill_numbers':[[]], 'Basepack':[[]],'partyhll_code':['Distributor'], 'outlet_latitude':df[df['path_x']=='Distributor']['outlet_latitude'].unique()[0], 'outlet_longitude':df[df['path_x']=='Distributor']['outlet_longitude'].unique()[0], 'day':['day']})])
		
				if((mixedall==len(overlap_outs.keys())) and (set(obeats)=={bt1})):
					#treat rem_outlets also
					remianing_outs_copy=remianing_outs.copy()
					for ro in remianing_outs:
						non_cluster_outs=list(set(df[df['van_id']!=bt1]['path_x'].unique())-{'Distributor'})
						if(len(non_cluster_outs)<=0):
							continue
						clus=find_nearest_cluster(ro,bt1,df,non_cluster_outs,[])
						if(clus.split('_')[-1]=='Rented'):
							clus2=clus.split('_')[0]
						else:
							clus2=clus
						
						if(ro not in outlets_allowed_forvan[clus2]):
							eligible_vs=[]
							for v in outlets_allowed_forvan.keys():
								if(v.split('_')[-1]=='Rented'):
									v2=v.split('_')[0]
								else:
									v2=v 
								if(ro in outlets_allowed_forvan[v2]):
									eligible_vs.append(v)
							non_cluster_outs=list(set(df[df['van_id'].isin(eligible_vs)]['path_x'].unique())-{'Distributor'})
							clus=find_nearest_cluster(ro,bt1,df,non_cluster_outs,[])
							if(clus.split('_')[-1]=='Rented'):
								clus2=clus.split('_')[0]
							else:
								clus2=clus 
							
						bt_bt=df[(df['van_id']==bt1) & (df['path_x'].isin(remianing_outs_copy))]
						if(outlets_mixing_allowed(df[df['van_id']==clus],bt_bt[bt_bt['path_x']==ro])):
							new_bt=resequence(pd.concat([df[df['van_id']==clus],bt_bt[bt_bt['path_x']==ro]]))
							if(len(bt_bt[bt_bt['path_x']!=ro])>0):
								modi_bt=resequence(bt_bt[bt_bt['path_x']!=ro])
							else:
								modi_bt=pd.DataFrame()
								
							sub_df=pd.concat([new_bt,modi_bt])
							df=df[~((df['van_id']==bt1) & (df['path_x'].isin(remianing_outs_copy)))].copy(deep=True)
							df=pd.concat([df[~df['van_id'].isin([clus])],sub_df])
							already_vis.append(ro)
							remianing_outs_copy.remove(ro)
						
				df.drop_duplicates(subset =['van_id','path_x'], keep = 'first', inplace = True)
				
				already_vis=[]
				vis_beats=[]
				fl=False
				
				for v in df['van_id'].unique():
					if(len(df[df['van_id']==v])>0):
						print(v)
						print(df[df['van_id']==v]['cum_weight'].unique()[0])
						print(df[df['van_id']==v]['endtime'].unique()[0])
						sdf=resequence(df[df['van_id']==v])
						print(sdf['cum_weight'].unique()[0])
						print(df[df['van_id']==v]['endtime'].unique()[0])
						df=pd.concat([df[df['van_id']!=v],sdf])
				
				df_inittt=df.copy(deep=True)
				
				def find_non_compliance_beats(df):
					df=pd.concat([df,subdf])
					all_beats=df['van_id'].unique()
					non_compliance_beats=[]
					for b in all_beats:
						odf=df[df['van_id']!=b]
						workdf_cluster=df[df['van_id']==b]
						van1=workdf_cluster['van_id'].unique()[0] 
						if (van1.split('_')[-1]=='Rented'):
							van1=van1.split('_')[0]
						van1wgt=workdf_cluster['cum_weight'].unique()[0]
						van1tm=workdf_cluster['endtime'].unique()[0]
						van1vol=workdf_cluster['cum_volume'].unique()[0]
						van1nbills=workdf_cluster['num_bills'].unique()[0]
						van1endtime=0
						tot_endtime=0
						if(van1 in van_multitrip_dict.keys()):
							if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
								if(van_cutoff_dict[van1]=='yes'):
									van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
								else:
									van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
								odf['van_id2']=odf['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
								for v in odf['van_id'].unique():
									if(len(odf[(odf['van_id']==v) & (odf['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
										tot_endtime=tot_endtime+odf[(odf['van_id']==v)]['endtime'].unique()[0]
						
							else:
								van1endtime=van_endtime_dict[van1]
						else:
							van1endtime=van_endtime_dict[van1]
						if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
							continue
						else:
							non_compliance_beats.append(b)
					return non_compliance_beats
				
				def find_non_empty_vans(df):
					empty_vans=[]
					van_path=df.groupby(['van_id'])['path_x'].apply(list)
					for i in van_path.index:
						if(list(set(van_path[i]))==['Distributor']):
							empty_vans.append(i)
					non_empty_vans=list(set(df['van_id'].unique())-set(empty_vans)) 
					return non_empty_vans   
				
				non_compliance_beats=find_non_compliance_beats(df)
				non_empty_vans=find_non_empty_vans(df)
				
				for btt in non_compliance_beats:
					df_init=df.copy(deep=True)
					vis_beats=[]
					non_empty_vans=find_non_empty_vans(df)
					prev_exchange_pairs=[]
					exchange_pairs=[]
					df_init2=df.copy(deep=True)
					clus=''
					prev_bt=''
					out_already_vis_in_beat={}
					for v in non_empty_vans:
						out_already_vis_in_beat[v]=[]
					assgn_emp_van=0
					while((not(all_beats_complied(df))) and (not(len(set(vis_beats))==len(non_empty_vans)))):
						bt=find_non_compliance_beat(df)
						if(bt not in out_already_vis_in_beat.keys()):
							out_already_vis_in_beat[bt]=[]
		#                    if(bt==clus):
		#                        clus_l=[]
		#                        for o in set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}:
		#                           non_cluster_outs=list(set(df[df['van_id']!=bt]['path_x'].unique())-{'Distributor'})                    
		#                           clus=find_nearest_cluster(o,bt,df,non_cluster_outs)
		#                           clus_l.append(clus)
		#                        if((len(set(clus_l))==1) and (clus_l[0]==prev_bt)):
		#                            df=df_init2.copy(deep=True)
		#                            bt=prev_bt
		#                    if(prevbt==bt):
		#                        df=df_init_init.copy(deep=True)
		#                        break
						print(bt)
						vis_beats.append(bt)
						already_vis=[]
						df_init2=df.copy(deep=True)
						#prev_exchange_pairs=exchange_pairs.copy()
						exchange_pairs=[]
						restricted_bts=[bt]
						while(not(check_limits_complied2(df[df['van_id']==bt],df))):
							if(bt not in out_already_vis_in_beat.keys()):
								out_already_vis_in_beat[bt]=[]
							cluster_outs=list(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}-set(already_vis)-set(out_already_vis_in_beat[bt]))                
							non_cluster_outs=list(set(df[~df['van_id'].isin(restricted_bts)]['path_x'].unique())-{'Distributor'})                    
							if((len(cluster_outs)>0) and (len(non_cluster_outs)>0)):
								print('find outlet to be swapped')
								dm=distance_matrix[non_cluster_outs].T[cluster_outs].T 
								values=list(dm.min(axis=1))
								mindistout=list(dm.index)[values.index(min(values))]
								
								
								clus=find_nearest_cluster(mindistout,bt,df,non_cluster_outs,cluster_outs)
								
								if(clus.split('_')[-1]=='Rented'):
									clus2=clus.split('_')[0]
								else:
									clus2=clus
									
								if((mindistout in cluster_outs) and (mindistout in non_cluster_outs)):
									print('mindistout in both')
									if(not((check_limits_complied2(df[(df['van_id']==bt) & (df['path_x']!=mindistout)],df)) and (df[(df['van_id'].isin([bt,clus])) & (df['path_x']==mindistout)]['weights'].sum()<van_weight_dict[clus2]))):
										already_vis.append(mindistout)
										continue
								if(mindistout not in outlets_allowed_forvan[clus2]):
									print('mindistout not allowed')
									already_vis.append(mindistout)
									continue
								#if(((mindistout,bt) in prev_exchange_pairs) and (clus==prev_bt)):
		#                            if((clus==prev_bt)):
		#                                print('same as prev exchanges/beat')
		#                                already_vis.append(mindistout)
		#                                continue
								
								clus_l=[]
								for o in set(df[df['van_id']==clus]['path_x'].unique())-{'Distributor'}:
								   ncos=list(set(df[df['van_id']!=clus]['path_x'].unique())-{'Distributor'})                    
								   c=find_nearest_cluster(o,clus,df,ncos,list(set(df[df['van_id']==clus]['path_x'].unique())-{'Distributor'}))
								   clus_l.append(c)
								if((len(set(clus_l))==1) and (clus_l[0]==bt)):
									if(clus.split('_')[-1]=='Rented'):
										ctyp='Rented'
									else:
										ctyp='Owned'
									if(not(check_limits_complied2(pd.concat([df[df['van_id']==clus],df[(df['van_id']==bt) & (df['path_x']==mindistout)]]),df))):
										print('adding res beat',clus)
										restricted_bts.append(clus)
										continue
								
								
								bt_bt=df[df['van_id']==bt]
								if(not(outlets_mixing_allowed(df[df['van_id']==clus],bt_bt[bt_bt['path_x']==mindistout]))):
									print('outlets_mixing_not_allowed')
									already_vis.append(mindistout)
									continue
								print(mindistout,clus2,'changed') 
								exchange_pairs.append((mindistout,clus))
								if(clus  not in out_already_vis_in_beat.keys()):
									out_already_vis_in_beat[clus]=[]                            
								out_already_vis_in_beat[clus].append(mindistout)
								new_bt=resequence(pd.concat([df[df['van_id']==clus],bt_bt[bt_bt['path_x']==mindistout]]))
								modi_bt=resequence(bt_bt[bt_bt['path_x']!=mindistout])
								sub_df=pd.concat([new_bt,modi_bt])
								df=pd.concat([df[~df['van_id'].isin([clus,bt])],sub_df])
								already_vis.append(mindistout)
								print(len(df[df['van_id']==bt]))
							else:
								
								print('clstrout empty hence using a empty van')
								df=df_init.copy(deep=True)
								assgn_emp_van=assgn_emp_van+1
								bt=find_non_compliance_beat(df)
								emp_van=find_low_cost_empty_van(df)
								#v_emp_van.append(emp_van)
								if(assgn_emp_van>6):
										emp_van='nil'
								if(emp_van!='nil'):
									sub_df=df[df['van_id']==emp_van]
									non_comply_beat=df[df['van_id']==bt].reset_index(drop=True)
									non_comply_beat_copy=non_comply_beat.copy(deep=True)
									
									if(emp_van.split('_')[-1]=='Rented'):
										emp_van2=emp_van.split('_')[0]
									else:
										emp_van2=emp_van
										
									for idx in reversed(non_comply_beat.index):
										if(non_comply_beat.iloc[idx]['path_x'] in outlets_allowed_forvan[emp_van2]):
											 sub_df=pd.concat([sub_df,pd.DataFrame(non_comply_beat.iloc[idx]).T])
											 non_comply_beat_copy=non_comply_beat_copy[non_comply_beat_copy.index!=idx].copy(deep=True)
											 sub_df=resequence(sub_df)
											 non_comply_beat_copy=resequence(non_comply_beat_copy)
										else:
											
											continue
										if(check_limits_complied2(non_comply_beat_copy,df)):
											break
									df=df[~df['van_id'].isin([bt,emp_van])]
									df=pd.concat([df,sub_df])
									df=pd.concat([df,non_comply_beat_copy])
									df_init=df.copy(deep=True)
								else:
									print('ROLL_BACK')
									assgn_emp_van=0
									df=df_init_init.copy(deep=True)
		
							prev_bt=bt
							
						if((len(set(vis_beats))==len(non_empty_vans)) and (not(all_beats_complied(df)))):
							print('ROLL_BACK')
							df=df_init_init.copy(deep=True)
							
		
			len(df)
			df['day']='day'
			all_outlets=list(set(df['path_x'].unique())-{'Distributor'})
			#df.to_csv('svr_2ndfeb_ac_merge_cc_test2_1_2.csv')
			nn_df=pd.DataFrame(columns=['outlet','outlet_neighbour','nghbr_dist'])
			for o in all_outlets:
				eligible_ngbrs_of_outlet=[]
				for v in df[df['path_x']==o]['van_id'].unique():
					#v=workdf_cluster['van_id'].unique()[0]
					 if(v.split('_')[-1]=='Rented'):
						  v2=v.split('_')[0]
					 else:
						  v2=v
					 eligible_ngbrs_of_outlet.extend(outlets_allowed_forvan[v2])
				eligible_ngbrs_of_outlet=list(set(all_outlets).intersection(set(eligible_ngbrs_of_outlet)))
				dist = pd.Series(distance_matrix[o][distance_matrix[o].index.isin(set(eligible_ngbrs_of_outlet)-{'Distributor',o})])    
				nearest_index = dist.idxmin()
				nn_df=pd.concat([nn_df,pd.DataFrame({'outlet':[o],'outlet_neighbour':[nearest_index],'nghbr_dist':[distance_matrix[o][nearest_index]]})])
				
			cluster_of=df.groupby(['path_x'])['van_id'].apply(list).to_dict()  
			
			nn_df=nn_df.sort_values(by=['nghbr_dist'], ascending=True).copy(deep=True)
			
			
			def beatouts_exchangable(out,nout):
				return True
			
			
			
			def push_to_cluster(out,workdf_cluster,workdf_ncluster):
				workdf_cluster=pd.concat([workdf_cluster,workdf_ncluster[workdf_ncluster['path_x']==out]])
				workdf_ncluster=workdf_ncluster[~workdf_ncluster['path_x'].isin([out])]
				workdf_cluster=resequence(workdf_cluster)
				workdf_ncluster=resequence(workdf_ncluster) 
				return workdf_cluster,workdf_ncluster
			
				
			def check_limits_complied(workdf_cluster,workdf_ncluster,datfr):
				datfr=pd.concat([datfr,subdf])
				van1=workdf_cluster['van_id'].unique()[0]
				van2=workdf_ncluster['van_id'].unique()[0]
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				if (van2.split('_')[-1]=='Rented'):
					van2=van2.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van2wgt=workdf_ncluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van2tm=workdf_ncluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van2vol=workdf_ncluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van2nbills=workdf_ncluster['num_bills'].unique()[0]
				van1endtime=0
				van2endtime=0
				van1_tot_time=0
				van2_tot_time=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								van1_tot_time=van1_tot_time+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
				
				if(van2 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van2]=='yes') and (not((van_cutoff_dict[van2]=='yes') and (van_endtime_dict[van2]==480)))):
						van2endtime=van_endtime_dict['_'.join(workdf_ncluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_ncluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								van2_tot_time=van2_tot_time+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]        
					else:
						van2endtime=van_endtime_dict[van2]
				else:
					van2endtime=van_endtime_dict[van2]
				
				if((van1wgt<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van2wgt<=van_weight_dict[van2]) and (van2vol<=van_volume_dict[van2]) and (van2nbills<=van_bill_dict[van2]) and (van1tm+van1_tot_time<=van1endtime) and (van2tm+van2_tot_time<=van2endtime)):
					return True
				else:
					return False
			
			def check_limits_complied2(workdf_cluster,datfr):
				datfr=pd.concat([datfr,subdf])
				if(len(workdf_cluster)<=0):
					return True
				van1=workdf_cluster['van_id'].unique()[0] 
				datfr=datfr[~((datfr['van_id'].isin(workdf_cluster['van_id'].unique())) & (datfr['path_x'].isin(workdf_cluster['path_x'].unique())))].copy(deep=True)
				workdf_cluster=resequence(workdf_cluster)
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van1endtime=0
				tot_endtime=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								tot_endtime=tot_endtime+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
					
				if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
					return True
				else:
					return False    
				
			nn_df=nn_df.reset_index(drop=True)  
			  
			for i,r in nn_df.iterrows():
				print('enter for')
				out=r['outlet']
				nout=r['outlet_neighbour']
				ndist=r['nghbr_dist']
				cluster_of=df.groupby(['path_x'])['van_id'].apply(list).to_dict()  
				#if in different cluster
				if(len(set(cluster_of[out]).intersection(set(cluster_of[nout])))==0):
					#print(i)
				  if(beatouts_exchangable(out,nout)):
					  df_init=df.copy(deep=True)
					  newdf=df[~df['van_id'].isin([cluster_of[out][0],cluster_of[nout][0]])]
					  workdf=df[df['van_id'].isin([cluster_of[out][0],cluster_of[nout][0]])]
					  workdf_cluster=workdf[workdf['van_id'].isin(cluster_of[out])]
					  workdf_ncluster=workdf[workdf['van_id'].isin(cluster_of[nout])]
					  tot_outs_cluster=len(set(workdf_ncluster['path_x'])-{'Distributor'})
					  if(outlets_mixing_allowed(workdf_ncluster,workdf_cluster[workdf_cluster['path_x']==out])):
						  workdf_ncluster=pd.concat([workdf_ncluster,workdf_cluster[workdf_cluster['path_x']==out]])
					  else:
						  continue
					  v=cluster_of[nout][0]
					  if(v.split('_')[-1]=='Rented'):
						  v2=v.split('_')[0]
					  else:
						  v2=v
					 
					  if(out not in outlets_allowed_forvan[v2]):
						  continue
					  workdf_cluster=workdf_cluster[~workdf_cluster['path_x'].isin([out])]
					  workdf_cluster=resequence(workdf_cluster)
					  workdf_ncluster=resequence(workdf_ncluster)
					  count=0
					  rev=False
					  already_vis=[]
					  while((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count!=tot_outs_cluster)):
    						 count=count+1
    						 print('enter while',count)
    						 cluster_outs=list(set(workdf_cluster['path_x'])-{'Distributor'})
    						 ncluster_outs=list(set(workdf_ncluster['path_x'])-{'Distributor'}-set(already_vis))
    						 dm=distance_matrix[cluster_outs].T[ncluster_outs].T 
    						 if((len(ncluster_outs)==0) or (len(cluster_outs)==0)):
    							 rev=True
    							 break
    						 values=list(dm.min(axis=1))
    						 if(min(values)>ndist):
    							 rev=True
    							 break
    						 #mindistout=list(dm.idxmin(axis=1))[values.index(min(values))]
    						 mindistout=list(dm.index)[values.index(min(values))]
    						 print('ncluster',mindistout in ncluster_outs)
    						 print('cluster',mindistout in cluster_outs)
    						 print('-----------------')
    						 v=workdf_cluster['van_id'].unique()[0]
    						 if(v.split('_')[-1]=='Rented'):
    							  v2=v.split('_')[0]
    						 else:
    							  v2=v
    						 if((not(mindistout in outlets_allowed_forvan[v2])) and (not(outlets_mixing_allowed(workdf_cluster,workdf_ncluster[workdf_ncluster['path_x']==mindistout])))):
    							 already_vis.append(mindistout)
    							 continue
    						 wgt_to_add=float(workdf_ncluster[workdf_ncluster['path_x']==mindistout]['weights'])
    						 if workdf_cluster['van_id'].unique()[0].split('_')[-1]=='Rented':
    							 wgt_can_handle=(van_weight_dict[(workdf_cluster['van_id'].unique()[0]).split('_')[0]])-(workdf_cluster['weights'].sum())
    						 else:
    							 wgt_can_handle=(van_weight_dict[workdf_cluster['van_id'].unique()[0]])-(workdf_cluster['weights'].sum())
    							 
    						 print(wgt_to_add,wgt_can_handle)
    						 if(wgt_to_add>wgt_can_handle):
    							 
    							 already_vis.append(mindistout)
    							 continue
    						 print(workdf_cluster['van_id'].unique(),workdf_ncluster['van_id'].unique(),len(workdf_cluster),len(workdf_ncluster),workdf_cluster['weights'].sum(),workdf_ncluster['weights'].sum())
    						 #workdf_cluster,workdf_ncluster=push_to_cluster(mindistout,workdf_cluster,workdf_ncluster)
    						 workdf_cluster_init=workdf_cluster.copy(deep=True)
    						 workdf_ncluster_init=workdf_ncluster.copy(deep=True)
    						 workdf_cluster,workdf_ncluster=push_to_cluster(mindistout,workdf_cluster,workdf_ncluster)
    						 print(workdf_cluster['van_id'].unique(),workdf_ncluster['van_id'].unique(),len(workdf_cluster),len(workdf_ncluster),workdf_cluster['weights'].sum(),workdf_ncluster['weights'].sum())
    						 already_vis.append(mindistout)
    						 if(not(check_limits_complied2(workdf_cluster,df_init))):
        							workdf_cluster=workdf_cluster_init.copy(deep=True)
        							workdf_ncluster=workdf_ncluster_init.copy(deep=True) 
					  if((rev) or ((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count==tot_outs_cluster))):
						  print('unchanged')
						  df=df_init.copy(deep=True)
					  else:
						  print('changed')
						  #if(not(check_limits_complied2(pd.concat([workdf_cluster,workdf_ncluster])))):
						  df=pd.concat([newdf,workdf_cluster]) 
						  df=pd.concat([df,workdf_ncluster]) 
			#              else:
			#                  workdf_cluster=pd.concat([workdf_cluster,workdf_ncluster])
			#                  workdf_cluster=resequence(workdf_cluster)
			#                  df=pd.concat([newdf,workdf_cluster]) 
			
			len(df)
			#df.to_csv('fnw_444230_7mar_wc_cc.csv')
			al_vis_bt=[]
			
			def find_nearest_cluster(rem_df,new_df):
			   clstrouts=list(rem_df['path_x'].unique()) #+list(new_df[new_df['van_id'].isin(rem_df['van_id'])]['path_x'].unique()) 
			   nclstrouts=list(new_df[~new_df['van_id'].isin(list(rem_df['van_id'])+al_vis_bt)]['path_x'].unique()) 
			   #dm=distance_matrix[clstrouts].T[nclstrouts].T 
			   #values=list(dm.min(axis=1))
			   mindistoutclus=[]
			   if(len(set(nclstrouts)-{'Distributor'})==0):
				   
				   return 'nil'
			   for co in clstrouts:
				   mindistout=pd.Series(distance_matrix[co][distance_matrix[co].index.isin(set(nclstrouts)-{'Distributor'})]).idxmin()
				   mindistoutclus.append(new_df[new_df['path_x']==mindistout]['van_id'].unique()[0])
			   mindistoutclus_cnt={}
			   for c in set(mindistoutclus):
				   mindistoutclus_cnt[c]=mindistoutclus.count(c)
				   
				   '''
				min_value1=min(values)
			   mindistout1=list(dm.index)[values.index(min_value1)]
			   clstrouts=list(new_df[new_df['van_id'].isin(rem_df['van_id'])]['path_x'].unique()) 
			   nclstrouts=list(new_df[~new_df['van_id'].isin(rem_df['van_id'])]['path_x'].unique()) 
			   dm=distance_matrix[clstrouts].T[nclstrouts].T 
			   values=list(dm.min(axis=1))
			   min_value2=min(values)
			   mindistout2=list(dm.index)[values.index(min_value2)]
			   if(min_value1>min_value2):
				   mindistout=mindistout2
			   else:
				   mindistout=mindistout1'''
			   
			   return max(mindistoutclus_cnt, key=mindistoutclus_cnt.get)
			
			
				
				
			def find_neighbour_outside(n_bts_outs,ot):
				dist = distance_matrix[ot][distance_matrix[ot].index.isin(list(n_bts_outs))]
				dist = pd.Series(dist)
				nearest_index = dist.idxmin() 
				return nearest_index
				
				
			def find_sim_beats(bt,df):
				return list(df[~(df['van_id']==bt)]['van_id'].unique())            
			
			def check_if_both_comb_possible(bt1,bt2,comb_df):
				c1=0
				if(bt1.split('_')[-1]=='Rented'):
					bt1=bt1.split('_')[0]
				if(bt2.split('_')[-1]=='Rented'):
					bt2=bt2.split('_')[0]
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
		#        if((len(set(outs).intersection(set(exclusive_outlets)))!=len(outs)) or (len(set(outs).intersection(set(exclusive_outlets)))!=0)):
		#            return False
				for o in outs:
					if(o in outlets_allowed_forvan[bt1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[bt2]):
						c2=c2+1
						
				if((c1==len(outs))):
					return True
				else:
					return False
			
			def dist_jump_abnormal(bt,sim_bts,df):
				bt_df=df[df['van_id']==bt]
				nbt_df=df[~(df['van_id']==bt)]
				bt_outs=list(bt_df['path_x'].unique())
				n_bts_outs=list(nbt_df['path_x'].unique())
				checked=[]
				flag=False
				rem_df=pd.DataFrame()
				inter_dist=[]
				intra_dist=[]
				for i in range(len(bt_outs)-1,0,-1):
					ot=bt_outs[i]
					if(ot=='Distributor'):
					  checked.append(ot)
					  continue
					not_bt=bt_outs[i-1]
					
					not_nbt=find_neighbour_outside(n_bts_outs,ot)
					print(ot,not_bt,not_nbt)
					checked.append(ot)
					intra_dist.append(distance_matrix[ot][not_bt])
					inter_dist.append(distance_matrix[ot][not_nbt])
				
				if((len(intra_dist[:-1])>0) and (max(intra_dist[:-1])>2*inter_dist[intra_dist.index(max(intra_dist[:-1]))])):
					flag=True
			#        if(len(bt_outs)-1-intra_dist.index(max(intra_dist[:-1]))>intra_dist.index(max(intra_dist[:-1]))):
			#            rem_df=bt_df[bt_df['path_x'].isin(bt_outs[1:intra_dist.index(max(intra_dist[:-1]))+1])]
			#            bt_df=bt_df[bt_df['path_x'].isin(bt_outs[intra_dist.index(max(intra_dist[:-1]))+1:]+['Distributor'])]
			#        else:
					rem_df=bt_df[bt_df['path_x'].isin(bt_outs[::-1][:intra_dist.index(max(intra_dist[:-1]))+1])]
					bt_df=bt_df[bt_df['path_x'].isin(bt_outs[::-1][intra_dist.index(max(intra_dist[:-1]))+1:])]
				print(len(rem_df),len(bt_df))
				return flag,pd.concat([nbt_df,bt_df]),rem_df     
					
				
			def outlet_shift_across_cluster(rem_df,new_df,df_init):
				cluster=find_nearest_cluster(rem_df,new_df)
				if(cluster=='nil'):
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
					 return df,dont_check_again 
				 
				al_vis_bt.append(cluster)
				workdf_cluster=new_df[new_df['van_id'].isin(list(rem_df['van_id'].unique()))]
				workdf_ncluster=new_df[new_df['van_id'].isin([cluster])]
				#create_two_beats(workdf_cluster,workdf_cluster,rem_df)
				newdf=new_df[~new_df['van_id'].isin(list(rem_df['van_id'].unique())+[cluster])]
				print(len(workdf_cluster),len(workdf_ncluster),len(rem_df))
				if(outlets_mixing_allowed(workdf_ncluster,rem_df)):
					workdf_ncluster=pd.concat([workdf_ncluster,rem_df])
				else:
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
					 return df,dont_check_again 
				print(len(workdf_cluster),len(workdf_ncluster),len(rem_df))
				workdf_cluster=resequence(workdf_cluster)
				workdf_ncluster=resequence(workdf_ncluster)
				if(not(check_if_both_comb_possible(cluster,rem_df['van_id'].unique()[0],workdf_ncluster))):
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
					 return df,dont_check_again 
					
				tot_outs_cluster=len(set(workdf_ncluster['path_x'])-{'Distributor'})
				rev=False
				already_vis=[]
				count=0
				tot_wgt_added=rem_df['weights'].sum()
				wgt_to_add=0
				rem_wgt=tot_wgt_added
				dont_check_again=False
				while((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count!=tot_outs_cluster)): 
					 print('--------------------') 
					 count=count+1
					 cluster_outs=list(set(workdf_cluster['path_x'])-{'Distributor'})
					 ncluster_outs=list(set(workdf_ncluster['path_x'])-{'Distributor'}-set(already_vis))
					 dm=distance_matrix[cluster_outs].T[ncluster_outs].T 
					 if((len(ncluster_outs)==0) or (len(cluster_outs)==0)):
						 print('rev True')
						 rev=True
						 break
					 values=list(dm.min(axis=1))
					 mindistout=list(dm.index)[values.index(min(values))]
		#             if(mindistout in rem_df['path_x'].unique()):
		#                 print('min dist out was in work cluster')
		#                 already_vis.append(mindistout)
		#                 continue
					 v=workdf_cluster['van_id'].unique()[0]
					 if(v.split('_')[-1]=='Rented'):
						  v2=v.split('_')[0]
					 else:
						  v2=v
					 if((not(mindistout in outlets_allowed_forvan[v2])) and (not(outlets_mixing_allowed(workdf_cluster,workdf_ncluster[workdf_ncluster['path_x']==mindistout])))):
							 print('min dist out not in outlets_allowed_forvan')
							 already_vis.append(mindistout)
							 continue
					 wgt_to_add=float(workdf_ncluster[workdf_ncluster['path_x']==mindistout]['weights'])
					 if workdf_cluster['van_id'].unique()[0].split('_')[-1]=='Rented':
						 wgt_can_handle=(van_weight_dict[(workdf_cluster['van_id'].unique()[0]).split('_')[0]])-(workdf_cluster['weights'].sum())
					 else:
						 wgt_can_handle=(van_weight_dict[workdf_cluster['van_id'].unique()[0]])-(workdf_cluster['weights'].sum())
					 if(wgt_to_add>wgt_can_handle):
						 print('wgt_to_add>wgt_can_handle')
						 already_vis.append(mindistout)
						 continue
					 workdf_cluster_init=workdf_cluster.copy(deep=True)
					 workdf_ncluster_init=workdf_ncluster.copy(deep=True)
					 workdf_cluster,workdf_ncluster=push_to_cluster(mindistout,workdf_cluster,workdf_ncluster)
					 already_vis.append(mindistout)
					 if(not(check_limits_complied2(workdf_cluster,df_init))):
    						print('workdf_cluster not cpmplying') 
    						workdf_cluster=workdf_cluster_init.copy(deep=True)
    						workdf_ncluster=workdf_ncluster_init.copy(deep=True) 
				
				if((rev) or ((not(check_limits_complied(workdf_cluster,workdf_ncluster,newdf))) and (count==tot_outs_cluster))):
					 print('not changed')
					 df=df_init.copy(deep=True)
					 dont_check_again=True
				else:
					  print('chnaged')
					 #if(not(check_limits_complied2(resequence(pd.concat([workdf_cluster,workdf_ncluster]))))):
					 #if(not(check_limits_complied2(pd.concat([workdf_cluster,workdf_ncluster])))):
					  df=pd.concat([newdf,workdf_cluster]) 
					  df=pd.concat([df,workdf_ncluster]) 
		#             else:
		#                  workdf_cluster=pd.concat([workdf_cluster,workdf_ncluster])
		#                  workdf_cluster=resequence(workdf_cluster)
		#                  df=pd.concat([newdf,workdf_cluster]) 
					
				return df,dont_check_again    
			
				 
						 
			df_copy=df.copy(deep=True)
			vans=owned_van_order_tofill+rental_van_order_tofill
			
			if(len(df_copy['van_id'].unique())>1):
				for bt in vans[::-1]:
					if((bt in df_copy['van_id'].unique()) or (bt in [b.split('_')[0] for b in df_copy['van_id'].unique()])):
						al_vis_bt=[]
						while True:
							bt_df=df[df['van_id']==bt]
							nbt_df=df[~(df['van_id']==bt)]
							sim_bts=find_sim_beats(bt,df)
							df_init=df.copy(deep=True)
							flag=False
							flag,new_df,rem_df=dist_jump_abnormal(bt,sim_bts,df)
							print(flag)
							if(flag):
								df,dont_check_again=outlet_shift_across_cluster(rem_df,new_df,df_init) 
								if(dont_check_again):
									break
							else:
								df=new_df.copy(deep=True)
								break
						
					
			len(df)
			new_tdf=pd.concat([new_tdf,df])
		
		
		if(len(tdf['del_type'].unique())>1):
        		new_tdf=pd.concat([new_tdf,subdf])
        		new_tdf.to_csv(rscode+'_2v9.csv')
		else:
        		new_tdf=tdf.copy(deep=True)
        		new_tdf=pd.concat([new_tdf,subdf])
        		new_tdf.to_csv(rscode+'_2v8.csv')
		
		#add fixed rate,mutitrips,plgmapping,exclusivity,area
		gdf=new_tdf.groupby(['van_id'])['path_x'].apply(list)
		removed_vans=[]
		for v in new_tdf['van_id'].unique():
		  if(set(gdf[v])=={'Distributor'}):
			  new_tdf=new_tdf[new_tdf['van_id']!=v]
			  removed_vans.append(v)
		
		
		#merge tow vans
		def underutilized(bt,btyp):
				if(btyp=='Rented'):
					b=bt.split('_')[0]
				else:
					b=bt
				if(new_tdf[new_tdf['van_id']==bt]['weights'].sum()<0.68*van_weight_dict[b]):
					return True
				else:
					return False
		
		def outlets_mixing_allowed(df1,df2):
				deltyp1=df1['del_type'].unique()[0]
				deltyp2=df2['del_type'].unique()[0]
		#        if((deltyp1.split('_')[0]=='All') and (deltyp2.split('_')[0]=='All')):
		#            plgclubtyp='default_underutilized'
		#        elif((deltyp1.split('_')[0]!='All') and (deltyp2.split('_')[0]!='All')):
		#            plgclubtyp='exception_underutilized'
		#        else:
		#            return False
				#c1=deltyp1.split('_')[1].split('|')
				#c2=deltyp2.split('_')[1].split('|')
				plgclubtyp='default_underutilized'
				c1list=[]
				l=[]
				for ri in df1['channel'].astype(str):
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9]+', ',', str(ri))).split(',')) 
				c1list=[i for i in l if i!='' if i!='nan'] 
				'''
				if(not('nan' in df1['channel'].astype(str).str.lower().unique())):
					for ri in df1['channel']:
						c1list=c1list+list(set(ri)-{''})
						c1list=list(set(c1list))'''
				c2list=[]
				l=[]
				for ri in df2['channel'].astype(str):
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9]+', ',', str(ri))).split(',')) 
				c2list=[i for i in l if i!='' if i!='nan'] 
				'''
				if(not('nan' in df2['channel'].astype(str).str.lower().unique())):
					for ri in df2['channel']:
						c2list=c2list+list(set(ri)-{''})
						c2list=list(set(c2list))'''
				for i,r in plg_clubbing_[plg_clubbing_['type'].isin([plgclubtyp])].iterrows():
				   if((len(plg_clubbing_[plg_clubbing_['type'].isin([plgclubtyp])])==1) or (len(set(c2list+c1list).intersection(set(r['channel'])))==len(set(c2list+c1list)))):
						   p1list=[]
						   '''
						   if(not('nan' in df1['plg'].astype(str).str.lower().unique())):
							   for ri in df1['plg']:
									p1list=p1list+list(set(ri)-{''})
									p1list=list(set(p1list))'''
						   l=[]
						   for ri in df1['plg'].astype(str):
							#print(sub)
								  l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(ri))).split(',')) 
						   ###print(set(l))
						   p1list=[i for i in l if i!='' if i!='nan'] 
						   p2list=[]
						   '''
						   if(not('nan' in df2['plg'].astype(str).str.lower().unique())):
							   for ri in df2['plg']:
								   p2list=p2list+list(set(ri)-{''})
								   p2list=list(set(p2list))'''
						   l=[]
						   for ri in df2['plg'].astype(str):
							#print(sub)
								#print(str(ri))
								  l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(ri))).split(',')) 
						   #print(set(l))
						   p2list=[i for i in l if i!='' if i!='nan']  
						   
						   if((len(p1list)==0) or (len(p2list)==0)):
							   return True
						   if(len(r['groups'])==0):
							   g=['DETS','D+F', 'PP', 'PP-B','PP-A','HUL','FNB']
							   if(len(set(p2list+p1list).intersection(set(g)))==len(set(p2list+p1list))):
								   #print(g,'T')
								   return True
						   for g in r['groups']:    
							   if(len(set(p2list+p1list).intersection(set(g)))==len(set(p2list+p1list))):
								   #print(g,'T')
								   return True
				   else:
						  continue
				
				return False
		
			 
		def resequence(clusterdf):
			w_dict=clusterdf.groupby(['path_x'])['weights'].sum().to_dict()
			v_dict=clusterdf.groupby(['path_x'])['volumes'].sum().to_dict()
			b_dict=clusterdf.groupby(['path_x'])['bill_numbers'].apply(list).to_dict()
			for k in b_dict.keys():
				l=[]
				l.extend(b_dict[k])
				b_dict[k]=l            
				
			for k in b_dict.keys():
				l=[]
				for sub in b_dict[k]:
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9_]+', ',', str(sub))).split(',')) 
				b_dict[k]=[i for i in l if i!='']
				
			bp_dict=clusterdf.groupby(['path_x'])['Basepack'].apply(list).to_dict()
			
			for k in bp_dict.keys():
				l=[]
				l.extend(bp_dict[k])
				bp_dict[k]=l
			for k in bp_dict.keys():
				l=[]
				for sub in bp_dict[k]:
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9_]+', ',', str(sub))).split(',')) 
				bp_dict[k]=[i for i in l if i!=''] 
			c_dict=clusterdf.groupby(['path_x'])['channel'].apply(list).to_dict()    
			p_dict=clusterdf.groupby(['path_x'])['plg'].apply(list).to_dict()
			
			for k in c_dict.keys():
				l=[]
				l.extend(c_dict[k])
				c_dict[k]=l
			for k in c_dict.keys():
				l=[]
				for sub in c_dict[k]:
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9]+', ',', str(sub))).split(',')) 
				c_dict[k]=[i for i in l if i!=''] 
			
			for k in p_dict.keys():
				l=[]
				l.extend(p_dict[k])
				p_dict[k]=l
			for k in p_dict.keys():
				l=[]
				for sub in p_dict[k]:
					#print(sub)
					l.extend((re.sub('[^A-Za-z0-9-\+]+', ',', str(sub))).split(',')) 
				p_dict[k]=[i for i in l if i!=''] 
				
			#b_dict=dict(zip(clusterdf['path_x'],clusterdf['bill_numbers']))
			#bp_dict=dict(zip(clusterdf['path_x'],clusterdf['Basepack']))
			lat_dict=dict(zip(clusterdf['path_x'],clusterdf['outlet_latitude']))
			long_dict=dict(zip(clusterdf['path_x'],clusterdf['outlet_longitude']))
			lat_dict['Distributor']=input_data['rs_latitude'].unique()[0]
			long_dict['Distributor']=input_data['rs_longitude'].unique()[0]
			pathseq=['Distributor']
			wgtseq=[0]
			volseq=[0]
			timeseq=[0]
			billseq=[[]]
			bpseq=[[]]
			chseq=[[]]
			plgseq=[[]]
			deltyp=list(clusterdf['del_type'].unique())[0]
			numbills=0
			cum_weight=0
			cum_volume=0
			van_id=list(clusterdf['van_id'].unique())[0]
			van_id_init=van_id
			day=list(clusterdf['day'].unique())[0]
			latseq=[lat_dict['Distributor']]
			longseq=[long_dict['Distributor']]
			mids=list(set(w_dict.keys())-{'Distributor'})
			cnt=0
			
			while(cnt!=len(mids)):
				cnt=cnt+1
				end_time_seq=0
				dist = distance_matrix[pathseq[-1]][distance_matrix[pathseq[-1]].index.isin(list(set(mids)-set(pathseq)))]
				if(len(dist)==0):
					break
				dist = pd.Series(dist)
				nearest_index = dist.idxmin()
				pathseq.append(nearest_index)
				wgtseq.append(w_dict[nearest_index])
				volseq.append(v_dict[nearest_index])
				distance = dist[nearest_index]
				if (van_id.split('_')[-1]=='Rented'):
					van_id=van_id.split('_')[0]
				
				travel_time = math.ceil(distance * (1/van_speed_dict[van_id]))
				end_time_seq = end_time_seq + travel_time
				service_time=int(service_time_details[(w_dict[nearest_index]>=service_time_details['load_range_from_kgs'].astype(float)) & (w_dict[nearest_index]<service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].reset_index(drop=True)[0])
				end_time_seq=end_time_seq+ service_time
				timeseq.append(travel_time+service_time)
				billseq.append(b_dict[nearest_index])
				bpseq.append(bp_dict[nearest_index])
				chseq.append(c_dict[nearest_index])
				plgseq.append(p_dict[nearest_index])
				numbills=numbills+len(b_dict[nearest_index])
				cum_weight=cum_weight+w_dict[nearest_index]
				cum_volume=cum_volume+v_dict[nearest_index]
				latseq.append(lat_dict[nearest_index])
				longseq.append(long_dict[nearest_index])
				
			df_=pd.DataFrame()
			pathseq.append('Distributor')
			wgtseq.append(0)
			volseq.append(0)
			if van_id.split('_')[-1]=='Rented':
				timeseq.append(math.ceil(distance_matrix[pathseq[-2]]['Distributor'] * (1/van_speed_dict[van_id.split('_')[0]])))
			else:
				timeseq.append(math.ceil(distance_matrix[pathseq[-2]]['Distributor'] * (1/van_speed_dict[van_id])))
			
			billseq.append([''])
			bpseq.append([''])
			chseq.append([''])
			plgseq.append([''])
			latseq.append(lat_dict['Distributor'])
			longseq.append(long_dict['Distributor'])
			
			return pd.concat([df_,pd.DataFrame({'path_x':pathseq, 'endtime':[sum(timeseq)]*len(pathseq), 'num_bills':[numbills]*len(pathseq), 'cum_weight':[cum_weight]*len(pathseq), 'van_id':[van_id_init]*len(pathseq), 'weights':wgtseq,'cum_volume':[cum_volume]*len(pathseq), 'volumes':volseq, 'del_type':deltyp,'time':timeseq, 'bill_numbers':billseq, 'Basepack':bpseq,'partyhll_code':pathseq, 'outlet_latitude':latseq, 'outlet_longitude':longseq, 'day':[day]*len(pathseq),'plg':plgseq,'channel':chseq})])
		
		
		def find_apt_van(comb_df,bt1,bt2):
			#if there is any ngbr beat in vicinity go for larger beat else go for smaller beat
			#usually this smaller beat wud be fully filled and this exercise wud be a waste of time
			#sometimes smaller beat wudnt make sense if in original beat it was fully utilised
			
			if(bt1.split('_')[-1]=='Rented'):
				b1=bt1.split('_')[0]
				b1typ='Rented'
			else:
				b1=bt1
				b1typ='Owned'
			if(bt2.split('_')[-1]=='Rented'):
				b2=bt2.split('_')[0]
				b2typ='Rented'
			else:
				b2=bt2
				b2typ='Owned'
			if((van_weight_dict[b1]<van_weight_dict[b2])):
				smaller_beat=bt1
				smaller_beat_typ=b1typ
				larger_beat=bt2
				larger_beat_typ=b2typ
			elif((van_weight_dict[b2]<van_weight_dict[b1])):
				smaller_beat=bt2
				smaller_beat_typ=b2typ
				larger_beat=bt1
				larger_beat_typ=b1typ
			else:
				if((van_endtime_dict[b1]<van_endtime_dict[b2])):
					smaller_beat=bt1
					smaller_beat_typ=b1typ
					larger_beat=bt2
					larger_beat_typ=b2typ
				else:
					smaller_beat=bt2
					smaller_beat_typ=b2typ
					larger_beat=bt1
					larger_beat_typ=b1typ
				
			c1=0
			outs=set(comb_df['path_x'].unique())-{'Distributor'}
			for o in outs:
				if(o in outlets_allowed_forvan[b1]):
					c1=c1+1
			c2=0
			for o in outs:
				if(o in outlets_allowed_forvan[b2]):
					c2=c2+1
					
			if((c1==len(outs)) and (c2!=len(outs))):
				return bt1,bt2
			elif((c2==len(outs)) and (c1!=len(outs))):
				return bt2,bt1
			else:
				if((larger_beat.split('_')[-1]=='Rented') and (smaller_beat.split('_')[-1]!='Rented')):
						return smaller_beat,larger_beat
				return larger_beat,smaller_beat
				
		def check_if_both_comb_possible(bt1,bt2,comb_df):
				c1=0
				if(bt1.split('_')[-1]=='Rented'):
					bt1=bt1.split('_')[0]
				if(bt2.split('_')[-1]=='Rented'):
					bt2=bt2.split('_')[0]
				outs=set(comb_df['path_x'].unique())-{'Distributor'}
				
				for o in outs:
					if(o in outlets_allowed_forvan[bt1]):
						c1=c1+1
				c2=0
				for o in outs:
					if(o in outlets_allowed_forvan[bt2]):
						c2=c2+1
						
				if((c1==len(outs)) or (c2==len(outs))):
					return True
				else:
					return False
		
		def check_limits_complied2(workdf_cluster,datfr):
				datfr=pd.concat([datfr,subdf])
				if(len(workdf_cluster)<=0):
					return True
				van1=workdf_cluster['van_id'].unique()[0] 
				datfr=datfr[~((datfr['van_id'].isin(workdf_cluster['van_id'].unique())) & (datfr['path_x'].isin(workdf_cluster['path_x'].unique())))].copy(deep=True)
				workdf_cluster=resequence(workdf_cluster)
				if (van1.split('_')[-1]=='Rented'):
					van1=van1.split('_')[0]
				van1wgt=workdf_cluster['cum_weight'].unique()[0]
				van1tm=workdf_cluster['endtime'].unique()[0]
				van1vol=workdf_cluster['cum_volume'].unique()[0]
				van1nbills=workdf_cluster['num_bills'].unique()[0]
				van1endtime=0
				tot_endtime=0
				if(van1 in van_multitrip_dict.keys()):
					if((van_multitrip_dict[van1]=='yes') and (not((van_cutoff_dict[van1]=='yes') and (van_endtime_dict[van1]==480)))):
						if(van_cutoff_dict[van1]=='yes'):
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_3']
						else:
							van1endtime=van_endtime_dict['_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1])+'_1']
						datfr['van_id2']=datfr['van_id'].str.split('_').apply(lambda x:'_'.join(x[:-1]))
						for v in datfr['van_id'].unique():
							if(len(datfr[(datfr['van_id']==v) & (datfr['van_id2']=='_'.join(workdf_cluster['van_id'].unique()[0].split('_')[:-1]))])>0):
								tot_endtime=tot_endtime+datfr[(datfr['van_id']==v)]['endtime'].unique()[0]
				
					else:
						van1endtime=van_endtime_dict[van1]
				else:
					van1endtime=van_endtime_dict[van1]
					
				if((int(van1wgt)<=van_weight_dict[van1]) and (van1vol<=van_volume_dict[van1]) and (van1nbills<=van_bill_dict[van1]) and (van1tm+tot_endtime<=van1endtime)):
					return True  
				else:
					if(van1wgt>van_weight_dict[van1]):
						print('can cons breached')
					if((van1vol>van_volume_dict[van1])):
						print('vol cons breached')
					if((van1nbills>van_bill_dict[van1])):
						print('bills cons breached')
					if((van1tm+tot_endtime>van1endtime)):
						print('endtime breached')
					return False  
		
		def find_nearest_cluster(mindistout,bt,df,non_cluster_outs,cluster_outs):
				#print(len(non_cluster_outs))
				outcluster_mindistout=pd.Series(distance_matrix[mindistout][distance_matrix[mindistout].index.isin(set(non_cluster_outs)-set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor',mindistout})]).idxmin()
				return list(set(df[df['path_x']==outcluster_mindistout]['van_id'].unique())-{bt})[0]
				
		def find_nearest_cluster2(df2,v,options):
		   clstrouts=list(df2[df2['van_id']==v]['path_x'].unique()) 
		   nclstrouts=list(df2[df2['van_id'].isin(list(options))]['path_x'].unique()) 
		   if(len(clstrouts)==0):
			   return 'nil'
		   if(len(set(nclstrouts)-{'Distributor'})>1):
			   mindistoutclus=[]
			   for co in clstrouts:
				   mindistout=pd.Series(distance_matrix[co][distance_matrix[co].index.isin(set(nclstrouts)-{'Distributor'})]).idxmin()
				   mindistoutclus.append(list(set(df2[df2['path_x']==mindistout]['van_id'].unique()).intersection(options))[0])
			   mindistoutclus_cnt={}
			   for c in set(mindistoutclus):
				   mindistoutclus_cnt[c]=mindistoutclus.count(c)
						  
			   return max(mindistoutclus_cnt, key=mindistoutclus_cnt.get)
		   else:
			   return 'nil'
		
		
		under_uts_dict={}
		
		#new_tdf=pd.DataFrame()
		def calc_under_uts_per(df2,v,v2):
			wgt_uts=(van_weight_dict[vn2]-df2[df2['van_id']==vn]['weights'].sum())/van_weight_dict[vn2]
			vol_uts=(van_volume_dict[vn2]-df2[df2['van_id']==vn]['volumes'].sum())/van_volume_dict[vn2]
			bill_uts=(van_bill_dict[vn2]-df2[df2['van_id']==vn]['num_bills'].unique()[0])/van_bill_dict[vn2]
			time_uts=(van_endtime_dict[vn2]-df2[df2['van_id']==vn]['endtime'].unique()[0])/van_endtime_dict[vn2]
			return (wgt_uts+vol_uts+bill_uts+time_uts)/4    
		
		def find_low_cost_empty_van(df):
			rem_vans=[]
			for van in owned_van_order_tofill:
				if(van not in new_tdf['van_id'].unique()):
					if((van_multitrip_dict[van]=='yes') and (van_cutoff_dict[van]=='yes') and (van_endtime_dict[van]==480) and (van.split('_')[:-1] in [uv.split('_')[:-1] for uv in new_tdf['van_id'].unique()])):
						continue
					rem_vans.append(van)
			
			rem_van_cost={}
			for rv in rem_vans:
			   rem_van_cost[rv]=van_cost_dict[rv]
			   
			return min(rem_van_cost, key=rem_van_cost.get)  
		
		
		for vn in new_tdf['van_id'].unique():
			if(vn.split('_')[-1]=='Rented'):
				vn2=vn.split('_')[0]
				btyp='Rented'
			else:
				vn2=vn
				btyp='Owned'
		
			under_uts_dict[vn]=calc_under_uts_per(new_tdf,v,vn2)
		
		df=new_tdf.copy(deep=True)
		subdf=df[df['del_type'].isin(list(set([str(v) for v in van_orderexclu_dict.values()])-{'nan'}))].copy(deep=True)
		df=df[~df['del_type'].isin(list(set([str(v) for v in van_orderexclu_dict.values()])-{'nan'}))].copy(deep=True)
		
		new_tdf=pd.DataFrame()
		
		for v in {k: v for k, v in sorted(under_uts_dict.items(), key=lambda item: item[1],reverse=True)}.keys():
			already_visited=[]
			flg=False
			if(len(df[df['van_id']==v]['path_x'])>1):
				for i in range(len(set(df['van_id'].unique())-{v})):
					cluster=find_nearest_cluster2(df,v,set(df['van_id'].unique())-{v}-set(already_visited))
					if((cluster!='nil') and (not(flg))):
						already_visited.append(cluster)
						if(outlets_mixing_allowed(df[df['van_id']==v],df[df['van_id']==cluster])):
							cmb=resequence(df[df['van_id'].isin([v,cluster])])
							dff_initt=df.copy(deep=True)
							if(check_if_both_comb_possible(v,cluster,cmb)):
								van_name,other_beat=find_apt_van(cmb,v,cluster)
								print(van_name,other_beat)
								cmb['van_id']=van_name 
								df=df[~df['van_id'].isin([van_name,other_beat])]
								df=pd.concat([df,cmb])
								vis_beats=[]
								non_empty_vans=[]
								out_already_vis_in_beat={}
								for van in df['van_id'].unique():
									if((outlets_mixing_allowed(df[df['van_id']==van_name],df[df['van_id']==van])) and (check_if_both_comb_possible(van_name,van,resequence(df[df['van_id'].isin([van_name,van])])))):
										non_empty_vans.append(van)
								
								while((not(all_beats_complied(df))) and (not(len(set(vis_beats))==len(non_empty_vans)))):
									bt=find_non_compliance_beat(df)
									if(bt not in out_already_vis_in_beat.keys()):
										out_already_vis_in_beat[bt]=[]
									print(bt)
									vis_beats.append(bt)
									already_vis=[]
									df_init2=df.copy(deep=True)
									#prev_exchange_pairs=exchange_pairs.copy()
									exchange_pairs=[]
									restricted_bts=[bt]
									while(not(check_limits_complied2(df[df['van_id']==bt],df))):
										cluster_outs=list(set(df[df['van_id']==bt]['path_x'].unique())-{'Distributor'}-set(already_vis)-set(out_already_vis_in_beat[bt]))                
										non_cluster_outs=list(set(df[~df['van_id'].isin(restricted_bts)]['path_x'].unique())-{'Distributor'})                    
										if((len(cluster_outs)>0) and (len(non_cluster_outs)>0)):
											print('find outlet to be swapped')
											dm=distance_matrix[non_cluster_outs].T[cluster_outs].T 
											values=list(dm.min(axis=1))
											mindistout=list(dm.index)[values.index(min(values))]
											
											
											clus=find_nearest_cluster(mindistout,bt,df,non_cluster_outs,cluster_outs)
											
											if(clus.split('_')[-1]=='Rented'):
												clus2=clus.split('_')[0]
											else:
												clus2=clus
												
											if((mindistout in cluster_outs) and (mindistout in non_cluster_outs)):
												print('mindistout in both')
												if(not((check_limits_complied2(df[(df['van_id']==bt) & (df['path_x']!=mindistout)],df)) and (df[(df['van_id'].isin([bt,clus])) & (df['path_x']==mindistout)]['weights'].sum()<van_weight_dict[clus2]))):
													already_vis.append(mindistout)
													continue
											if(mindistout not in outlets_allowed_forvan[clus2]):
												print('mindistout not allowed')
												already_vis.append(mindistout)
												continue
											#if(((mindistout,bt) in prev_exchange_pairs) and (clus==prev_bt)):
					#                            if((clus==prev_bt)):
					#                                print('same as prev exchanges/beat')
					#                                already_vis.append(mindistout)
					#                                continue
											
											clus_l=[]
											for o in set(df[df['van_id']==clus]['path_x'].unique())-{'Distributor'}:
											   ncos=list(set(df[df['van_id']!=clus]['path_x'].unique())-{'Distributor'})                    
											   c=find_nearest_cluster(o,clus,df,ncos,list(set(df[df['van_id']==clus]['path_x'].unique())-{'Distributor'}))
											   clus_l.append(c)
											if((len(set(clus_l))==1) and (clus_l[0]==bt)):
												if(clus.split('_')[-1]=='Rented'):
													ctyp='Rented'
												else:
													ctyp='Owned'
												if(not(check_limits_complied2(pd.concat([df[df['van_id']==clus],df[(df['van_id']==bt) & (df['path_x']==mindistout)]]),df))):
													print('adding res beat',clus)
													restricted_bts.append(clus)
													continue
											
											
											bt_bt=df[df['van_id']==bt]
											if(not(outlets_mixing_allowed(df[df['van_id']==clus],bt_bt[bt_bt['path_x']==mindistout]))):
												print('outlets_mixing_not_allowed')
												already_vis.append(mindistout)
												continue
											print(mindistout,clus2) 
											exchange_pairs.append((mindistout,clus))
											if(clus not in out_already_vis_in_beat.keys()):
												out_already_vis_in_beat[clus]=[]
											out_already_vis_in_beat[clus].append(mindistout)
											new_bt=resequence(pd.concat([df[df['van_id']==clus],bt_bt[bt_bt['path_x']==mindistout]]))
											modi_bt=resequence(bt_bt[bt_bt['path_x']!=mindistout])
											sub_df=pd.concat([new_bt,modi_bt])
											df=pd.concat([df[~df['van_id'].isin([clus,bt])],sub_df])
											already_vis.append(mindistout)
										else:
											#flg=True
											print('clstrout empty hence using a empty van')
											print('ROLL_BACK')
											df=dff_initt.copy(deep=True)
					
										prev_bt=bt
										
									if((len(set(vis_beats))==len(non_empty_vans)) and (not(all_beats_complied(df)))):
										flg=True
										print('ROLL_BACK')
										df=dff_initt.copy(deep=True)
		
		new_tdf=pd.concat([new_tdf,df])  
		new_tdf=pd.concat([new_tdf,subdf]) 
		new_tdf=pd.concat([new_tdf,edf]) 
			
		new_tdf.to_csv(rscode+'_3v9.csv')
		
		def check_assign_underuts_owned(new_tdf):    
			b3=[]
			owned_vans_to_consider=[]
			eligible_vans_foreach_beat={}
			for v in owned_van_order_tofill:
				owned_vans_to_consider.append(v)
			
			b2index=0 
			mts={}
			mts_riv={}
			b_ind_dict={}
			beat_cost_dict={}
			for b in new_tdf['van_id'].unique():
				b_ind_dict[b2index]=b
				beat_cost_dict[b2index]={}
				eligible_vans_foreach_beat[b2index]=[b]
								
				if(b.split('_')[-1]=='Rented'):
					b2=b.split('_')[0]
					btyp='Rented'
				else:
					b2=b
					btyp='Owned'
				
				d={}
				d['path']=list(new_tdf[new_tdf['van_id']==b]['path_x'])
				d['cum_weight']=new_tdf[new_tdf['van_id']==b]['cum_weight'].unique()[0]
				d['end_time']=new_tdf[new_tdf['van_id']==b]['endtime'].unique()[0]
				d['cum_volume']=new_tdf[new_tdf['van_id']==b]['cum_volume'].unique()[0]
				d['bills']=new_tdf[new_tdf['van_id']==b]['num_bills'].unique()[0]
				
				if(van_fixedrate_dict[b2]=='yes'):
					beat_cost_dict[b2index][b]=van_cost_dict[b2]
				else:
					if(van_perkmrate_dict[b]==0):
					   beat_cost_dict[b2index][b]=van_baserate_dict[b2]+(d['end_time']/60)*van_perhourrate_dict[b2]
					else:
					   beat_cost_dict[b2index][b]=van_baserate_dict[b2]+(d['end_time']*(1/van_speed_dict[b2]))*van_perkmrate_dict[b2] 
		
				for v in owned_vans_to_consider:
					lr=False
					for l in set(input_data[input_data['partyhll_code'].isin(list(new_tdf[new_tdf['van_id']==b]['path_x']))]['lane_restrictions'].unique())-{10000}:
						if(van_weight_dict[v]>=l):
						  lr=True
						  break
					if((d['cum_weight']<=van_weight_dict[v]) and (d['end_time']<=van_endtime_dict[v]) and (d['bills']<=van_outlet_dict[v]) and (d['cum_volume']<=van_volume_dict[v]) and (v!=b2) and (not(lr))):
						if(len(set(d['path']).intersection(set(outlets_allowed_forvan[b2])))==len(set(d['path'])-{'Distributor'})):
							if(str(van_areassign_dict[v])!='nan'):
								if(len(set(input_data[input_data['partyhll_code'].isin(d['path'])]['area_name']).intersection({van_areassign_dict[v]}))<1):
									continue
							if(str(van_orderexclu_dict[v])!='nan'):
								if(len(set(input_data[input_data['partyhll_code'].isin(d['path'])]['order_type']).intersection({van_orderexclu_dict[v]}))<1):
									continue
							eligible_vans_foreach_beat[b2index].append(v)
							if(van_fixedrate_dict[v]=='yes'):
									beat_cost_dict[b2index][v]=van_cost_dict[v]
							else:
								if(van_perkmrate_dict[v]==0):
								   beat_cost_dict[b2index][v]=van_baserate_dict[v]+(d['end_time']/60)*van_perhourrate_dict[v]
								else:
								   beat_cost_dict[b2index][v]=van_baserate_dict[v]+(d['end_time']*(1/van_speed_dict[v]))*van_perkmrate_dict[v] 
		
				
				b2index=b2index+1  
			
			for v in owned_vans_to_consider:    
				if((van_multitrip_dict[v]=='yes') and (int(v.split('_')[-1])==1)):
					mts[v]=[]
					k=v
					if((van_cutoff_dict[v]=='yes')):
					   mts_riv[v]=[] 
				elif((van_multitrip_dict[v]=='yes') and (int(v.split('_')[-1])>1)):
					if(not((van_cutoff_dict[v]=='yes') and (int(v.split('_')[-1])==3))):
						if('_'.join(v.split('_')[:-1])+'_1' in mts.keys()):
							mts['_'.join(v.split('_')[:-1])+'_1'].append(v)
					else:
						mts_riv['_'.join(v.split('_')[:-1])+'_1'].append(v)
				
				
			
			bv_var= pulp.LpVariable.dicts("beat van ",((bi,v) for bi in eligible_vans_foreach_beat.keys() for v in eligible_vans_foreach_beat[bi]),lowBound=0,upBound=1,cat='Binary')
			model1 = pulp.LpProblem("cost", pulp.LpMinimize)
			model1 += pulp.lpSum([bv_var[(bi,v)]*beat_cost_dict[bi][v] for bi in eligible_vans_foreach_beat.keys() for v in eligible_vans_foreach_beat[bi]])    
		
			for bi in eligible_vans_foreach_beat.keys():
				model1 += pulp.lpSum([bv_var[(bi,v)] for v in eligible_vans_foreach_beat[bi]])==1
			
			for ov in owned_van_order_tofill:
				model1 += pulp.lpSum([bv_var[(bi,v)] for bi in eligible_vans_foreach_beat.keys() for v in eligible_vans_foreach_beat[bi] if ov==v])<=1
				
			for k in mts.keys():
				for b in mts[k]:
					model1+=pulp.lpSum([bv_var[(bi,k)] for bi in eligible_vans_foreach_beat.keys() if k in eligible_vans_foreach_beat[bi]])-pulp.lpSum([bv_var[(bi,b)] for bi in eligible_vans_foreach_beat.keys() if b in eligible_vans_foreach_beat[bi]])>=0
			
			for k in mts_riv.keys():
				for b in mts_riv[k]:
					model1+=pulp.lpSum([bv_var[(bi,k)] for bi in eligible_vans_foreach_beat.keys() if k in eligible_vans_foreach_beat[bi]])+pulp.lpSum([bv_var[(bi,b)] for bi in eligible_vans_foreach_beat.keys() if b in eligible_vans_foreach_beat[bi]])==1
		
			result=model1.solve(pulp.PULP_CBC_CMD(maxSeconds=100))  
			#print(result)            
			ntdf=pd.DataFrame()
			if(result==1):
				for bi in eligible_vans_foreach_beat.keys():
					for v in eligible_vans_foreach_beat[bi]:
						if(bv_var[bi,v].varValue==1):
							print(b_ind_dict[bi],v)
							sdf=new_tdf[new_tdf['van_id']==b_ind_dict[bi]]
							sdf['van_id']=v
							ntdf=pd.concat([sdf,ntdf])
			else:
				return new_tdf
						
			return ntdf
					
		
		print('REARRANGE---------------------------------------')
		
		new_tdf=check_assign_underuts_owned(new_tdf)
		new_tdf.to_csv(rscode+'_4v9.csv')
		
		
		def resequence_beat_for_sw(bt,count,buf):
			van_id=bt['van_id']
			bt_new={}              
			sequence=[]
			speed=1/van_speed_dict[van_id]
			end_time_seq=0
			sequence.append('Distributor') 
			#mids=ids.copy() 
			path = ['Distributor']
			ids=set(bt['sequence'])-{'Distributor'}
			c=1
			it=1
			modified_ids = list(ids)
			bt_new['sequence']=['Distributor']
			bt_new['end_time']=0
			bt_new['bills']=0
			bt_new['cum_weight']=0
			bt_new['van_id']=bt['van_id']
			bt_new['del_type']=bt['del_type']
			bt_new['wgt_sequence']=[0]
			bt_new['cum_volume']=0
			bt_new['vol_sequence']=[0]
			bt_new['time_sequence']=[buf]
			bt_new['bills_cov']=[['']]
			bt_new['base_pack_cov']=[['']]
			bt_new['plg']=[['']]
			bt_new['channel']=[['']]    
			for i in range(0,len(modified_ids)):
				dist = distance_matrix[path[-1]][distance_matrix[path[-1]].index.isin(modified_ids)]
				dist = pd.Series(dist)
				nearest_index = dist.idxmin()
				while((c<count) & (it==1)):
					dist=dist.drop(nearest_index)
					nearest_index = dist.idxmin()
					c=c+1
				it=100
				 
				for i in range(1,len(bt['sequence'])):
					if(bt['sequence'][i]==nearest_index):
						distance=distance_matrix[bt['sequence'][i-1]][bt['sequence'][i]]
						service_time=bt['time_sequence'][i]-math.ceil(distance * speed)
						wgt=bt['wgt_sequence'][i]
						vol=bt['vol_sequence'][i]
						bc=bt['bills_cov'][i]
						bpc=bt['base_pack_cov'][i]
						ps=bt['plg'][i]
						cs=bt['channel'][i]
						break
				bt_new['bills_cov'].append(bc)
				bt_new['base_pack_cov'].append(bpc)
				bt_new['plg'].append(ps)
				bt_new['channel'].append(cs)
				bt_new['wgt_sequence'].append(wgt)
				bt_new['vol_sequence'].append(vol)
				distance = dist[nearest_index]
				end_time_seq += service_time
				travel_time = math.ceil(distance * speed)
				end_time_seq += travel_time
				bt_new['time_sequence'].append(travel_time+service_time)
				bt_new['bills'] = bt_new['bills'] + len(bc)
				path.append(nearest_index)
				bt_new['sequence'].append(nearest_index)
				modified_ids.remove(nearest_index)
				bt_new['end_time']=bt_new['end_time']+end_time_seq
				bt_new['cum_weight']=bt_new['cum_weight']+wgt
				bt_new['cum_volume']=bt_new['cum_volume']+vol
			
			
			bt_new['bills_cov'].append([''])
			bt_new['base_pack_cov'].append([''])
			bt_new['plg'].append([''])
			bt_new['channel'].append([''])
			bt_new['wgt_sequence'].append(0)
			bt_new['vol_sequence'].append(0)
			bt_new['time_sequence'].append(math.ceil(distance_matrix[bt_new['sequence'][-1]]['Distributor'] * speed))
			bt_new['sequence'].append('Distributor')
			bt_new['end_time']=bt_new['end_time']+math.ceil(distance_matrix[bt_new['sequence'][-2]]['Distributor'] * speed)
		
			return bt_new
		
		def check_for_sw_overlap(bt):
			ots_w_srwndw=list(set(bt['sequence']).intersection(set(ol_closure_time.keys())))
			if(len(ots_w_srwndw)>0):
				for i in range(0,len(bt['sequence'])):
					if(bt['sequence'][i] in ots_w_srwndw):
						print(i)
						#print(bt['time_sequence'][i],ol_closure_time[bt['sequence'][i]][0],bt['time_sequence'][i],ol_closure_time[bt['sequence'][i]][1])
						if(((sum(bt['time_sequence'][:i]) > ol_closure_time[bt['sequence'][i]][0]) and (sum(bt['time_sequence'][:i]) < ol_closure_time[bt['sequence'][i]][1])) or ((sum(bt['time_sequence'][:i+1])>ol_closure_time[bt['sequence'][i]][0]) and (sum(bt['time_sequence'][:i+1]) < ol_closure_time[bt['sequence'][i]][1]))):
							print(sum(bt['time_sequence'][:i]),ol_closure_time[bt['sequence'][i]][0],sum(bt['time_sequence'][:i+1]),ol_closure_time[bt['sequence'][i]][1])
							return True
			return False
		
		beat_list5=[]
		for b in new_tdf['van_id'].unique():
			min_beat={}
			partdf=new_tdf[new_tdf['van_id']==b]
			min_beat['sequence']=list(partdf['path_x'])
			min_beat['end_time']=partdf['endtime'].unique()[0]
			min_beat['bills']=partdf['num_bills'].unique()[0]
			min_beat['cum_weight']=partdf['cum_weight'].unique()[0]
			min_beat['van_id']=partdf['van_id'].unique()[0]
			min_beat['del_type']=partdf['del_type'].unique()[0]
			min_beat['wgt_sequence']=list(partdf['weights'])
			min_beat['cum_volume']=partdf['cum_volume'].unique()[0]
			min_beat['vol_sequence']=list(partdf['volumes'])
			min_beat['time_sequence']=list(partdf['time'])
			min_beat['bills_cov']=list(partdf['bill_numbers'])
			min_beat['base_pack_cov']=list(partdf['Basepack'])
			min_beat['plg']=list(partdf['plg'])
			min_beat['channel']=list(partdf['channel'])
			beat_list5.append(min_beat)
			
		beat_list6=[]
		for bt in beat_list5:
			count=1
			bt_init=copy.deepcopy(bt)
			buffer=0
			j=0
			while(check_for_sw_overlap(bt)):
				if((count>=len(bt['sequence'])-1) or (j>0)):
					j=j+1
					if(j==1):
						print(bt['van_id'],'add buffer')
						bt=copy.deepcopy(bt_init)
						count=1
					#list(set(bt['sequence']).intersection(set(ol_closure_time.keys())))[0]
					#buffer=ol_closure_time[list(set(bt['sequence']).intersection(set(ol_closure_time.keys())))[0]][1]
					#for i in range(0,len(bt['sequence'])):
					buffer=buffer+30
					bt['time_sequence'][0]=bt['time_sequence'][0]+buffer
					#break
				bt=resequence_beat_for_sw(bt,count,buffer)
				count=count+1
			
			beat_list6.append(bt)
		
		print(beat_list6)
		output_df=pd.DataFrame()
		for b in beat_list6:
			add_to_output(b)
			
		lat_long['path']=lat_long.index    
		print(output_df.shape)
		print(output_df.columns)
		output_df2 = pd.merge(output_df, lat_long, left_on = 'path', right_on = 'partyhll_code', how = 'left') 
		output_df2.to_csv(rscode+'_5v9.csv')
		
		import datetime as dt
		from datetime import datetime
		
		details_sheet=pd.read_csv(rscode+'_5v9.csv')
		details_sheet.drop_duplicates(subset =['van_id','path_x'], keep = 'first', inplace = True)
		details_sheet['van_id_1']=details_sheet['van_id'].apply(lambda x:x.split('_')[0])
		van_wth_grt_thre=details_sheet.groupby(['van_id_1'])['van_id'].nunique()[details_sheet.groupby(['van_id_1'])['van_id'].nunique()==1].index
		
		
		for i in van_wth_grt_thre:
			van=details_sheet[details_sheet['van_id_1']==i]['van_id'].unique()[0]
			if (len(van.split('_'))>1) & (van.split('_')[-1]!='Rented'):
				if int(van.split('_')[-1])==3:
					van1=van.split('_')[0]+'_'+'1'
					details_sheet.loc[details_sheet['van_id_1']==i,'van_id']=van1
			
		
		transaction_input= pd.read_excel(transfilename,sheet_name='transactional_data')
			
		van_details = pd.read_excel(masterfilename, sheet_name = 'vehicle_master')
		rs_master = pd.read_excel(masterfilename, sheet_name = 'rental_vehicle_master')
		sku_master = pd.read_excel(masterfilename,sheet_name='product_master')
		outlet_data = pd.read_excel(masterfilename,sheet_name='hllcode_master_constraint')
		outlet_master = pd.read_excel(masterfilename,sheet_name='outlet_master')
		service_time_details = pd.read_excel(masterfilename,sheet_name='service_time')
		party_packing_details = pd.read_excel(masterfilename,sheet_name='party_packing_details')
		plg_clubbing = pd.read_excel(masterfilename,sheet_name='clubbing_details')
		
		transaction_input_v1=transaction_input.copy(deep=True)
		party_pack=party_packing_details['party_packing'].unique()[0]

		details_sheet.rename(columns={'partyhll_code':'path'},inplace=True)
		ssdf=details_sheet[details_sheet['path'] == 'Distributor'].copy(deep=True)
		buffer_time_vandict=dict(zip(ssdf['van_id'],ssdf['time']))
		
		sku_master['volume'] = sku_master['unit_length'] * sku_master['unit_breadth'] * sku_master['unit_height']
		
		details_sheet = details_sheet[details_sheet['path'] != 'Distributor'].reset_index(drop = True)
		details_sheet['bill_numbers_org'] = details_sheet['bill_numbers']
		s = details_sheet['bill_numbers'].str.split(',').apply(pd.Series, 1).stack()
		del details_sheet['bill_numbers']
		s.index = s.index.droplevel(-1)
		s.name = 'bill_numbers'
		details_sheet = details_sheet.join(s)
		print(details_sheet.shape)
		details_sheet['bill_numbers'] = details_sheet['bill_numbers'].str.replace('}', '').str.replace('{', '').str.replace('[', '').str.replace(']', '').str.replace("'", "").str.replace(" ", "")
		
		details_sheet.shape
		
		if party_pack=='yes':
			details_sheet['Basepack']= details_sheet['Basepack'].apply(lambda y:list(map(lambda x: (x.replace('[', '').replace(']', '').replace(' ', '').replace('{','').replace('}','').replace("'",'')), y.split(','))))
		   
		else:
			details_sheet['Basepack'] = details_sheet['Basepack'].apply(lambda y:list(map(lambda x: int(x.replace('[', '').replace(']', '').replace('{','').replace('}','').replace("'", "").replace(" ", "")), y.split(','))))
			
		
		if party_pack=='yes':
			
			transaction_input_v1['crate_no']=transaction_input_v1['crate_no'].str.lower()
			party_packing_details['crate_type']=party_packing_details['crate_type'].str.lower()
			transaction_input_v1['basepack_code']=transaction_input_v1['basepack_code'].astype(str)
			transaction_input_v1['crate_no'] = transaction_input_v1['crate_no'].astype(str).str.lower()
			transaction_input_v1.loc[transaction_input_v1['basepack_code']=='-','basepack_code']=transaction_input_v1['crate_no']    
			transaction_input_v1['crate_no']=transaction_input_v1['crate_no'].str[:7]
			party_packing_details['crate_type']=party_packing_details['crate_type'].str[:7]
			
			len_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_length'])))
			wid_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_width'])))
			h_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_height'])))
			wt_dict=dict(list(zip(party_packing_details['crate_type'],party_packing_details['crate_weight'])))
			#transaction_input_v1['new_weight']=transaction_input_v1['crate_no'].map(wt_dict)
			transaction_input_v1['length']=transaction_input_v1['crate_no'].map(len_dict)
			transaction_input_v1['width']=transaction_input_v1['crate_no'].map(wid_dict)
			transaction_input_v1['height']=transaction_input_v1['crate_no'].map(h_dict)
			transaction_input_v1['length']=transaction_input_v1['length']*30
			transaction_input_v1['width']=transaction_input_v1['width']*30
			transaction_input_v1['height']=transaction_input_v1['height']*30
			transaction_input_v1['weight']=transaction_input_v1['weight'].astype(float)
			transaction_input_v2=transaction_input_v1[transaction_input_v1['crate_no']=='nan'].reset_index(drop=True)
			transaction_input_v3=transaction_input_v1[transaction_input_v1['crate_no']!='nan'].reset_index(drop=True)
			transaction_input_v3['new_weight']=transaction_input_v3['crate_no'].map(wt_dict)
			transaction_input_v2['new_weight']=0
			transaction_input_v3['new_weight']=transaction_input_v3['new_weight']/1000
			transaction_input_v3['multi_fact']=transaction_input_v3['weight']/transaction_input_v3['new_weight']
			transaction_input_v2.loc[transaction_input_v2['crate_no'].isna(),'crate_no']=transaction_input_v2['basepack_code']
			transaction_input_v2['multi_fact']=1
			sku_master['basepack_code']=sku_master['basepack_code'].astype(str)
			sku_len_dict=dict(list(zip(sku_master['basepack_code'],sku_master['unit_length'])))
			sku_w_dict=dict(list(zip(sku_master['basepack_code'],sku_master['unit_breadth'])))
			sku_h_dict=dict(list(zip(sku_master['basepack_code'],sku_master['unit_height'])))
			transaction_input_v2['length']=transaction_input_v2['basepack_code'].map(sku_len_dict)
			transaction_input_v2['width']=transaction_input_v2['basepack_code'].map(sku_w_dict)
			transaction_input_v2['height']=transaction_input_v2['basepack_code'].map(sku_h_dict)
			transaction_input_v1=pd.concat([transaction_input_v2,transaction_input_v3]).reset_index(drop=True)
			transaction_input_v1['multi_fact']=transaction_input_v1['multi_fact'].fillna(0)
			transaction_input_v1['multi_fact']=np.where(transaction_input_v1['multi_fact']<=1,1,transaction_input_v1['multi_fact'])
			transaction_input_v1['multi_fact']=transaction_input_v1['multi_fact'].apply(np.floor)
			transaction_input_v1['multi_fact']=1
						
		
		'''
		if party_pack=='yes':
			input_data['PARTY_HLL_CODE']=input_data['partyhll_code']
			input_data['BASEPACK CODE']=input_data['basepack_code']
			input_data['SERVICING PLG']=input_data['servicing_plg']
			input_data['BILL_NUMBER']=input_data['bill_number']
			input_data['NET_SALES_WEIGHT_IN_KGS']=input_data['weight']
			input_data['NET_SALES_QTY']=0
			di={'DETS-S':'DETS','DF-S':'D+F', 'PP-S':'PP', 'PPB-S':'PP-B'}
			input_data=input_data.replace({"SERVICING PLG": di}).copy(deep=True)
			input_data=input_data[~input_data["SERVICING PLG"].isna()].copy(deep=True)
			input_data['BASEPACK CODE']=input_data['BASEPACK CODE'].astype(str)
		else:
			
			input_data['PARTY_HLL_CODE']=input_data['partyhll_code']
			input_data['BASEPACK CODE']=input_data['basepack_code']
			input_data['SERVICING PLG']=input_data['servicing_plg']
			input_data['BILL_NUMBER']=input_data['bill_number']
			input_data['NET_SALES_WEIGHT_IN_KGS']=input_data['net_sales_weight_kgs']
			input_data['NET_SALES_QTY']=input_data['net_sales_qty']
			di={'DETS-S':'DETS','DF-S':'D+F', 'PP-S':'PP', 'PPB-S':'PP-B'}
			input_data=input_data.replace({"SERVICING PLG": di}).copy(deep=True)
			input_data=input_data[~input_data["SERVICING PLG"].isna()].copy(deep=True)
			input_data['BASEPACK CODE']=input_data['BASEPACK CODE'].astype(str)
		'''        
		
				
		
		if(party_pack=='yes'):        
			#transaction_input_v1['NET_SALES_WEIGHT_IN_KGS']=input_data['weight']
			transaction_input_v1['net_sales_qty']=1
			transaction_input_v1['volume']=transaction_input_v1['length']*transaction_input_v1['width']*transaction_input_v1['height']*transaction_input_v1['multi_fact']
		
		transaction_input_v1.dtypes
		details_sheet['bill_numbers']=details_sheet['bill_numbers'].astype(str)
		bill_weights = {}
		bill_volumes = {}
		for bill in details_sheet['bill_numbers'].unique():
			if party_pack=='yes':
				
				basepack_codes = details_sheet[details_sheet['bill_numbers'] == bill]['Basepack']
				tot_vol=0
				
				for basepack in list(basepack_codes)[0]:
					volum=transaction_input_v1[(transaction_input_v1['basepack_code']==basepack) & (transaction_input_v1['bill_number']==bill)]['volume'].sum()
					print(volum,basepack)
					tot_vol+=volum
				
				weight = transaction_input_v1[(transaction_input_v1['bill_number'] == bill) & (transaction_input_v1['basepack_code'].isin(list(basepack_codes)[0]))]['weight'].sum()
				print(weight)
				bill_weights[bill] = weight
				
				#vol = transaction_input_v1[transaction_input_v1['basepack_code'].isin(list(basepack_codes)[0])]['volume'].sum()
				bill_volumes[bill] = tot_vol
			else:
				basepack_codes = details_sheet[details_sheet['bill_numbers'] == bill]['Basepack']
				tot_vol=0
				for basepack in list(basepack_codes)[0]:
					
					volum=sku_master[sku_master['basepack_code'].isin([basepack])]['volume'].sum()
					
					net_sales=transaction_input_v1[(transaction_input_v1['bill_number'] == bill) & (transaction_input_v1['basepack_code']==basepack)]['net_sales_qty'].sum()
					vol=volum*net_sales
					tot_vol+=vol
					
					
				weight = transaction_input_v1[(transaction_input_v1['bill_number'] == bill) & (transaction_input_v1['basepack_code'].isin(list(basepack_codes)[0]))]['net_sales_weight_kgs'].sum()
				bill_weights[bill] = weight
				#vol = sku_master[sku_master['basepack_code'].isin(list(basepack_codes)[0])]['volume'].sum()
				bill_volumes[bill] = tot_vol
				
			
		
		details_sheet['bill_weight']= details_sheet['bill_numbers'].map(bill_weights)
		details_sheet['bill_volume'] = details_sheet['bill_numbers'].map(bill_volumes)
		
		
		details_sheet['bill_weight'].sum().sum()
		details_sheet.groupby(['bill_numbers'])['bill_volume'].sum().sum()
		
		def check_multitrip_trip_id(van_name):
			if(van_name[-2] == '_'):
				return van_name[-1]
			else:
				return 1
		def check_multitrip_van_name(van_name):
			if(van_name[-2] == '_'):
				return van_name[:-2]
			else:
				return van_name
		
		
		details_sheet['vehicle name'] = details_sheet['van_id'].apply(check_multitrip_van_name)
		details_sheet['trip_sequence'] = details_sheet['van_id'].apply(check_multitrip_trip_id)
		details_sheet['trip_id'] = 'trip_' + details_sheet['vehicle name']+'_'+details_sheet['trip_sequence'].apply(str)
		
		
		vehicle_type_map = dict(zip(van_details['vehicle_name'], van_details['vehicle_type']))
		details_sheet['vehicle_type'] = details_sheet['vehicle name'].map(vehicle_type_map)
		details_sheet.loc[details_sheet['vehicle_type'].isna(),'vehicle_type']=details_sheet[details_sheet['vehicle_type'].isna()]['vehicle name'].apply(lambda x:x.split('_')[0])
		
		
		vehicle_rental_type_map = dict(zip(van_details['vehicle_name'], van_details['rental_type']))
		details_sheet['rental_type'] = details_sheet['vehicle name'].map(vehicle_rental_type_map)
		
		details_sheet.loc[details_sheet['rental_type'].isna(),'rental_type']='rented'
		
		# if trip id is resolved use commented code
		vehicle_wgt_map = details_sheet.groupby(['vehicle name', 'trip_id'])['cum_weight', 'cum_volume'].max().reset_index()
		# vehicle_wgt_map = details_sheet.groupby('van_id')['cum_weight', 'cum_volume'].max().reset_index()
		
		van_details['max_volume'] = van_details['vehicle_dimensions_length'] * van_details['vehicle_dimensions_breadth'] * van_details['vehicle_dimensions_height'] * 27000
		rs_master['max_volume'] = rs_master['vehicle_dimensions_length'] * rs_master['vehicle_dimensions_breadth'] * rs_master['vehicle_dimensions_height'] * 27000
		
		
		rent_max_volume=dict(zip(rs_master['rental_vehicle_capacity_kgs'].astype(str),rs_master['max_volume']))
		
		
		vehicle_wgt_map= pd.merge(vehicle_wgt_map, van_details[['vehicle_name', 'vehicle_capacity_kgs', 'max_volume']], left_on = 'vehicle name', right_on = 'vehicle_name', how = 'left')
		
		vehicle_wgt_map_v2=vehicle_wgt_map[~vehicle_wgt_map['max_volume'].isna()].reset_index(drop=True)
		vehicle_wgt_map_v1=vehicle_wgt_map[vehicle_wgt_map['max_volume'].isna()].reset_index(drop=True)
		
		
		vehicle_wgt_map_v1.isna().sum()
		
		vehicle_wgt_map_v1['vehicle_name']=vehicle_wgt_map_v1['vehicle name']
		vehicle_wgt_map_v1['vehicle_capacity_kgs']=vehicle_wgt_map_v1['vehicle name'].apply(lambda x:x.split('_')[0])
		
		
		vehicle_wgt_map_v1['max_volume']=vehicle_wgt_map_v1['vehicle_capacity_kgs'].map(rent_max_volume)
		
		vehicle_wgt_map=pd.concat([vehicle_wgt_map_v2,vehicle_wgt_map_v1]).reset_index(drop=True)
		vehicle_wgt_map['vehicle_capacity_kgs']=vehicle_wgt_map['vehicle_capacity_kgs'].astype(int)
		
		vehicle_wgt_map['max_volume']=vehicle_wgt_map['max_volume'].astype(float)
		vehicle_wgt_map['vehicle_utilised_weight'] = round(((vehicle_wgt_map['cum_weight']/ vehicle_wgt_map['vehicle_capacity_kgs'].astype(float)) * 100), 2)
		vehicle_wgt_map['vehicle_utilised_volume'] = round(((vehicle_wgt_map['cum_volume']/ vehicle_wgt_map['max_volume']) * 100), 2)
		
		
		details_sheet = pd.merge(details_sheet, vehicle_wgt_map[['vehicle_name', 'trip_id', 'vehicle_capacity_kgs', 'max_volume', 'vehicle_utilised_weight', 'vehicle_utilised_volume']], left_on = ['vehicle name', 'trip_id'], right_on = ['vehicle_name', 'trip_id'], how = 'left')
		details_sheet.rename(columns = {'max_volume' : 'vehicle_capacity_volume', 'vehicle_utilised_weight' : 'perc_utilization_weight', 'vehicle_utilised_volume' : 'perc_utilization_volume'}, inplace = True)
		
		details_sheet .isna().sum()
		
		
		service_time_details['load_range_to_kgs'] = service_time_details['load_range_to_kgs'].replace(['above'],100000)
		
		ol_weights = details_sheet.groupby(['van_id','path'])['weights'].max().reset_index()
		
		
		
		def find_service_time(value):
			#print(value)
			#print(value,service_time_details[(value >= service_time_details['load_range_from_kgs'].astype(float)) & (value< service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'])
			if(value==0):
				return 0
			return service_time_details[(value> service_time_details['load_range_from_kgs'].astype(float)) & (value<= service_time_details['load_range_to_kgs'].astype(float))]['service_time_in_minutes'].iloc[0]
		
		
		ol_weights['service_time_minutes'] = ol_weights['weights'].apply(lambda x: find_service_time(x))
		
		
		#ol_weights.to_csv('ol_weights.csv')
		ol_weights['comb']=ol_weights['van_id']+'_'+ol_weights['path']
		
		
		details_sheet = pd.merge(details_sheet, ol_weights[['van_id','path','comb','service_time_minutes']], left_on = ['van_id','path'],right_on=['van_id','path'], how = 'left')
		
		
		#details_sheet.to_csv('details_sheet.csv')
		
		service_time_dict = dict(zip(ol_weights['comb'], ol_weights['service_time_minutes']))
		# start = dt.time(9, 0, 0)
		# details_sheet['end_time_minutes'] = details_sheet['endtime'].apply(lambda x: (dt.datetime.combine(dt.date(1,1,1), start) + dt.timedelta(minutes = x)).time())
		#need service time dict
		# details_sheet['start_time'] = details_sheet['endtime'] - details_sheet['service_time_minutes']
		# details_sheet['start_time'] = details_sheet['start_time'].apply(lambda x: (dt.datetime.combine(dt.date(1,1,1), start) + dt.timedelta(minutes = x)).time())
		
		#details_sheet = pd.merge(details_sheet, transaction_input_v1[['bill_number', 'beat_name','fresh/old', 'bill_date','priority_flag', 'servicing_plg', 'party_code', 'party_name', 'order_number', 'order_date', 'smn_tagged', 'order_type']], left_on = 'bill_numbers', right_on = 'bill_number', how = 'left')
		
		
		if(outlet_data['delivery_expectation_from_bill_date'].iloc[0] == 'N+1'):
			# delivery_date = transaction_input['bill_date'].max() + dt.timedelta(days = 1)
			# while(delivery_date.day in outlet_data['monthly_holiday'].unique()):
			#     delivery_date = delivery_date + dt.timedelta(days = 1)
			details_sheet['execution_date'] = transaction_input_v1['bill_date'].max() + dt.timedelta(days = 1)
		elif (outlet_data['delivery_expectation_from_bill_date'].iloc[0] == 'N+2'):
			details_sheet['execution_date'] = transaction_input_v1['bill_date'].max() + dt.timedelta(days = 2)
			
		else:
			details_sheet['execution_date'] = transaction_input_v1['bill_date'].max() + dt.timedelta(days = 0)
		
		ol_type_dict = dict(zip(outlet_master['partyhll_code'], outlet_master['outlet_type']))
		area_dict = dict(zip(outlet_master['partyhll_code'], outlet_master['area_name']))
		details_sheet['outlet_type'] = details_sheet['path'].map(ol_type_dict)
		details_sheet['area_name'] = details_sheet['path'].map(area_dict)
		details_sheet['cluster_name'] = np.NaN
		
		exclusive_dict = dict(zip(outlet_data['partyhll_code'], outlet_data['exclusivity']))
		details_sheet['exclusivity'] = details_sheet['path'].map(exclusive_dict)
		
		
		if(party_packing_details['party_packing'].iloc[0].lower() == 'yes'):
			#pass
			details_sheet['pb']=details_sheet['path']+details_sheet['bill_numbers'].astype(str)
			transaction_input_v1['pb']=transaction_input_v1['partyhll_code']+transaction_input_v1['bill_number'].astype(str)
			details_sheet['no_of_crates'] = details_sheet['pb'].map(dict(transaction_input_v1.groupby(['pb'])['crate_no'].nunique()))
		else:
			details_sheet['no_of_crates'] = np.NaN
			
		#if(not(party_packing_details['party_packing'].iloc[0].lower() == 'yes')):
		#    case_map=dict(zip(sku_master['basepack_code'],sku_master['case_config']))
		#    transaction_input_v1['cases']=transaction_input_v1['basepack_code'].map(case_map)
		#    transaction_input_v1['pb']=transaction_input_v1['partyhll_code']+transaction_input_v1['basepack_code'].astype(str)
		#    pbsalesmap=transaction_input_v1.groupby(['pb'])['net_sales_qty'].sum().to_dict()
		#    pbcasemap=transaction_input_v1.groupby(['pb'])['cases'].unique().to_dict()
		#    noofcases={}
		#    for pb in pbsalesmap.keys():
		#        if((str(pbcasemap[pb][0])!='nan') and (str(pbsalesmap[pb])!='nan')):
		#            noofcases[pb]=pbsalesmap[pb]/pbcasemap[pb][0]
		#        else:
		#            noofcases[pb]=np.NaN
		#            
		#    s = details_sheet.apply(lambda x: pd.Series(x['Basepack']),axis=1).stack().reset_index(level=1, drop=True)
		#    s.name = 'Basepack2'
		#    details_sheet=details_sheet.join(s).copy(deep=True)
		#    details_sheet['pb']=details_sheet['path']+details_sheet['Basepack2'].astype(str)
		#    details_sheet['no_of_cases']=details_sheet['pb'].map(noofcases)
		#else:
		#    details_sheet['no_of_cases'] = np.NaN 
			
		details_sheet['no_of_cases'] = np.NaN 
		
		transaction_input_v1=transaction_input_v1.reset_index(drop=True)   
		details_sheet['rs_code'] = transaction_input_v1['rs_code'].iloc[0]
		details_sheet['rs_name'] = transaction_input_v1['rs_name'].iloc[0]
		
			
		
		
		'''
		familiarity
		bill_weight
		bill_volume
		travel_time
		TODO
		'''
		
		details_sheet['familiarity'] = np.NaN
		
		del details_sheet['Basepack']
		
		
		details_sheet.drop_duplicates(inplace = True)
		details_sheet.reset_index(drop = True, inplace = True)
		details_sheet.columns
		
		details_sheet.columns
		rename_dict = {
			'path' : 'partyhll_code',
			'bill_numbers':'bill_number',
			'perc_utilization_weight' : 'vehicle_utilised_weight',
			'perc_utilization_volume' : 'vehicle_utilised_volume'
			}
		details_sheet_1 = details_sheet.rename(columns = rename_dict)
		
		
		col_seq = ['rs_code', 'van_id','rs_name', 'partyhll_code',   'bill_number',
		 'outlet_type', 'cluster_name', 'area_name',  'execution_date', 
		  'bill_weight', 'bill_volume', 'no_of_crates', 'no_of_cases', 'vehicle name', 
		 'vehicle_utilised_weight', 'vehicle_utilised_volume', 'trip_id', 
		 'trip_sequence', 'familiarity', 'exclusivity']
		
		'''
		col_seq = ['rs_code', 'rs_name', 'beat_name', 'partyhll_code', 'party_code', 'servicing_plg', 
		 'party_name', 'outlet_type', 'cluster_name', 'area_name', 'bill_number', 'bill_date', 
		 'execution_date', 'fresh/old', 'priority_flag', 'order_number', 'order_date', 'smn_tagged', 
		 'order_type', 'bill_weight', 'bill_volume', 'no_of_crates', 'no_of_cases', 'vehicle name', 
		 'vehicle_utilised_weight', 'vehicle_utilised_volume', 'trip_id', 
		 'trip_sequence', 'familiarity', 'exclusivity']
		'''
		details_sheet_1 = details_sheet_1[col_seq]
		
		details_sheet_1.drop_duplicates(inplace = True)
		
		
		details_sheet_1 = details_sheet_1.loc[:,~details_sheet_1.columns.duplicated()]
		
		
		details_sheet_final = pd.DataFrame()
		for vehicle in details_sheet_1['vehicle name'].unique():
			vehicle_df = details_sheet_1[details_sheet_1['vehicle name'] == vehicle]
			
			if len(van_details[van_details['vehicle_name'] == vehicle]['vehicle_speed'])==0:
				speed=2
			else:
				speed = (60/ van_details[van_details['vehicle_name'] == vehicle]['vehicle_speed'].iloc[0])
			for trip in vehicle_df['trip_id'].unique():
				trip_df = vehicle_df[vehicle_df['trip_id'] == trip].reset_index(drop = True)
				num_stores_mapping = dict(zip(trip_df['partyhll_code'].unique(), list(range(1, len(trip_df['partyhll_code'].unique()) + 1))))
				van_id=trip_df['van_id'].unique()[0]
				trip_df['outlet_sequence'] = trip_df['partyhll_code'].map(num_stores_mapping)
				vehicle_name=trip_df['van_id'][0]
				travel_time_seq = []
				end_time_seq = []
				start_time_seq = []
				dist_seq = []
				service_time_seq = []
				end_time = 0
				last_ol = 'Distributor'
				for ol in trip_df['partyhll_code']:
					if(last_ol != ol):
						travel_time = math.ceil(distance_matrix[last_ol][ol] * speed)
						travel_time_seq.append(travel_time)
						start_time = end_time + travel_time
						dist_seq.append(distance_matrix[last_ol][ol])
						start_time_seq.append(start_time)
						end_time = start_time + service_time_dict[vehicle_name+'_'+ol]
						end_time_seq.append(end_time)
						service_time_seq.append(service_time_dict[vehicle_name+'_'+ol])
						last_ol = ol
					else:
						travel_time_seq.append(travel_time_seq[-1])
						start_time_seq.append(start_time_seq[-1])
						end_time_seq.append(end_time_seq[-1])
						dist_seq.append(dist_seq[-1])
						service_time_seq.append(service_time_seq[-1])
						
				trip_df['travel_time'] = travel_time_seq
				trip_df['end_time'] = end_time_seq
				trip_df['start_time'] = start_time_seq
				trip_df['distance'] = dist_seq
				trip_df['service_time'] = service_time_seq
				if(buffer_time_vandict[van_id]>0):
					start = dt.time(9+(buffer_time_vandict[van_id]//60),buffer_time_vandict[van_id]%60, 0)
				else:
					start = dt.time(9, 0, 0)
				trip_df['end_time'] = trip_df['end_time'].apply(lambda x: (dt.datetime.combine(dt.date(1,1,1), start) + dt.timedelta(minutes = x)).time())
				trip_df['start_time'] = trip_df['start_time'].apply(lambda x: (dt.datetime.combine(dt.date(1,1,1), start) + dt.timedelta(minutes = x)).time())
		
				details_sheet_final = pd.concat([details_sheet_final, trip_df]).reset_index(drop = True)
				
		print(details_sheet_final.shape)
		
		details_sheet['start_time'] = details_sheet_final['start_time']
		details_sheet['end_time'] = details_sheet_final['end_time']
		
		new_df=transaction_input_v1.groupby(['partyhll_code','bill_number','fresh/old', 'priority_flag', 'order_number', 'order_date',
			   'smn_tagged', 'order_type','beat_name','party_code',
			   'servicing_plg', 'party_name','bill_date'])['net_sales_qty'].sum().reset_index()
		
		#details_sheet_final['beat_name']
		details_sheet_final=pd.merge(details_sheet_final,new_df[['partyhll_code','bill_number','fresh/old', 'priority_flag', 'order_number', 'order_date',
			   'smn_tagged', 'order_type','beat_name','party_code',
			   'servicing_plg', 'party_name','bill_date']],left_on=['partyhll_code','bill_number'],right_on=['partyhll_code','bill_number'],how='left')
			
			
		col_seq_1 = ['rs_code', 'rs_name',  'partyhll_code',  
		  'outlet_type', 'cluster_name', 'area_name', 
		 'execution_date', 'bill_number','fresh/old', 'priority_flag', 'order_number', 'order_date',
			   'smn_tagged', 'order_type','beat_name','party_code',
			   'servicing_plg', 'party_name','bill_date',
		  'bill_weight', 'bill_volume', 'no_of_crates', 'no_of_cases', 'vehicle name', 
		 'vehicle_utilised_weight', 'vehicle_utilised_volume', 'outlet_sequence','start_time', 'end_time', 
		 'travel_time', 'distance', 'service_time', 'trip_id', 'trip_sequence', 'familiarity', 'exclusivity']
		
		details_sheet_final = details_sheet_final[col_seq_1]
		
		# Familiarity
		
		owned_vehicles=list(details_sheet[details_sheet['rental_type']!='rented']['vehicle name'].unique())
		rental_vehicles=list(set(details_sheet['vehicle name'].unique())-set(owned_vehicles))
		details_sheet_copy = details_sheet.copy(deep = True)
		
		def find_familiarity(cur_data, prev_week_data):
				a = cur_data.groupby('beat_name')['vehicle name'].apply(lambda x: list(set(x)))
				b = a.apply(pd.Series).stack()
				b.index = b.index.droplevel(-1)
				b.name = 'vehicles'
				b = b.reset_index()
				c = b.groupby(['beat_name', 'vehicles']).size().reset_index()
			
				df = pd.concat([c, prev_week_data]).reset_index(drop = True)
				fam_series = df.groupby(['beat_name', 'vehicles']).size()
				familiarity = (len(fam_series[fam_series > 1])/ len(c))*100
				return familiarity
		
		path = os.getcwd()
		if('output_file.xlsx' in list(os.listdir(path))):
			prev_output = pd.read_excel('output_file.xlsx', sheet_name = 'details_sheet')
			execution_date = details_sheet['execution_date'].iloc[0]
			
			dates_tb_considered = []
			for i in [7, 14, 21, 28]:
				dates_tb_considered.append(execution_date - dt.timedelta(days = i))
				
			# prev_weeks_output = details_sheet.copy(deep = True)
			prev_weeks_output = prev_output[prev_output['exectuion_date'].isin(dates_tb_considered)].reset_index(drop = True)
			
			beat_vehicle_list = prev_weeks_output.groupby('beat_name')['vehicle name'].apply(lambda x: list(set(x)))
			beat_vehicle_list_1 = beat_vehicle_list.apply(pd.Series, 1).stack()
			beat_vehicle_list_1.index = beat_vehicle_list_1.index.droplevel(-1)
			beat_vehicle_list_1.name = 'vehicles'
			beat_vehicle_df = beat_vehicle_list_1.reset_index()
			prev_week_df = beat_vehicle_df.groupby(['beat_name', 'vehicles']).size().reset_index()
				
			
			cur_fam = find_familiarity(details_sheet, prev_week_df)
			owned_vehicle_weight = van_details[van_details['vehicle_name'].isin(owned_vehicles)][['vehicle_name', 'vehicle_capacity_kgs']]
			for weight in owned_vehicle_weight['vehicle_capacity_kgs'].unique():
				num_vehicles = len(owned_vehicle_weight[owned_vehicle_weight['vehicle_capacity_kgs'] == weight])
				if(num_vehicles > 1):
					for vehicles in list(itertools.combinations(owned_vehicles, 2)):
						temp_df = details_sheet.copy(deep = True)
						temp_df.loc[temp_df[temp_df['vehicle name'] == vehicles[0]].index, 'vehicle name'] = 'temp'
						temp_df.loc[temp_df[temp_df['vehicle name'] == vehicles[1]].index, 'vehicle name'] = vehicles[0]
						temp_df.loc[temp_df[temp_df['vehicle name'] == 'temp'].index, 'vehicle name'] = vehicles[1]
						fam = find_familiarity(temp_df, prev_week_df)
						# print(fam, cur_fam)
						if(fam > cur_fam):
							details_sheet = temp_df.copy(deep = True)
							cur_fam = fam
			
			details_sheet['familiarity'] = cur_fam
		else:
			details_sheet['familiarity'] = np.NaN   
			
			
		
		'''
		beat_name	partyhll_code	party_code	servicing_plg	party_name	outlet_type	cluster_name	
		area_name	bill_number	bill_date	execution_date	fresh/old	priority_flag	order_number	
		order_date	smn_tagged	order_type	bill_weight	bill_volume	no_of_crates	no_of_cases	vehicle name	
		vehicle_utilised_weight	vehicle_utilised_volume	outlet_sequence	start_time	end_time	
		travel_time	  trip_id	  trip_sequence	   familiarity	    exclusivity
		'''
		
		rate_check_dict = dict(zip(van_details['vehicle_name'], van_details['fixed_rate']))
		details_sheet.drop_duplicates(inplace = True)
		
		vehicle_summary = pd.DataFrame()
		rs_summary_1 = pd.DataFrame()
		
		vehicle_summary_1 = pd.DataFrame()
		rs_summary_2 = pd.DataFrame()
		
		
		details_sheet.shape
		
		details_sheet_final_beat_name=details_sheet_final[['bill_number','beat_name']].drop_duplicates()
		
		details_sheet=pd.merge(details_sheet,details_sheet_final_beat_name,left_on='bill_numbers',right_on='bill_number',how='left')
		
		
		
		for vehicle in owned_vehicles:
			print(vehicle)
			vehicle_df = details_sheet[details_sheet['vehicle name'] == vehicle]
			speed = (60/ van_details[van_details['vehicle_name'] == vehicle]['vehicle_speed'].iloc[0])
			for trip in vehicle_df['trip_id'].unique():
				trip_df = vehicle_df[vehicle_df['trip_id'] == trip].reset_index(drop = True)
				print(van_details[van_details['vehicle_name'] == vehicle]['vehicle_capacity_kgs'].reset_index(drop=True))
		#        if typ==owned_vehicles
				max_weight = (van_details[van_details['vehicle_name'] == vehicle]['vehicle_capacity_kgs'].reset_index(drop=True)[0])*(van_details[van_details['vehicle_name'] == vehicle]['weight_factor'].reset_index(drop=True)[0])
				trip_weight = trip_df['bill_weight'].sum()
				
				wgt_util = round((trip_weight/ max_weight)*100, 2)
				
				
				max_volume = van_details[van_details['vehicle_name'] == vehicle]['max_volume'].iloc[0] * van_details[van_details['vehicle_name'] == vehicle]['volume_factor'].iloc[0]
				trip_volume = trip_df['bill_volume'].sum()
				vol_util = round((trip_volume/ max_volume)*100, 2)
				
				dist = [distance_matrix['Distributor'][trip_df['path'].iloc[0]]]
				ol_dist = [0]
				num_stores = []
				if(len(trip_df['path']) > 1):
					for i in range(len(trip_df['path'])):
						if(i == 0):
							ol_dist.append(0)
						else:
							ol_dist.append(distance_matrix[trip_df['path'].iloc[i]][trip_df['path'].iloc[i+1]])    
						dist.append(distance_matrix[trip_df['path'].iloc[i]][trip_df['path'].iloc[i+1]])
						num_stores.append(i+1)
						if(i+1 == (len(trip_df) - 1)):
							break
				else:
					num_stores.append(1)
				
				
				trip_df['distance'] = pd.Series(dist)
				trip_df['ol_distance'] = pd.Series(ol_dist)
				trip_df['num_stores'] = pd.Series(num_stores)
					
				total_service_time = trip_df.groupby(['path'])['service_time_minutes'].max().sum()
				travel_time_rs = trip_df['distance'].sum() * speed
				travel_time_ol = trip_df['ol_distance'].sum() * speed
				total_time_spent = total_service_time + travel_time_rs
				start_time = trip_df['start_time'].min()
				end_time = trip_df['end_time'].max()
				
				time_util =travel_time_rs +total_service_time 
				perc_time_util = round(((trip_df['endtime'].max()/ 480)*100), 2)
				trip_df_1 = trip_df[['path', ]]
				total_distance = trip_df['distance'].sum() 
				distance_ol_ol = trip_df['ol_distance'].sum() 
				dist_rs_firsl_ol = trip_df[trip_df['num_stores'] == 1]['distance'].iloc[0]
				last_ol = trip_df[trip_df['num_stores'] == (trip_df['num_stores'].max())]['path'].iloc[0]
				dist_last_ol_rs = distance_matrix[last_ol]['Distributor']
				farthest_ol_dist = distance_matrix['Distributor'][distance_matrix['Distributor'].index.isin(trip_df['path'])].max()
		
				if(rate_check_dict[vehicle].lower() == 'no'):
					
					if(np.isnan(van_details[van_details['vehicle_name'] == vehicle]['per_km_rate_rupees'].iloc[0])):
						rate = van_details[van_details['vehicle_name']==vehicle]['base_rate_rupees'].iloc[0]+math.ceil((time_util/60)) * van_details[van_details['vehicle_name'] == vehicle]['per_hour_rate_rupees'].iloc[0]
							
					else:
						
						rate = van_details[van_details['vehicle_name']==vehicle]['base_rate_rupees'].iloc[0]+(math.ceil(total_distance) * van_details[van_details['vehicle_name'] == vehicle]['per_km_rate_rupees'].iloc[0])
				else:
					rate = van_details[van_details['vehicle_name'] == vehicle]['cost_per_day_rupees'].iloc[0]
					
				if(int(trip[-1]) >1):
					rate = rate * (van_details[van_details['vehicle_name'] == vehicle]['second_trip_cost'].iloc[0]-1)
					
					
					
		
						
				trip_df['no_of_bills'] = len(trip_df)
				trip_df['no_of_outlets'] = len(trip_df['path'].unique())
		
				
				criss_cross_ols = 0
				for ol in trip_df['path'].unique():
					other_trip_ols = details_sheet[details_sheet['van_id'] != vehicle]['path'].unique()
					dist_series = distance_matrix[ol][distance_matrix[ol].index.isin(other_trip_ols)]
					if(len(dist_series[dist_series < 0.2]) > 1):
						criss_cross_ols +=1
				num_ols = trip_df['num_stores'].max()
				perc_criss_cross = round((criss_cross_ols/ num_ols)*100, 2)
				trip_df['percent_criss_cross'] = perc_criss_cross
				
				trip_df['total_available_time_minutes'] = 480
		
				temp_df = pd.DataFrame(trip_df[['rs_code', 'rs_name', 'execution_date', 'vehicle_name', 'vehicle_type', 'rental_type', 
						 'vehicle_capacity_kgs', 'vehicle_capacity_volume', 'trip_id', 'trip_sequence', 
						 'perc_utilization_weight', 'perc_utilization_volume', 'no_of_bills', 'no_of_outlets', 
						 'percent_criss_cross', 'total_available_time_minutes']].iloc[0]).T
				temp_df['vehicle_capacity_kgs'] = max_weight
				temp_df['vehicle_capacity_volume'] = max_volume
				temp_df['perc_utilization_weight'] = wgt_util
				temp_df['perc_utilization_volume'] = vol_util
				temp_df['no_of_bills'] = temp_df['no_of_bills'].astype(int)
				temp_df['no_of_outlets'] = temp_df['no_of_outlets'].astype(int)
				temp_df['percent_criss_cross'] = temp_df['percent_criss_cross'].astype(float)
				temp_df['cost_of_utilization'] = rate
				temp_df['total_available_time_minutes'] = temp_df['total_available_time_minutes'].astype(int)
		
				temp_df['total_time_utilised_minutes'] = time_util
				temp_df['perc_time_utilized'] = perc_time_util
				temp_df['total_service_time_minutes'] = total_service_time
				temp_df['total_distance_travelled_km (ol to ol)'] = distance_ol_ol
				temp_df['total_distance_travelled_km (RS to RS)'] = total_distance
				temp_df['distance_RS_to_first_ol_km'] = dist_rs_firsl_ol
				temp_df['distance_last_ol_to_RS_km'] = dist_last_ol_rs
				temp_df['RS_to_farthest_ol_distanace_km'] = farthest_ol_dist
				temp_df['no_sales_beats_clubbed'] = trip_df['beat_name'].nunique()
				temp_df['no_salesman_clubbed'] = trip_df['beat_name'].nunique()
				
				vehicle_summary = pd.concat([vehicle_summary, temp_df]).reset_index(drop = True)
		
				
				rs_summary_temp = temp_df.copy(deep = True)
				rs_summary_temp['Number of channels clubbed'] = len(set(trip_df['outlet_type']))
				rs_summary_temp['Weight utilized (kgs)'] = trip_weight
				rs_summary_temp['Volume utilized (ft3)'] = trip_volume
				rs_summary_temp['Trip start time'] = trip_df[trip_df['path'] != 'Distributor']['start_time'].min()
				rs_summary_temp['Trip end time'] = trip_df[trip_df['path'] != 'Distributor']['end_time'].min()
				rs_summary_temp['Total travel time (min) - RS to RS'] = rs_summary_temp['total_distance_travelled_km (RS to RS)'] * speed
				rs_summary_temp['Total travel time (min) - OL to OL'] = rs_summary_temp['total_distance_travelled_km (ol to ol)'] * speed
				
				path = os.getcwd()
				if('output_file.xlsx' in list(os.listdir(path))):
					prev_output = pd.read_excel('output_file.xlsx', sheet_name = 'details_sheet')
					execution_date = details_sheet['execution_date'].iloc[0]
					
					dates_tb_considered = []
					for i in [7, 14, 21, 28]:
						dates_tb_considered.append(execution_date - dt.timedelta(days = i))
						
					# prev_weeks_output = details_sheet.copy(deep = True)
					prev_weeks_output = prev_output[prev_output['exectuion_date'].isin(dates_tb_considered)].reset_index(drop = True)
					
					beat_vehicle_list = prev_weeks_output.groupby('beat_name')['vehicle name'].apply(lambda x: list(set(x)))
					beat_vehicle_list_1 = beat_vehicle_list.apply(pd.Series, 1).stack()
					beat_vehicle_list_1.index = beat_vehicle_list_1.index.droplevel(-1)
					beat_vehicle_list_1.name = 'vehicles'
					beat_vehicle_df = beat_vehicle_list_1.reset_index()
					prev_week_df = beat_vehicle_df.groupby(['beat_name', 'vehicles']).size().reset_index()
						
					
					rs_summary_temp['familiarity'] = find_familiarity(rs_summary_temp, prev_week_df)
				else:
					rs_summary_temp['familiarity'] = np.NaN
		
				
				rs_summary_1 = pd.concat([rs_summary_1, rs_summary_temp]).reset_index(drop = True)
		
		
		trip_df.isna().sum()
		if len(rental_vehicles)>0:    
			for vehicle in rental_vehicles:
			#print(vehicle)
				vehicle_df = details_sheet[details_sheet['vehicle name'] == vehicle]
				for trip in vehicle_df['trip_id'].unique():
					trip_df = vehicle_df[vehicle_df['trip_id'] == trip].reset_index(drop = True)
					speed=2
					#print(van_details[van_details['vehicle_name'] == vehicle]['vehicle_capacity_kgs'].reset_index(drop=True))
					max_weight = int(vehicle.split('_')[0])*1.25
					trip_weight = trip_df['bill_weight'].sum()
					wgt_util = round((trip_weight/ max_weight)*100, 2)
					#rs_master.columns
					print(rs_master[rs_master['rental_vehicle_capacity_kgs'] == vehicle.split('_')[0]]['max_volume'])
					max_volume = rs_master[rs_master['rental_vehicle_capacity_kgs'] == int(vehicle.split('_')[0])]['max_volume'].iloc[0]*0.9
					trip_volume = trip_df['bill_volume'].sum()
					vol_util = round((trip_volume/ max_volume)*100, 2)
					
					dist = [distance_matrix['Distributor'][trip_df['path'].iloc[0]]]
					ol_dist = [0]
					num_stores = []
					if(len(trip_df['path']) > 1):
						for i in range(len(trip_df['path'])):
							if(i == 0):
								ol_dist.append(0)
							else:
								ol_dist.append(distance_matrix[trip_df['path'].iloc[i]][trip_df['path'].iloc[i+1]])    
							dist.append(distance_matrix[trip_df['path'].iloc[i]][trip_df['path'].iloc[i+1]])
							num_stores.append(i+1)
							if(i+1 == (len(trip_df) - 1)):
								break
					else:
						num_stores.append(1)
					
					trip_df['distance'] = pd.Series(dist)
					trip_df['ol_distance'] = pd.Series(ol_dist)
					trip_df['num_stores'] = pd.Series(num_stores)
					
					total_service_time = trip_df.groupby(['path'])['service_time_minutes'].max().sum()
					
					travel_time_rs = trip_df['distance'].sum() * speed
					travel_time_ol = trip_df['ol_distance'].sum() * speed
					total_time_spent = total_service_time + travel_time_rs
					start_time = trip_df['start_time'].min()
					end_time = trip_df['end_time'].max()
					
					time_util = travel_time_rs +total_service_time
					perc_time_util = round(((trip_df['endtime'].max()/ 480)*100), 2)
					trip_df_1 = trip_df[['path', ]]
					total_distance = trip_df['distance'].sum() 
					distance_ol_ol = trip_df['ol_distance'].sum() 
					dist_rs_firsl_ol = trip_df[trip_df['num_stores'] == 1]['distance'].iloc[0]
					last_ol = trip_df[trip_df['num_stores'] == (trip_df['num_stores'].max())]['path'].iloc[0]
					dist_last_ol_rs = distance_matrix[last_ol]['Distributor']
					farthest_ol_dist = distance_matrix['Distributor'][distance_matrix['Distributor'].index.isin(trip_df['path'])].max()
			
			
					rate = rs_master[rs_master['rental_vehicle_capacity_kgs'] == int(vehicle.split('_')[0])]['rental_costs_per_day_rupees'].iloc[0]
							
					trip_df['no_of_bills'] = len(trip_df)
					trip_df['no_of_outlets'] = len(trip_df['path'].unique())
					
					criss_cross_ols = 0
					for ol in trip_df['path'].unique():
						other_trip_ols = details_sheet[details_sheet['van_id'] != vehicle]['path'].unique()
						dist_series = distance_matrix[ol][distance_matrix[ol].index.isin(other_trip_ols)]
						if(len(dist_series[dist_series < 0.2]) > 1):
							criss_cross_ols +=1
					num_ols = trip_df['num_stores'].max()
					perc_criss_cross = round((criss_cross_ols/ num_ols)*100, 2)
					trip_df['percent_criss_cross'] = perc_criss_cross
					
					trip_df['total_available_time_minutes'] = 480
			
					temp_df = pd.DataFrame(trip_df[['rs_code', 'rs_name', 'execution_date', 'vehicle_name', 'vehicle_type', 'rental_type', 
							 'vehicle_capacity_kgs', 'vehicle_capacity_volume', 'trip_id', 'trip_sequence', 
							 'perc_utilization_weight', 'perc_utilization_volume', 'no_of_bills', 'no_of_outlets', 
							 'percent_criss_cross', 'total_available_time_minutes']].iloc[0]).T
					temp_df['vehicle_capacity_kgs'] = max_weight
					temp_df['vehicle_capacity_volume'] = max_volume
					temp_df['perc_utilization_weight'] = wgt_util
					temp_df['perc_utilization_volume'] = vol_util
					temp_df['no_of_bills'] = temp_df['no_of_bills'].astype(int)
					temp_df['no_of_outlets'] = temp_df['no_of_outlets'].astype(int)
					temp_df['percent_criss_cross'] = temp_df['percent_criss_cross'].astype(float)
					temp_df['cost_of_utilization'] = rate
					temp_df['total_available_time_minutes'] = temp_df['total_available_time_minutes'].astype(int)
			
					temp_df['total_time_utilised_minutes'] = time_util
					temp_df['perc_time_utilized'] = perc_time_util
					temp_df['total_service_time_minutes'] = total_service_time
					temp_df['total_distance_travelled_km (ol to ol)'] = distance_ol_ol
					temp_df['total_distance_travelled_km (RS to RS)'] = total_distance
					temp_df['distance_RS_to_first_ol_km'] = dist_rs_firsl_ol
					temp_df['distance_last_ol_to_RS_km'] = dist_last_ol_rs
					temp_df['RS_to_farthest_ol_distanace_km'] = farthest_ol_dist
					temp_df['no_sales_beats_clubbed'] = trip_df['beat_name'].nunique()
					temp_df['no_salesman_clubbed'] = trip_df['beat_name'].nunique()
					
					vehicle_summary_1 = pd.concat([vehicle_summary_1, temp_df]).reset_index(drop = True)
			
			
					rs_summary_temp = temp_df.copy(deep = True)
					rs_summary_temp['Number of channels clubbed'] = len(set(trip_df['outlet_type']))
					rs_summary_temp['Weight utilized (kgs)'] = trip_weight
					rs_summary_temp['Volume utilized (ft3)'] = trip_volume
					rs_summary_temp['Trip start time'] = trip_df[trip_df['path'] != 'Distributor']['start_time'].min()
					rs_summary_temp['Trip end time'] = trip_df[trip_df['path'] != 'Distributor']['end_time'].min()
					
					
					rs_summary_temp['Total travel time (min) - RS to RS'] = rs_summary_temp['total_distance_travelled_km (RS to RS)'] * speed
					rs_summary_temp['Total travel time (min) - OL to OL'] = rs_summary_temp['total_distance_travelled_km (ol to ol)'] * speed
					
					path = os.getcwd()
					if('output_file.xlsx' in list(os.listdir(path))):
						prev_output = pd.read_excel('output_file.xlsx', sheet_name = 'details_sheet')
						execution_date = details_sheet['execution_date'].iloc[0]
						
						dates_tb_considered = []
						for i in [7, 14, 21, 28]:
							dates_tb_considered.append(execution_date - dt.timedelta(days = i))
							
						# prev_weeks_output = details_sheet.copy(deep = True)
						prev_weeks_output = prev_output[prev_output['exectuion_date'].isin(dates_tb_considered)].reset_index(drop = True)
						
						beat_vehicle_list = prev_weeks_output.groupby('beat_name')['vehicle name'].apply(lambda x: list(set(x)))
						beat_vehicle_list_1 = beat_vehicle_list.apply(pd.Series, 1).stack()
						beat_vehicle_list_1.index = beat_vehicle_list_1.index.droplevel(-1)
						beat_vehicle_list_1.name = 'vehicles'
						beat_vehicle_df = beat_vehicle_list_1.reset_index()
						prev_week_df = beat_vehicle_df.groupby(['beat_name', 'vehicles']).size().reset_index()
							
						
						rs_summary_temp['familiarity'] = find_familiarity(rs_summary_temp, prev_week_df)
					else:
						rs_summary_temp['familiarity'] = np.NaN
					
						rs_summary_2 = pd.concat([rs_summary_2, rs_summary_temp]).reset_index(drop = True)
		
		
		vehicle_summary_2=pd.concat([vehicle_summary,vehicle_summary_1]).reset_index(drop=True)
		rs_summary_3 =pd.concat([rs_summary_1,rs_summary_2]).reset_index(drop=True)
		rs_rename_dict = {
			'total_service_time_minutes' : 'Total service time (min)',
			'perc_utilization_weight' : '% utilization (weight)',
			'perc_utilization_volume' : '% utilization (volume)',
			'vehicle_capacity_kgs' : 'Weight available (kgs)',
			'vehicle_capacity_volume' : 'Volume available (ft3)',
			'total_time_utilised_minutes' : 'Total time spent per trip (min)',
			'total_distance_travelled_km (RS to RS)' : 'Total distance (km) - RS to RS',
			'total_distance_travelled_km (ol to ol)' : 'Total distance (km) - OL to OL',
			'distance_RS_to_first_ol_km' : 'Distance between RS & first outlet in trip (km)',
			'distance_last_ol_to_RS_km' : 'Distance between last outlet & RS (km)',
			'RS_to_farthest_ol_distanace_km' : 'Distance between RS & farthest outlet in trip (km)',
			'no_of_bills' : 'Total bills loaded (count)',
			'no_of_outlets' : 'Total outlets loaded (count)',
			'cost_of_utilization' : 'Cost incurred (Rs.)',
			'percent_criss_cross' : 'Percentage criss cross in the trip',
			'no_sales_beats_clubbed' : 'Number of sales beat combined',
			'no_salesman_clubbed' : 'Number of SMN codes clubbed',
			}
		
		rs_summary_3 = rs_summary_3.rename(columns = rs_rename_dict)
		rs_summary_3['% utilization (volume)']=rs_summary_3['% utilization (volume)'].astype(float)
		rs_summary_3['% utilization (weight)']=rs_summary_3['% utilization (weight)'].astype(float)
		
		
		rs_summary_final = rs_summary_3.groupby(['vehicle_name', 'trip_id'])['Weight available (kgs)', 'Weight utilized (kgs)', '% utilization (weight)',
		 'Volume available (ft3)', 'Volume utilized (ft3)', '% utilization (volume)', 
		 'Total service time (min)', 'Total travel time (min) - RS to RS', 'Total travel time (min) - OL to OL',
		 'Total time spent per trip (min)', 'Trip start time', 'Trip start time', 'Trip end time',
		 'Total distance (km) - RS to RS', 'Total distance (km) - OL to OL',
		 'Distance between RS & first outlet in trip (km)', 'Distance between last outlet & RS (km)',
		 'Distance between RS & farthest outlet in trip (km)', 'Total bills loaded (count)', 
		 'Total outlets loaded (count)', 'Cost incurred (Rs.)', 'Percentage criss cross in the trip', 
		 'Number of channels clubbed', 'familiarity','Number of sales beat combined','Number of SMN codes clubbed'].sum()
		
		rs_summary_final = rs_summary_final.T
		
		
		'''
		rs_summary_final = rs_summary_3.groupby(['vehicle_name', 'trip_id'])['Weight available (kgs)', 'Weight utilized (kgs)', '% utilization (weight)',
		 'Volume available (ft3)', 'Volume utilized (ft3)', '% utilization (volume)', 
		 'Total service time (min)', 'Total travel time (min) - RS to RS', 'Total travel time (min) - OL to OL',
		 'Total time spent per trip (min)', 'Trip start time', 'Trip start time', 'Trip end time',
		 'Total distance (km) - RS to RS', 'Total distance (km) - OL to OL',
		 'Distance between RS & first outlet in trip (km)', 'Distance between last outlet & RS (km)',
		 'Distance between RS & farthest outlet in trip (km)', 'Total bills loaded (count)', 
		 'Total outlets loaded (count)', 'Cost incurred (Rs.)', 'Percentage criss cross in the trip', 
		 'Number of sales beat combined', 'Number of SMN codes clubbed', 'Number of channels clubbed', 'familiarity'].sum()
		
		'''
		
		
		def round_off(value):
			
			if(type(value) == float):
				
				return round(value, 2)
			else:
				return value
			
		rs_summary_final = rs_summary_final.apply(lambda x: round_off(x))
		
		
		unassigned_bills = set(transaction_input_v1['bill_number']) - set(details_sheet['bill_numbers'])
		unassigned_df = transaction_input_v1[transaction_input_v1['bill_number'].isin(unassigned_bills)].reset_index(drop = True)
		unassigned_outlets = list(set(unassigned_df['partyhll_code']))
		print('Unassigned bills, ols', len(unassigned_bills), len(unassigned_outlets))
		details_sheet_final['global_trip_ids']=details_sheet_final['execution_date'].astype(str)+'_'+details_sheet_final['trip_id']
		vehicle_summary_2['global_trip_ids']=vehicle_summary_2['execution_date'].astype(str)+'_'+vehicle_summary_2['trip_id']
		############### unassigned reasons #############
		missing_lat_long=list(outlet_data[outlet_data['outlet_latitude'].isna()]['partyhll_code'].unique())
		geo_location_dict={ i : 'missing geolocation' for i in missing_lat_long }
		
		splg_na=list(transaction_input[transaction_input['servicing_plg'].isna()]['partyhll_code'].unique())
		plg_na_dict={ i : 'plg missing' for i in splg_na }
		
		self_pick=list(outlet_data[outlet_data['self_pickup']=='yes']['partyhll_code'].unique())
		self_pick_dict={ i : 'self pickup' for i in self_pick }
		
		
		missing_outlets=list(transaction_input[~(transaction_input['partyhll_code'].isin(list(outlet_data['partyhll_code'].unique())))]['partyhll_code'].unique())
		mis_out_dict={ i : 'info in master_data missing' for i in missing_outlets }
		
		final_dict={**geo_location_dict,**plg_na_dict,**self_pick_dict,**mis_out_dict}
		unassigned_df['reasons']=unassigned_df['partyhll_code'].map(final_dict)
		unassigned_df.loc[unassigned_df['reasons'].isna(),'reasons']='outliers'
		
		
		
		
		############################################
		
		# writer = pd.ExcelWriter(rscode+'_final_output.xlsx')
		# details_sheet_final.to_excel(writer, 'details_sheet', index = False)
		# vehicle_summary_2.to_excel(writer, 'vehicle_summary', index = False)
		# rs_summary_final.to_excel(writer, 'rs_summary')
		# unassigned_df.to_excel(writer, 'unassigned_outlets', index = False)
		# writer.save()
		
		return details_sheet_final,vehicle_summary_2,unassigned_df
    
	except:
		return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
#main(date,'43A469','SHOGUN RPRT_updated.xlsx','Master_Input_Delivery Beat Opt - UNITED DISTRIBUTORS 43A469_v1.xlsx')
         
transfilename='431936_transaction_input_10th feb.xlsx'
masterfilename='Master_Input_Delivery Beat Opt -KAAVIAMMAN ENTERPRISES 431936 .xlsx'
orders_data= pd.read_excel(transfilename,sheet_name='transactional_data') 
cd=pd.DataFrame()
vs=pd.DataFrame()
ac=pd.DataFrame()
rscode='431936'

for date in ['2020-02-10']: 
    details_sheet_final,vehicle_summary_2,unassigned_df = main(date,rscode,transfilename,masterfilename)
    print('completed',len(details_sheet_final))
    f = open("logs_idf_v9.txt", "a")
    f.write('\n '+str(date)+' executed '+str(time.time()))
    f.close()
    if(len(details_sheet_final)>0):
        details_sheet_final['date']=date
        vehicle_summary_2['date']=date
        unassigned_df['date']=date
        cd = pd.concat([cd,details_sheet_final]) 
        vs= pd.concat([vs,vehicle_summary_2]) 
        ac = pd.concat([ac,unassigned_df]) 
        ac.drop_duplicates(subset =['partyhll_code','reasons'], keep = 'first', inplace = True)
        cd.to_csv('details_sheet_v9'+str(date)+'.csv')
        vs.to_csv('vehicle_summary_v9'+str(date)+'.csv')
        ac.to_csv('unassigned_outlets_v9'+str(date)+'.csv')
    else:
        f = open("logs_idf_v9.txt", "a")
        f.write('\n '+str(date)+' not executed '+str(time.time()))
        f.close()
        
writer = pd.ExcelWriter(rscode+'_updated_output_report_v9.xlsx')
cd.to_excel(writer, 'details_sheet', index = False)
vs.to_excel(writer, 'vehicle_summary', index = False)
ac.to_excel(writer, 'unassigned_outlets', index = False)
writer.save()
               
             

     
     
     



    
         