import os

import pandas

path_original = './datasets/original/'
path_generated = './datasets/generated/'


def read_csv(name):
    data = pandas.read_csv(path_original + name)
    data = remove_columns(data, name)

    print(data.dtypes)
    print(data.select_dtypes('object').columns)

    #data["Source Port"] = data["Source Port"].astype(int)

    #print(data.dtypes)
    #print(data.select_dtypes('object').columns)

    return data


def write_csv(sample, name, early_stop):
    sample.to_csv(path_generated + 'early_stop' + str(early_stop) + '/' + name, index=False)
    return 0


def get_datasets():
    return list(filter(lambda file: file.endswith('.csv'), os.listdir(path_original)))


def remove_columns(data, name):
    # print(data.isnull().values.any())

    if name == 'ADFANet_Shuffled.csv':
        data.drop('Time1', inplace=True, axis=1)
        data.drop('Time2', inplace=True, axis=1)

    elif name == 'AndMal_Shuffled.csv':
        data.drop('Timestamp', inplace=True, axis=1)
        data.drop('Flow ID', inplace=True, axis=1)
        data.drop('Source IP', inplace=True, axis=1)
        data.drop('Destination IP', inplace=True, axis=1)

    elif name == 'CICIDS17_Shuffled_Reduced.csv':
        data.drop('Flow_ID', inplace=True, axis=1)
        data.drop('Source_IP', inplace=True, axis=1)
        data.drop('Destination_IP', inplace=True, axis=1)
        data.drop('Timestamp', inplace=True, axis=1)

    elif name == 'CIDDS_Shuffled.csv':
        data.drop('Date_first_seen', inplace=True, axis=1)
        data.drop('Src_IP_Addr', inplace=True, axis=1)
        data.drop('Dst_IP_Addr', inplace=True, axis=1)

    elif name == 'CTU_Shuffled.csv':
        data.drop('Details', inplace=True, axis=1)
        data.drop('Dir', inplace=True, axis=1)
        data.drop('StartTime', inplace=True, axis=1)
        data.drop('SrcAddr', inplace=True, axis=1)
        data.drop('DstAddr', inplace=True, axis=1)

    elif name == 'ISCX_Shuffled.csv':
        data.drop('source', inplace=True, axis=1)
        data.drop('destination', inplace=True, axis=1)
        data.drop('startDateTime', inplace=True, axis=1)
        data.drop('stopDateTime', inplace=True, axis=1)

    elif name == 'NGDIS_Shuffled.csv':
        data.drop('Date', inplace=True, axis=1)
        data.drop('Time', inplace=True, axis=1)

    elif name == 'UGR_Shuffled.csv':
        data.drop('Timestamp', inplace=True, axis=1)
        data.drop('IP_S', inplace=True, axis=1)
        data.drop('IP_D', inplace=True, axis=1)

    return data
