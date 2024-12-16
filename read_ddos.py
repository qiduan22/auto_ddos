# syn 20
# ['172.16.0', '52.43.17', '192.168.50', '172.217.10', '8.6.0', '104.110.151', '162.248.19', '173.194.175', '52.36.47', '216.58.217', '216.58.219', '74.208.236', '23.194.142', '23.194.140', '74.125.28', '172.217.12', '34.201.83', '54.164.24', '23.220.46', '23.33.85']
# dns 78
# ['172.16.0', '192.168.50', '54.210.144', '8.6.0', '34.208.208', '74.208.236', '172.217.10', '172.217.12', '54.218.239', '52.203.113', '35.173.44', '23.194.142', '54.222.199', '23.15.4', '104.36.115', '172.217.0', '72.21.91', '208.185.50', '34.216.156', '34.204.21', '52.7.108', '108.177.112', '172.217.2', '172.217.1', '173.241.244', '38.69.238', '125.56.201', '52.89.179', '0.0.0', '172.217.7', '172.217.9', '216.58.219', '94.31.29', '52.36.71', '96.6.27', '52.11.213', '52.10.142', '172.217.6', '216.239.32', '34.193.24', '104.88.90', '172.217.11', '23.194.141', '162.248.19', '52.114.75', '107.178.246', '172.217.5', '104.88.29', '104.88.60', '208.185.55', '34.211.202', '104.88.52', '173.194.175', '204.154.111', '54.192.49', '23.194.140', '209.85.232', '91.189.89', '192.0.73', '172.217.197', '172.217.3', '65.55.44', '65.52.108', '13.107.4', '134.170.51', '40.69.216', '104.88.46', '4.2.2', '8.8.8', '23.194.109', '52.34.90', '35.167.70', '54.186.208', '18.235.81', '52.200.108', '173.194.68', '52.34.107', '91.189.88']

import datetime


def read_pre(file):
    src = []
    # file = dir+'DrDos_DNS.csv'
    with open(file, 'r') as f:

        line = f.readline()
        while line:
            parts = line.split(',')
            srcip = parts[2].strip()
            if '.' in srcip:
                ip_parts = srcip.split('.')
                prefix = ip_parts[0] + '.' + ip_parts[1] + '.' + ip_parts[2]
                if prefix not in src:
                    src.append(prefix)
            line = f.readline()
    return src

def read_feature(file, index_set):
    features = []
    with open(file, 'r') as f:
        count = 0
        line = f.readline()
        while line:

            parts = line.split(',')
            # print(parts)
            if count > 0:
                feature_set = []
                for i in index_set:
                    feature_set.append(parts[i])
                features.append(feature_set)
            count += 1
            # if count > 10:
            #     break
            line = f.readline()
    return features

def read_feature_all(file, total):
    features = []
    with open(file, 'r') as f:
        count = 0
        line = f.readline()
        while line:
            if count > 0:
                parts = line.split(',')
                # print(parts)
                features.append(parts)
            count += 1
            if count > total:
                break
            line = f.readline()
    return features

def get_prefix(ip):
    ip_parts = ip.split('.')
    prefix = ip_parts[0] + '.' + ip_parts[1] + '.' + ip_parts[2]
    return prefix

def process_feature_testing(file, features, max_val, strings, paras):
    total = len(features)
    converted = [[] for _ in range(total)]
    cols = len(paras[file]["index"])

    last_date = 0
    for i in range(total):
        for j in range(cols):
            ind = paras[file]["index"][j]
            if paras[file]["type"][j] == 'prefix':
                # print('current value: ',  features[i][ind])
                prefix = get_prefix(features[i][ind])
                if prefix in strings[j]:
                    converted[i].append(strings[j].index(prefix) + 1)
                else:
                    converted[i].append(0)
            elif paras[file]["type"][j] == 'time':

                date_value = datetime.datetime.strptime(features[i][ind], '%Y-%m-%d %H:%M:%S.%f')
                if i > 0:
                    diff = date_value - last_date
                    diff = diff.total_seconds()
                else:
                    diff = 0
                converted[i].append(diff)
                last_date = date_value
            elif paras[file]["type"][j] == 'string':
                str_value = features[i][ind]
                if str_value not in strings[j]:
                    converted[i].append(0)
                else:
                    converted[i].append(strings[j].index(str_value) + 1)
            elif paras[file]["type"][j] == 'value':
                if max_val[j] > 0:
                    converted[i].append(float(features[i][ind])/max_val[j] )
                else:
                    converted[i].append(float(features[i][ind]))
    return converted

def process_feature_by_index(file,  features, paras):
    total = len(features)
    converted = [[] for _ in range(total)]
    strings  = [[]]

    cols = len(paras[file]["index"])
    strings = [[] for _ in range(cols)]
    max_val = [0 for _ in range(cols)]
    for j in range(cols):

        ind = paras[file]["index"][j]
        # print('current index: ', ind)
        if paras[file]["type"][j] == "value":
            max_val[j] = max(int(feature[ind]) for feature in features)
    print('max_val: ', max_val)
    last_date = 0
    for i in range(total):
        for j in range(cols):
            ind = paras[file]["index"][j]
            if paras[file]["type"][j] == 'prefix':
                # print('current value: ',  features[i][ind])
                prefix = get_prefix(features[i][ind])
                if prefix not in strings[j]:
                    strings[j].append(prefix)
                converted[i].append(strings[j].index(prefix) + 1)
            elif paras[file]["type"][j] == 'time':

                date_value = datetime.datetime.strptime(features[i][ind], '%Y-%m-%d %H:%M:%S.%f')
                if i > 0:
                    diff = date_value - last_date
                    diff = diff.total_seconds()
                else:
                    diff = 0
                converted[i].append(diff)
                last_date = date_value
            elif paras[file]["type"][j] == 'string':
                str_value = features[i][ind]
                if str_value not in strings[j]:
                    strings[j].append(str_value)
                converted[i].append(strings[j].index(str_value) + 1)
            elif paras[file]["type"][j] == 'value':
                if max_val[j] > 0:
                    converted[i].append(float(features[i][ind])/max_val[j] )
                else:
                    converted[i].append(float(features[i][ind]))
    return converted, strings, max_val

