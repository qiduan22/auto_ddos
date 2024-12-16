
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

