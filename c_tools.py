import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pylab  as plt
import seaborn as sns
import sklearn.metrics as metries
from impala.dbapi import connect

def  v2_SplitData(df, col, numOfSplit, special_attribute=[]):
    if special_attribute != []:
        df = df.loc[~df[col].isin(special_attribute)]
        numOfSplit -= 1
    N = df.shape[0]
    n = N/numOfSplit
    splitPointIndex = [i * n for i in range(1, numOfSplit)]
    rawValues = sorted(list(df[col]))
    splitPoint = [rawValues[int(i)] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    print("{}分箱的切分点是：".format(col), splitPoint)
    return splitPoint
def v2_AssignGroup(x, bin,special_attribute=[]):
    numBin = len(bin) + 1 + len(special_attribute)
    if x  in special_attribute:
        i = special_attribute.index(x) +1
        return "NA"
    elif  x <= bin[0]:
        return  "(-00,{}]".format(bin[0])
    elif   x > bin[-1]:
        return "({},+00])".format(bin[-1])
    else:
        for i in range(0, numBin - 1):
            if bin[i] < x <= bin[i + 1]:
                return "({},{}]".format(bin[i], bin[i + 1])
def v2_equif_bin(df,cols_list,target):
    for col in cols_list:
        numOfSplit = 10
        print("{}开始进行分箱".format(col))
        if -99 not in set(df[col]):
            splitPoint = v2_SplitData(df, col, numOfSplit, special_attribute=[])
            df[col +"_Bin"] = df[col].map(lambda x: v2_AssignGroup(x, splitPoint,special_attribute=[]))
        else:
            print("{}包含-99".format(col))
            splitPoint = v2_SplitData(df, col, numOfSplit, special_attribute=[-99])
            df[col + "_Bin"] = df[col].map(lambda x: v2_AssignGroup(x, splitPoint, special_attribute=[-99]))
    return df

def v2_cat_woe_iv(df,cols_list,target,min_iv_value,address):
    error_var_name = []
    zerobad_numbers_varname=[]
    IV_high_dict = {}
    for col in cols_list:
        group = len(df[col].value_counts())
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        try:
           regroup.reset_index(level=0, inplace=True)
        except:
           print(regroup)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        regroup['good'] = regroup['total'] - regroup['bad']
        G = N - B
        regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
        regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
        group1 = regroup.shape[0]
        try:
           if group1 >=2:
             regroup['WOE'] = regroup.apply(lambda x: round(np.log(x.bad_pcnt * 1.0 / x.good_pcnt), 4), axis=1)
             WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
             for k, v in WOE_dict.items():
                WOE_dict[k] = v['WOE']
             IV = regroup.apply(lambda x: (x.bad_pcnt - x.good_pcnt) * np.log(x.bad_pcnt * 1.0 / x.good_pcnt), axis=1)
             IV = sum(IV)
             regroup["IV"] = IV
             regroup["var_name"] = col
             regroup["bad_rate"] = regroup.apply(lambda x: x.bad * 1.0/x.total,axis = 1)
             regroup["init_group"] = group
             regroup["end_group"] = group1
            # regroup = regroup.sort_values(by="bad_rate")
             if IV > min_iv_value:
                IV_high_dict[col] = regroup["IV"][regroup["var_name"]==col].values[0]
                print("IV大于0.02的变量详情*****************************\n",regroup)
                print(regroup)
                regroup.to_csv(address+"regroup1.csv",mode="a",header=True,encoding="utf_8_sig")
           else:
                error_var_name.append(col)
        except:
             zerobad_numbers_varname.append(col)
             
    df_error = pd.DataFrame(error_var_name,columns=["error_var_name"])
    print("出错的变量有：",len(df_error))
    df_zeroerror = pd.DataFrame(zerobad_numbers_varname,columns=["zerobad_numbers_varname"])
    print("出错的变量有：",len(df_zeroerror))
    df_error.to_excel(address+"error_name.xlsx",header=TabError)
    df_zeroerror.to_excel(address+"df_zeroerror.xlsx",header=TabError)
    df_high_var = pd.Series(IV_high_dict)
    return df_high_var


def AssignBin(x, cutOffPoints, special_attribute=[]):
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x) + 1
        return "b{}NA".format(0 - i)
    if x < cutOffPoints[0]:
        return "b{}[-inf, {})".format(0, cutOffPoints[0])
    elif x >= cutOffPoints[-1]:
        return "b{}[{},inf)".format((numBin - 1), cutOffPoints[-1])
    else:
        for i in range(0, numBin - 1):
            if cutOffPoints[i] <= x < cutOffPoints[i + 1]:
                return "b{}[{},{})".format((i + 1), cutOffPoints[i], cutOffPoints[i + 1])


def SplitData(df, col, numOfSplit, special_attribute=[]):
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = N / numOfSplit
    splitPointIndex = [int(i * n) for i in range(1, numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint


def Chi2(df, total_col, bad_col, overallRate):
    df2 = df.copy()
    df2["expected"] = df[total_col].apply(lambda x: x * overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


def BinBadRate(df, col, target, grantRateIndicator=0):
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup["bad_rate"] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[col], regroup["bad_rate"]))
    if grantRateIndicator == 0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


def ChiMerge(df, col, target, max_interval=5, special_attribute=[]):
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  # 如果原始属性的取值个数低于max_interval，不执行这段函数
        print("原始属性的取值个数低于max_interval,")
        print(colLevels)
        return colLevels[:-1]
    else:
        if len(special_attribute) >= 1:
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))
        if N_distinct > 100:
            split_x = SplitData(df2, col, 100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df2[col]
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]
        split_intervals = max_interval - len(special_attribute)
        while (len(groupIntervals) > split_intervals):
            chisqList = []
            for k in range(len(groupIntervals) - 1):
                temp_group = groupIntervals[k] + groupIntervals[k + 1]
                df2b = regroup.loc[regroup["temp"].isin(temp_group)]
                chisq = Chi2(df2b, "total", "bad", overallRate)
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined + 1]
            groupIntervals.remove(groupIntervals[best_comnbined + 1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]
        groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
        df2['temp_Bin'] = groupedvalues
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        while minBadRate == 0 or maxBadRate == 1:
            indexForBad01 = regroup[regroup['bad_rate'].isin([0, 1])].temp_Bin.tolist()
            bin = indexForBad01[0]
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            else:

                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])
            if len(cutOffPoints)!=0:
                groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupedvalues
                (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
                [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
            else:
                break
        # cutOffPoints = special_attribute + cutOffPoints
    return cutOffPoints


def AssignGroup(x, bin):
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return 10e10
    else:
        for i in range(N - 1):
            if bin[i] < x <= bin[i + 1]:
                return bin[i + 1]


def BadRateEncoding(df, col, target):
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding': badRateEnconding, 'bad_rate': br_dict}


def CalcWOE(df, col, target):
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: round(np.log(x.bad_pcnt * 1.0 / x.good_pcnt), 4), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.bad_pcnt - x.good_pcnt) * np.log(x.bad_pcnt * 1.0 / x.good_pcnt), axis=1)
    IV = sum(IV)
    regroup["IV"] = IV
    regroup["var_name"] = col
    # regroup.to_csv("./data/fi.csv")
    return {"WOE": WOE_dict, 'IV': IV}, regroup


def KS(df, pro, target):
    total = df.groupby([pro])[target].count()
    bad = df.groupby([pro])[target].sum()
    all = pd.DataFrame({'total': total, 'bad': bad})
    all["good"] = all["total"] - all["bad"]
    all[pro] = all.index
    all = all.sort_values(by=pro, ascending=True)
    all.index = range(len(all))
    num_bad = all['bad'].sum()
    num_good = all['good'].sum()
    all['badCumRate'] = all['bad'].cumsum() / num_bad
    all['goodCumRate'] = all['good'].cumsum() / num_good
    ks_array = all.apply(lambda x: abs(x.badCumRate - x.goodCumRate), axis=1)
    all["ks"] = ks_array
    ks = max(ks_array)
    return ks


def Vif(df, columns_name):
    x = np.matrix(df[columns_name])
    vif_list = [vif(x, i) for i in range(x.shape[1])]
    df_vif = pd.DataFrame({"var_name": columns_name, "vif": vif_list})
    max_vif = df_vif[df_vif["vif"] == df_vif["vif"].max()]
    return max_vif, df_vif


def AUC(y_true, pos_probablity_list):
    auc_value = 0
    fpr, tpr, thresholds = roc_curve(y_true, pos_probablity_list, pos_label=1)
    auc_value = auc(fpr, tpr)
    if auc_value < 0.5:
        auc_value = 1 - auc_value
    return auc_value


def Assign(x, cutoff):
    numbin = 20
    if x < cutoff[0]:
        return "%d[0,%.3f)" % (0, cutoff[0] * 100)  # '%.2f' % a
    elif x >= cutoff[-1]:
        return "%d[%.3f,100)" % ((numbin - 1), cutoff[-1] * 100)
    else:
        for i in range(0, numbin - 1):
            if cutoff[i] <= x < cutoff[i + 1]:
                return "%d[%.3f ,%.3f)" % ((i + 1), cutoff[i] * 100, cutoff[i + 1] * 100)


def score_situation(y_test, pos_scores):
    df = pd.DataFrame({"scores": pos_scores, "y": y_test})
    numOfSplit = 20  # 此处改
    N = df.shape[0]
    n = N / numOfSplit
    splitPointIndex = [int(i * n) for i in range(1, 20)]  # 此处改
    rawValues = sorted(list(df["scores"]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    df["scores" + "_b"] = df["scores"].map(lambda x: Assign(x, splitPoint))
    total = df.groupby(["scores_b"])["y"].count()
    total = pd.DataFrame({"total": total})
    bad_num = df.groupby(["scores_b"])["y"].sum()
    bad_num = pd.DataFrame({"bad_num": bad_num})
    regroup = total.merge(bad_num, left_index=True, right_index=True, how="left")
    regroup["客户占比"] = regroup.apply(lambda x: x.total * 1.0 / N, axis=1)
    regroup['good_num'] = regroup['total'] - regroup['bad_num']
    regroup.reset_index(level=0, inplace=True)
    regroup["overrate"] = regroup.apply(lambda x: x.bad_num * 1.0 / x.total, axis=1)
    regroup.columns = ["分数段", "客户人数", "逾期客户人数", "客户占比", "好客户人数", "逾期率"]
    a = regroup["分数段"].tolist()
    number = []
    for i in a:
        dex = i.index("[")
        i = i[:dex]
        number.append(int(i))
    regroup["number"] = number
    regroup = regroup.sort_values(by="number", ascending=False)
    total = sum(regroup["客户人数"])
    total_bad = sum(regroup["逾期客户人数"])
    total_good = sum(regroup["好客户人数"])
    ks_list = []
    recall_list = []
    number = list(regroup["number"])
    no_list = []
    cust_cum_pcnt = []
    for no in number:
        no_list.append(no)
        df_number = regroup[regroup["number"].isin(no_list)]
        df_cum_bad = sum(df_number["逾期客户人数"])
        df_cum_good = sum(df_number["好客户人数"])
        df_cum_cust = float("%10.4f" % (sum(df_number["客户人数"]) / N))
        tpr = df_cum_bad / total_bad
        fpr = df_cum_good / total_good
        ks = float("%10.4f" % ((tpr - fpr)))  # 保留三位小数
        ks_list.append(ks)
        recall = float("%10.4f" % ((df_cum_bad / total_bad)))  # # 保留三位小数
        recall_list.append(recall)
        cust_cum_pcnt.append(df_cum_cust)
    regroup["累计客户占比"] = cust_cum_pcnt  # 计算累计客户占比
    regroup["ks"] = ks_list
    regroup["累计Recall"] = recall_list
    scores = regroup["分数段"].tolist()
    low_list = []
    high_list = []
    for a in scores:
        low = float(a[a.index("[") + 1:a.index(",")])
        high = float(a[a.index(",") + 1:a.index(")")])
        low_list.append(low)
        high_list.append(high)
    regroup["低分数"] = low_list
    regroup["高分数"] = high_list
    return regroup


def score_situation_test(y_test, pos_scores, regroup):
    pos_scores = [float("%.3f" % (i * 100)) for i in pos_scores]  # 分数* 100保留三位小数
    df = pd.DataFrame({"scores": pos_scores, "y": y_test})
    N = df.shape[0]
    number = regroup["number"].tolist()
    no_list = []
    total_list = []
    bad_count_list = []
    good_count_list = []
    over_badrate_list = []
    ks_list = []
    recall_list = []
    total_pcnt = []
    for no in number:
        low = regroup[regroup["number"] == no]["低分数"].tolist()[0]
        high = regroup[regroup["number"] == no]["高分数"].tolist()[0]
        df1 = df[df["scores"] >= low]
        df1 = df1[df1["scores"] < high]
        total = df1.shape[0]
        total_pc = total * 1.0 / N
        bad_count = sum(df1['y'].tolist())  # # 坏客户人数
        good_count = total - bad_count  # 好客户
        over_badrate = float("%10.4f" % (bad_count / total))  # 逾期率
        total_pcnt.append(total_pc)
        total_list.append(total)
        bad_count_list.append(bad_count)
        good_count_list.append(good_count)
        over_badrate_list.append(over_badrate)
    regroup_test = pd.DataFrame()
    regroup_test["分数段"] = regroup["分数段"].tolist()
    regroup_test["客户人数"] = total_list
    regroup_test["客户占比"] = total_pcnt
    regroup_test["逾期客户人数"] = bad_count_list
    regroup_test["好客户人数"] = good_count_list
    regroup_test["逾期率"] = over_badrate_list
    regroup_test["number"] = number
    no_list = []
    total_bad = sum(regroup_test["逾期客户人数"])
    total_good = sum(regroup_test["好客户人数"])
    cust_cum_pcnt = []
    for no in number:
        no_list.append(no)
        df_number = regroup_test[regroup_test["number"].isin(no_list)]
        df_cum_bad = sum(df_number["逾期客户人数"])
        df_cum_good = sum(df_number["好客户人数"])
        df_cum_cust = float("%10.4f" % (sum(df_number["客户人数"]) / N))
        tpr = df_cum_bad / total_bad
        fpr = df_cum_good / total_good
        ks = float("%10.4f" % ((tpr - fpr)))  # 保留三位小数
        ks_list.append(ks)
        recall = float("%10.4f" % ((df_cum_bad / total_bad)))  # 保留三位小数
        recall_list.append(recall)
        cust_cum_pcnt.append(df_cum_cust)
    regroup_test["累计客户占比"] = cust_cum_pcnt
    regroup_test["ks"] = ks_list
    regroup_test["累计Recall"] = recall_list
    return regroup_test



def auc_ruce(x_test, pro_test, x_train, pro_train,address):
    fpr_test, tpr_test, th_test = metries.roc_curve(x_test, pro_test)
    fpr_train, tpr_train, th_train = metries.roc_curve(x_train, pro_train)
    plt.figure(figsize=[3, 3])
    plt.plot(fpr_train, tpr_train, "b--")
    plt.plot(fpr_test, tpr_test, "r--")
    plt.title("roc curve")
    plt.savefig(address+"roc_curve.png", dpi=100)
    plt.show()


def psi_score(pos_scores, regroup):
    pos_scores = [float("%.3f" % (i * 100)) for i in pos_scores]  # 分数* 100保留三位小数
    df = pd.DataFrame({"scores": pos_scores})
    N = df.shape[0]
    number = regroup["number"].tolist()
    no_list = []
    total_list = []
    bad_count_list = []
    good_count_list = []
    over_badrate_list = []
    ks_list = []
    recall_list = []
    total_pcnt = []
    for no in number:
        low = regroup[regroup["number"] == no]["低分数"].tolist()[0]
        high = regroup[regroup["number"] == no]["高分数"].tolist()[0]
        df1 = df[df["scores"] >= low]
        df1 = df1[df1["scores"] < high]
        total = df1.shape[0]
        total_pc = total * 1.0 / N
        total_pcnt.append(total_pc)
        total_list.append(total)
    regroup_test = pd.DataFrame()
    regroup_test["分数段"] = regroup["分数段"].tolist()
    regroup_test["客户人数"] = total_list
    regroup_test["客户占比"] = total_pcnt
    regroup_test["number"] = number
    return regroup_test


def judge_stable_analyze(data,col,address):
    num_counts=pd.crosstab(data['applymonth'],data[col],margins=True)
    num_counts_percents=num_counts.div(num_counts['All'],axis=0)
    num_counts_percents=num_counts_percents.drop('All',axis=1)
    num_counts_percents=num_counts_percents.drop('All',axis=0)
    
    bad_percents=pd.crosstab(index=data['applymonth'],columns=data[col],values=data['fraud_y'],aggfunc='mean',margins=True)
    bad_percents=bad_percents.drop('All',axis=1)
    bad_percents=bad_percents.drop('All',axis=0)
    
    plt.cla()
    plt.figure(2)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for cn in num_counts_percents.columns:
       plt.sca(ax1)
       plt.plot(num_counts_percents[cn])
       plt.xticks(rotation=90)
       plt.legend()
       plt.title(col)
       plt.sca(ax2)
       plt.plot(bad_percents[cn])
       plt.xticks(rotation=90)
       plt.legend()
       plt.title(col)
    plt.savefig(address+col+'.jpg')
    plt.show()
    
    count_data=pd.concat([num_counts_percents,bad_percents],axis=0)
    count_data.to_csv(address+col+'.csv')