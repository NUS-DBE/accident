


import pandas as pd


def datasetprepare(safety_weight=[300,29,1]):
    # 假设你的DataFrame名为df
    # 使用每列的平均值来填充缺失值
    file_path = r'./output.csv'
    df = pd.read_csv(file_path) # sheet_name不指定时默认返回全表数据

    df.drop(columns=['mon'], inplace=True)
    for column_name in ['mc','P','sub']:  #['mc','P','sub','mon']
        # 假设你的DataFrame名为df，"column_name"是你要处理的字符串列名
        unique_strings = df[column_name].unique()
        df[column_name] = df[column_name].apply(lambda x: list(unique_strings).index(x))

    df['H'].fillna(0, inplace=True)
    df['H1'].fillna(0, inplace=True)
    df['H2'].fillna(0, inplace=True)
    df['H3'].fillna(0, inplace=True)
    df['H'] = df['H'].apply(lambda x: 1 if x != 0 else x)
    df.fillna(df.mean(), inplace=True)
    # # df['H2'] = df['H2'].map(lambda x: x ** 2)
    #
    df['HW']=df['H1']*safety_weight[0]/sum(safety_weight)+df['H2']*safety_weight[1]/sum(safety_weight)+df['H3']*safety_weight[2]/sum(safety_weight)
    W = df['HW'] / df['HW'].sum()
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    # 使用fit_transform来标准化DataFrame中的所有列
    df_scaled = scaler.fit_transform(df)

    # 将标准化后的数据重新转换为DataFrame
    df_normalized = pd.DataFrame(df_scaled, columns=df.columns)# df_normalized 中的所有列现在都在0到1之间

    Y = df_normalized['H']
    X = df_normalized.drop(columns=['HW','H','H1','H2','H3'])

    return X,Y,W



