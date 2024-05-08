


import pandas as pd

#

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
    df_scaled = scaler.fit_transform(df)   #We keep the feature range in the training set is same as the testing set
                                            # thus, we use the fit_transform method on the training set and the testing set.

    # 将标准化后的数据重新转换为DataFrame
    df_normalized = pd.DataFrame(df_scaled, columns=df.columns)# df_normalized 中的所有列现在都在0到1之间

    Y = df_normalized['H']
    X = df_normalized.drop(columns=['HW','H','H1','H2','H3'])

    # X['1']=Y
    # X.to_csv('toSMD.csv')
    import numpy as np
    X,Y,W=np.array(X), np.array(Y),np.array(W)
    W = W.reshape(-1, 1)
    # Concatenate X and W horizontally
    XW = np.concatenate((X, W), axis=1)

    # Extract W from the combined array
    # If W was the last column added, it can be accessed as follows:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(XW, Y, test_size=0.3)
    X_trainW=X_train[:, -1]
    X_trainW = X_trainW / X_trainW.sum()
    X_testW=X_test[:, -1]
    return  X_train[:, 0:-1],X_trainW, y_train, X_test[:, 0:-1], X_testW,y_test

    # return X,Y,W






    # X.to_csv('tocag.csv')
    # mc
    # P
    # sub
    # F_a
    # F_b
    # F_i
    # Ga
    # Gb
    # band2
    # band3
    # L
    # J
    # I
    # B
# def datasetprepare_scale(safety_weight=[300,29,1]):
#     # 假设你的DataFrame名为df
#     # 使用每列的平均值来填充缺失值
#     file_path = r'./output.csv'
#     df = pd.read_csv(file_path) # sheet_name不指定时默认返回全表数据
#
#     df.drop(columns=['mon'], inplace=True)
#     for column_name in ['mc','P','sub']:  #['mc','P','sub','mon']
#         # 假设你的DataFrame名为df，"column_name"是你要处理的字符串列名
#         unique_strings = df[column_name].unique()
#         df[column_name] = df[column_name].apply(lambda x: list(unique_strings).index(x))
#
#     df['H'].fillna(0, inplace=True)
#     df['H1'].fillna(0, inplace=True)
#     df['H2'].fillna(0, inplace=True)
#     df['H3'].fillna(0, inplace=True)
#     df['H'] = df['H'].apply(lambda x: 1 if x != 0 else x)
#     df.fillna(df.mean(), inplace=True)
#     # # df['H2'] = df['H2'].map(lambda x: x ** 2)
#     #
#     df['HW']=df['H1']*safety_weight[0]/sum(safety_weight)+df['H2']*safety_weight[1]/sum(safety_weight)+df['H3']*safety_weight[2]/sum(safety_weight)
#     df['HW'] = df['HW'] / df['HW'].sum()
#     from sklearn.preprocessing import MinMaxScaler
#
#     # scaler = MinMaxScaler()
#     # # 使用fit_transform来标准化DataFrame中的所有列
#     # df_scaled = scaler.fit_transform(df)
#     #
#     #
#     # # 将标准化后的数据重新转换为DataFrame
#     # df_normalized = pd.DataFrame(df_scaled, columns=df.columns)# df_normalized 中的所有列现在都在0到1之间
#     #
#     # Y = df_normalized['H']
#     # X = df_normalized.drop(columns=['HW','H','H1','H2','H3'])
#     #
#     # # X['1']=Y
#     # # X.to_csv('toSMD.csv')
#     # import numpy as np
#     # X,Y,W=np.array(X), np.array(Y),np.array(W)
#     # W = W.reshape(-1, 1)
#     # # Concatenate X and W horizontally
#     # XW = np.concatenate((X, W), axis=1)
#     #
#     # # Extract W from the combined array
#     # # If W was the last column added, it can be accessed as follows:
#     #
#     # from sklearn.model_selection import train_test_split
#     # X_train, X_test, y_train, y_test=train_test_split(XW, Y, test_size=0.3)
#     # X_trainW=X_train[:, -1]
#     # X_trainW = X_trainW / X_trainW.sum()
#     # X_testW=X_test[:, -1]
#
#
#     import numpy as np
#     from sklearn.preprocessing import MinMaxScaler
#     from sklearn.model_selection import train_test_split
#
#     # 假设df是你的原始DataFrame
#
#     # 分离特征、标签和权重
#     Y = df['H']
#     W = np.array(df['HW']).reshape(-1, 1)  # 假设"W"列已经存在于DataFrame中
#     X = df.drop(columns=['HW', 'H', 'H1', 'H2', 'H3'])
#
#
#     # 分割数据和权重
#     X_train, X_test, y_train, y_test, W_train, W_test = train_test_split(X, Y, W, test_size=0.3, random_state=42)
#
#     # 初始化MinMaxScaler
#     scaler = MinMaxScaler()
#
#     # 分别标准化训练数据和测试数据
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     max_values = X_train.max()
#     min_values = X_train.min()
#
#     # 输出最大值和最小值
#     print("最大值:")
#     print(max_values)
#     print("最小值:")
#     print(min_values)
#
#     X_trainW=W_train
#     X_testW=W_test
#
#     #
#     # # 现在你的数据已经准备好用于模型训练
#
#     return  X_train_scaled,X_trainW, y_train, X_test_scaled, X_testW,y_test