import matplotlib.pyplot as plt
import numpy as np
# Provided accuracy values for each method
data_to_plot = [
    [0.6968785724936629] * 20,
    [0.822504234374738, 0.8236397748592871, 0.8236397748592871, 0.8236397748592871, 0.8236397748592871,
     0.8236397748592871, 0.8236397748592871, 0.8236397748592871, 0.8236397748592871, 0.8236397748592871,
     0.8236397748592871, 0.8236397748592871, 0.8236397748592871, 0.8236397748592871, 0.8236397748592871,
     0.8236397748592871, 0.8236397748592871, 0.822504234374738, 0.8213716108452951, 0.822504234374738],
    [0.8187183383991897] * 20,
    [0.8028665413533834, 0.7933918447628965, 0.8086622807017544, 0.8251315140555646, 0.8071326974564926,
     0.8223484469355753, 0.8347431077694236, 0.8354719836804527, 0.8194166416441464, 0.8229047120070522,
     0.8229819634588703, 0.8305169399816865, 0.8236397748592871, 0.8220721857647653, 0.8142840172419645,
     0.819833532913933, 0.8288576300085251, 0.8415963606286188, 0.8475286267402586, 0.8480445544554456],
    [0.7971304850866897] * 20,
    [0.7882958231954582, 0.7874873609706774, 0.7856066917438758, 0.7867024137699187, 0.7848371148212502,
     0.7870462633451958, 0.7840904333263554, 0.7807688428696296, 0.7807688428696296, 0.7782571547005728,
     0.7747519473095987, 0.768502331002331, 0.7663503197854342, 0.7663503197854342, 0.7652761634164191,
     0.7631313000440517, 0.7631313000440517, 0.7609907013423691, 0.7620604928017719, 0.7598007137310334],
    [0.7545904294908277] * 20
]

data_group_2_updated = [
    [0.8712972420837591] * 20,
    [0.9753609406068989, 0.9738466864775973, 0.9758658008658009, 0.9811067877215738, 0.9829543817181061,
     0.9829543817181061, 0.9682611832611833, 0.9682611832611833, 0.9718424619999424, 0.9811067877215738,
     0.9811067877215738, 0.9811067877215738, 0.98125, 0.9756424677684521, 0.9700455843743689,
     0.9738466864775973, 0.9738466864775973, 0.9738466864775973, 0.9756424677684521, 0.9700455843743689],
    [0.9564835101476014] * 20,
    [0.9908424908424909, 0.9796681096681097, 0.9850490196078432, 0.9791576611625026, 0.98125,
     0.9890510948905109, 0.9944649446494465, 0.9944649446494465, 0.9872727272727273, 0.9962962962962962,
     0.9942811869778162, 0.9962962962962962, 0.9943445433849168, 0.9890510948905109, 0.9855072463768115,
     0.9814488906840542, 0.9890510948905109, 0.9924989191526157, 0.9924989191526157, 0.992407652855414],
    [0.9872727272727279] * 20,
    [0.9517928858290304, 0.9470592464768479, 0.9480608419838523, 0.9426201125378734, 0.9546663790624101,
     0.9449410750215579, 0.9411396953147456, 0.948641312194981, 0.9487424547283703, 0.9432556801840667,
     0.9580352782326409, 0.9500631385604408, 0.9448222588004274, 0.9483446751083929, 0.9390387543451406,
     0.9361087914302992, 0.9449410750215579, 0.9279304664470263, 0.9306420009220839, 0.9268312010142923],
    [0.9255606282481988] * 20
]

# Re-creating the new data set after the code execution state reset
data_group_3 =  [
    [0.7057823129251699] * 20,
    [0.8666512193709088, 0.8687402190923317, 0.8666512193709088, 0.8666512193709088, 0.8666512193709088,
     0.8666512193709088, 0.8666512193709088, 0.8666512193709088, 0.8666512193709088, 0.8666512193709088,
     0.8649220425433257, 0.8649220425433257, 0.8649220425433257, 0.8628352545907432, 0.8649220425433257,
     0.8649220425433257, 0.8649220425433257, 0.8670218935769175, 0.8670218935769175, 0.8670218935769175],
    [0.8541919191919192] * 20,
    [0.8454329004329004, 0.8433902657165113, 0.8577025002880516, 0.8444620481060061, 0.8405568518998646,
     0.8520833333333333, 0.8462724417163769, 0.8602034035422265, 0.8561839598129222, 0.834860270815327,
     0.8484103170710597, 0.8757012724117987, 0.8722029237226805, 0.829543817181061, 0.8313620123321616,
     0.837558766692625, 0.8501123401313515, 0.8539074202097017, 0.8824754901960784, 0.8728494856056022],
    [0.8449770304238537] * 20,
    [0.7931523209300987, 0.7949165947656025, 0.7984644775776379, 0.7984644775776379, 0.8002483705171275,
     0.803836775674125, 0.802039088508782, 0.7964510293737618, 0.792653688230396, 0.7888563470870302,
     0.7870665632451377, 0.7814817476286289, 0.7850590059436644, 0.7850590059436644, 0.7794679536901377,
     0.7756686489126375, 0.7685446482563999, 0.7662762288013797, 0.7644880644233535, 0.7627054015942905],
    [0.7647363722751751] * 20
]

data_group_4=  [
    [0.6606278265496143] * 20,
    [0.8948113971183422, 0.8852922130484053, 0.862360369111219, 0.9143030597377367, 0.9186983972802332,
     0.9073174680265501, 0.8878258054071555, 0.8879674599320058, 0.891067670390157, 0.8727884086125951,
     0.8870325400679941, 0.8940747935891209, 0.8961631860126276, 0.9015986724947385, 0.8914845394204306,
     0.9146430305973774, 0.8915978630403107, 0.8728428039501376, 0.9101343694350008, 0.8839120932491501],
    [0.8727697506880363] * 20,
    [0.8943378662781285, 0.8936012627489072, 0.8722478549457666, 0.9041605957584588, 0.8655374777400032,
     0.8588149587178242, 0.8676542010684798, 0.8853205439533754, 0.876643192488263, 0.862360369111219,
     0.873712967459932, 0.8853772057633156, 0.88385543143921, 0.9020236360692893, 0.8972316658572121,
     0.9118544600938967, 0.8967014732070584, 0.888080783551886, 0.9074307916464304, 0.9317994171928121],
    [0.8680115751983163] * 20,
    [0.8220119929679786, 0.8200894254774309, 0.8131256371765951, 0.8084175543657133, 0.7860290753212975,
     0.8031314971944161, 0.7601526815602097, 0.7959268862434075, 0.8011126006437992, 0.8007272844035225,
     0.7729644465494128, 0.7706686039510969, 0.798808730623811, 0.7673934159087443, 0.781754473280727,
     0.7861775826222375, 0.7743652316312524, 0.7795509460316441, 0.771764347009384, 0.755581064917759],
    [0.7552980983038058] * 20
]


# differences = []
# for array in data_to_plot:
#     first_10_mean = sum(array[:10]) / 10
#     last_10_mean = sum(array[10:]) / 10
#     difference = first_10_mean - last_10_mean
#     differences.append(difference)
# print(differences)
# differences = []
# for array in data_group_2_updated:
#     first_10_mean = sum(array[:10]) / 10
#     last_10_mean = sum(array[10:]) / 10
#     difference = first_10_mean - last_10_mean
#     differences.append(difference)
# print(differences)
# differences = []
# for array in data_group_3:
#     first_10_mean = sum(array[:10]) / 10
#     last_10_mean = sum(array[10:]) / 10
#     difference = first_10_mean - last_10_mean
#     differences.append(difference)
# print(differences)
# differences = []
# for array in data_group_4:
#     first_10_mean = sum(array[:10]) / 10
#     last_10_mean = sum(array[10:]) / 10
#     difference = first_10_mean - last_10_mean
#     differences.append(difference)
# print(differences)
# exit(0)

# Names of the different methods, based on the order of the provided data lists
labels = ['No Resampling', 'EAT-SMOTE', 'SMOTE', 'EAT-ROS', 'ROS', 'EAT-ADASYN', 'ADASYN']
labels=labels[-2]



all_data1 = [a - b for a, b in zip(data_to_plot[-2], data_to_plot[-1])]
all_data2 = [a - b for a, b in zip(data_to_plot[-4], data_to_plot[-3])]
all_data3 = [a - b for a, b in zip(data_to_plot[-6], data_to_plot[-5])]
data_group_2_updated1 = [a - b for a, b in zip(data_group_2_updated[-2], data_group_2_updated[-1])]
data_group_2_updated2 = [a - b for a, b in zip(data_group_2_updated[-4], data_group_2_updated[-3])]
data_group_2_updated3 = [a - b for a, b in zip(data_group_2_updated[-6], data_group_2_updated[-5])]
data_group_3_updated1=[a - b for a, b in zip(data_group_3[-2], data_group_3[-1])]
data_group_3_updated2=[a - b for a, b in zip(data_group_3[-4], data_group_3[-3])]
data_group_3_updated3=[a - b for a, b in zip(data_group_3[-6], data_group_3[-5])]
data_group_4_updated1=[a - b for a, b in zip(data_group_4[-2], data_group_4[-1])]
data_group_4_updated2=[a - b for a, b in zip(data_group_4[-4], data_group_4[-3])]
data_group_4_updated3=[a - b for a, b in zip(data_group_4[-6], data_group_4[-5])]
all_data=[all_data1,all_data2,all_data3]
all_data_group=[data_group_2_updated1,data_group_2_updated2,data_group_2_updated3]
all_data_group_3=[data_group_3_updated1,data_group_3_updated2,data_group_3_updated3]
all_data_group_4=[data_group_4_updated1,data_group_4_updated2,data_group_4_updated3]

plt.figure(figsize=(10, 6))
# 首先有图（fig），然后有轴（ax）
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
# Extended-Accident-Triangle

box1=plt.boxplot(all_data,labels=['EAT-ADASYN','EAT-ROS','EAT-SMOTE'],positions=(1,1.4,1.8),widths=0.3,vert=True,patch_artist=True)
box2=plt.boxplot(all_data_group,labels=['EAT-ADASYN','EAT-ROS','EAT-SMOTE'],positions=(2.5,2.9,3.3),widths=0.3,vert=True,patch_artist=True)
box3=plt.boxplot(all_data_group_3,labels=['EAT-ADASYN','EAT-ROS','EAT-SMOTE'],positions=(4,4.4,4.8),widths=0.3,vert=True,patch_artist=True)
box3=plt.boxplot(all_data_group_3,labels=['EAT-ADASYN','EAT-ROS','EAT-SMOTE'],positions=(4,4.4,4.8),widths=0.3,vert=True,patch_artist=True)
box4=plt.boxplot(all_data_group_4,labels=['EAT-ADASYN','EAT-ROS','EAT-SMOTE'],positions=(5.5,5.9,6.3),widths=0.3,vert=True,patch_artist=True)
colors = ['pink', 'lightblue', 'lightgreen']
for patch, color in zip(box1['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(box2['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(box3['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(box4['boxes'], colors):
    patch.set_facecolor(color)
x_position=[1.4,2.9,4.4,5.9]
x_position_fmt=['GaussianNB','Decision Tree','Logistic Regression','CNN']
plt.xticks([i  for i in x_position], x_position_fmt, fontsize=16,fontstyle='normal')
# 加刻度名称
# plt.setp(axes, xticks=[1, 2, 3],
#          xticklabels=['x1', 'x2', 'x3'])
# 我们的刻度数是哪些，以及我们想要它添加的刻度标签是什么。
plt.title('Increasing Balanced Accuracy of Different Methods on SMD Dataset',fontsize=16)
plt.ylabel('Increasing Balanced Accuracy',fontsize=16)
# plt.xlabel('Methods')
# plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
plt.legend(box1['boxes'], ['EAT-ADASYN','EAT-ROS','EAT-SMOTE'], loc='upper left',fontsize=16)  # 绘制表示框，右下角绘制
plt.show()

# plt.figure(figsize=(10, 6))
# plt.boxplot(data_to_plot, labels=labels)
# plt.title('Boxplot of Different Methods')
# plt.ylabel('Balanced Accuracy')
# plt.xlabel('Methods')
# plt.xticks(rotation=45)
# plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping
# plt.show()
