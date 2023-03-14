# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = ['Times New Roman'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
#
# #lambda取值
# # x_data = ['0', '0.01', '0.1', '1', '10', '100']
# # #ip
# # y_data = [90.93, 91.01, 91.10, 91.07, 90.45, 69.32]
# # y_data1 = [84.33, 84.39, 84.59, 84.98, 84.79, 61.49]
# # y_data2 = [89.70, 89.79, 89.89, 89.86, 89.14, 63.88]
#
# # # pu
# # y_data = [97.26, 97.27, 97.57, 97.37, 95.07, 88.33]
# # y_data1 = [96.68, 96.60, 97.17, 96.59, 90.34, 80.45]
# # y_data2 = [96.42, 96.42, 96.77, 96.56, 92.94, 86.76]
#
# # #sv
# # y_data = [97.65, 97.60, 97.83, 97.67, 90.35, 79.61]
# # y_data1 = [98.01, 98.01, 98.17, 98.10, 91.02, 80.21]
# # y_data2 = [97.11, 97.09, 97.58, 97.26, 90.01, 78.96]
#
# #PATCH SIZE
# # #IP
# # x_data = ['1*1,11*11', '3*3,11*11', '5*5,11*11', '7*7,11*11', '9*9,11*11', '11*11,11*11']
# # y_data = [85.87, 90.56, 90.99, 91.10, 90.31, 89.59]
# # y_data1 = [81.25, 86.11, 85.20, 84.59, 84.17, 83.20]
# # y_data2 = [83.88, 89.27, 89.76, 89.89, 88.99, 88.18]
#
# # #PU
# # x_data = ['1*1,13*13', '3*3,13*13', '5*5,13*13', '7*7,13*13', '9*9,13*13', '11*11,13*13']
# # y_data = [94.33, 97.38, 97.57, 97.49, 97.13, 97.01]
# # y_data1 = [93.54, 97.53, 97.17, 96.47, 95.84, 95.20]
# # y_data2 = [92.43, 96.51, 96.77, 96.66, 96.18, 96.04]
# #
# # #SV
# # x_data = ['1*1,11*11', '3*3,11*11', '5*5,11*11', '7*7,11*11', '9*9,11*11', '11*11,11*11']
# # y_data = [91.93, 95.75, 96.80, 97.04, 97.83, 96.97]
# # y_data1 = [95.22, 97.45, 97.59, 97.55, 98.17, 97.59]
# # y_data2 = [91.04, 95.27, 96.44, 96.71, 97.58, 96.64]
#
#
# #siamese
# #IP
# x_data = ['OA', 'AA', 'Kappa']
# y_data_ip = [91.10, 84.59, 89.89]
# y_data1_ip = [90.59, 82.94, 89.30]
# #pu
# y_data_pu = [97.57, 97.17, 96.77]
# y_data1_pu = [97.44, 96.73, 96.59]
# #sv
# y_data_sv = [97.83, 98.17, 97.58]
# y_data1_sv = [97.40, 97.34, 96.31]
# # plt.figure(figsize=(15,3))
# # plt.subplot(1,3,1)##图包含1行3列子图，当前画在第一行第一列子图上
# # plt.plot(x_data, y_data_ip, color='#F9B9B7', linewidth=2.0, label='siamese')
# # plt.plot(x_data, y_data1_ip, color='#96C9DC', linewidth=2.0, label='no_siamese')
# # plt.title('(a)')
# # plt.xlabel('metrics')
# # plt.ylabel('accuracy %')
# # plt.subplot(1,3,2)##图包含1行3列子图，当前画在第一行第一列子图上
# # plt.plot(x_data, y_data_pu, color='#F9B9B7', linewidth=2.0, label='siamese')
# # plt.plot(x_data, y_data1_pu, color='#96C9DC', linewidth=2.0, label='no_siamese')
# # plt.title('(b)')
# # plt.xlabel('metrics')
# # plt.ylabel('accuracy %')
# # plt.subplot(1,3,3)##图包含1行3列子图，当前画在第一行第一列子图上
# # plt.plot(x_data, y_data_sv, color='#F9B9B7', linewidth=2.0, label='siamese')
# # plt.plot(x_data, y_data1_sv, color='#96C9DC', linewidth=2.0, label='no_siamese')
# # plt.title('(c)')
# # plt.xlabel('metrics')
# # plt.ylabel('accuracy %')
# # ax = plt.gca()
# # plt.legend()
# #
# # y_major_locator = plt.MultipleLocator(5)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.show()
#
# #siamese-OA，AA
# #OA
# # x_data = ['siamese', 'no_siamese']
# # y_data_oa = [91.10, 90.59]
# # y_data1_oa = [97.57, 97.44]
# # y_data2_oa = [97.83, 97.40]
# #
# # y_data_aa = [84.59, 82.94]
# # y_data1_aa = [97.17, 96.73]
# # y_data2_aa = [98.17, 97.34]
# #
# # y_data_kappa = [89.89, 89.30]
# # y_data1_kappa = [96.77, 96.59]
# # y_data2_kappa = [97.58, 96.31]
# #
# # plt.figure(figsize=(15,3))
# # plt.subplot(1,3,1)##图包含1行3列子图，当前画在第一行第一列子图上
# # plt.plot(x_data, y_data_oa, color='#F9B9B7', linewidth=2.0, label='IP')
# # plt.plot(x_data, y_data1_oa, color='#96C9DC', linewidth=2.0, label='PU')
# # plt.plot(x_data, y_data2_oa, color='#F5D491', linewidth=2.0, label='SV')
# # plt.title('(a)')
# # plt.xlabel('datasets')
# # plt.ylabel('accuracy %')
# # plt.subplot(1,3,2)##图包含1行3列子图，当前画在第一行第一列子图上
# # plt.plot(x_data, y_data_aa, color='#F9B9B7', linewidth=2.0, label='IP')
# # plt.plot(x_data, y_data1_aa, color='#96C9DC', linewidth=2.0, label='PU')
# # plt.plot(x_data, y_data2_aa, color='#F5D491', linewidth=2.0, label='SV')
# # plt.title('(b)')
# # plt.xlabel('datasets')
# # plt.ylabel('accuracy %')
# # plt.subplot(1,3,3)##图包含1行3列子图，当前画在第一行第一列子图上
# # plt.plot(x_data, y_data_kappa, color='#F9B9B7', linewidth=2.0, label='IP')
# # plt.plot(x_data, y_data1_kappa, color='#96C9DC', linewidth=2.0, label='PU')
# # plt.plot(x_data, y_data2_kappa, color='#F5D491', linewidth=2.0, label='SV')
# # plt.title('(c)')
# # plt.xlabel('datasets')
# # plt.ylabel('accuracy %')
# # ax = plt.gca()
# # plt.legend()
# #
# # y_major_locator = plt.MultipleLocator(5)
# # ax.yaxis.set_major_locator(y_major_locator)
# # plt.show()
#
#
#
#
# # 1.不同大小数据集
# x_data_ip = ['0.5', '1', '3', '5']
# y_data_ip = [80.29, 91.10, 97.20, 98.39]
# y_data1_ip = [73.73, 84.59, 96.45, 97.56]
# y_data2_ip = [77.50, 89.89, 96.81, 98.16]
# x_data_pu = ['0.25', '0.5', '1', '2']
# y_data_pu = [96.02, 97.57, 99.13, 99.52]
# y_data1_pu = [94.03, 97.17, 98.61, 99.24]
# y_data2_pu = [94.72, 96.77, 98.84, 99.36]
# x_data_sv = ['0.25', '0.5', '1', '2']
# y_data_sv = [96.11, 97.83, 98.98, 99.24]
# y_data1_sv = [97.56, 98.17, 98.97, 99.31]
# y_data2_sv = [96.01, 97.58, 98.59, 99.06]
# plt.figure(figsize=(15,3))
# plt.subplot(1,3,1)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_ip, y_data_ip, color='#F9B9B7', linewidth=2.0, label='OA')
# plt.plot(x_data_ip, y_data1_ip, color='#96C9DC', linewidth=2.0, label='AA')
# plt.plot(x_data_ip, y_data2_ip, color='#F5D491', linewidth=2.0, label='Kappa')
# plt.title('(a)')
# plt.xlabel('datasets')
# plt.ylabel('accuracy %')
# plt.subplot(1,3,2)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_pu, y_data_pu, color='#F9B9B7', linewidth=2.0, label='OA')
# plt.plot(x_data_pu, y_data1_pu, color='#96C9DC', linewidth=2.0, label='AA')
# plt.plot(x_data_pu, y_data2_pu, color='#F5D491', linewidth=2.0, label='Kappa')
# plt.title('(b)')
# plt.xlabel('datasets')
# plt.ylabel('accuracy %')
# plt.subplot(1,3,3)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_sv, y_data_sv, color='#F9B9B7', linewidth=2.0, label='OA')
# plt.plot(x_data_sv, y_data1_sv, color='#96C9DC', linewidth=2.0, label='AA')
# plt.plot(x_data_sv, y_data2_sv, color='#F5D491', linewidth=2.0, label='Kappa')
# plt.title('(c)')
# plt.xlabel('datasets')
# plt.ylabel('accuracy %')
# ax = plt.gca()
# plt.legend()
#
# y_major_locator = plt.MultipleLocator(5)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.show()
#
#
# # 数据集数量之间的比较
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = ['Times New Roman'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# x_data_ip = ['0.5', '1', '3', '5']
# y_data_ip = [80.29, 91.10, 97.20, 98.39]
# y_data1_ip = [79.23, 89.34, 96.12, 97.74]
# y_data2_ip = [71.45, 81.06, 92.02, 96.97]
# y_data3_ip = [75.43, 86.74, 93.74, 97.59]
# y_data4_ip = [66.61, 80.14, 91.68, 96.58]
# y_data5_ip = [59.02, 78.12, 90.44, 97.89]
# y_data6_ip = [44.89, 55.96, 67.91, 73.19]
# x_data_pu = ['0.25', '0.5', '1', '2']
# y_data_pu = [96.02, 97.57, 99.13, 99.52]
# y_data1_pu = [95.33, 95.80, 98.58, 99.07]
# y_data2_pu = [92.37, 95.62, 98.16, 98.89]
# y_data3_pu = [93.33, 95.07, 98.56, 99.03]
# y_data4_pu = [92.32, 95.46, 97.79, 98.94]
# y_data5_pu = [93.33, 96.98, 98.86, 99.35]
# y_data6_pu = [79.66, 84.86, 89.42, 91.32]
# x_data_sv = ['0.25', '0.5', '1', '2']
# y_data_sv = [96.11, 97.83, 98.98, 99.24]
# y_data1_sv = [93.84, 96.25, 97.64, 98.74]
# y_data2_sv = [90.69, 95.58, 97.58, 98.54]
# y_data3_sv = [92.32, 95.52, 97.77, 98.79]
# y_data4_sv = [88.87, 94.03, 95.01, 98.34]
# y_data5_sv = [90.80, 96.42, 98.08, 98.82]
# y_data6_sv = [82.01, 85.16, 88.65, 91.02]
# plt.figure(figsize=(15, 3))
# plt.subplot(1,3,1)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_ip, y_data_ip, color='#D14081', linewidth=2.0, marker='*', label='SSACC')
# plt.plot(x_data_ip, y_data1_ip, color='#FFB758', linewidth=2.0, marker='o', label='DBDA')
# plt.plot(x_data_ip, y_data2_ip, color='#97D8C4', linewidth=2.0, marker='v', label='DBMA')
# plt.plot(x_data_ip, y_data3_ip, color='#E66B6E', linewidth=2.0, marker='d', label='FDSSC')
# plt.plot(x_data_ip, y_data4_ip, color='#0A719E', linewidth=2.0, marker='X', label='SSRN')
# plt.plot(x_data_ip, y_data5_ip, color='#7E2E85', linewidth=2.0, marker='h', label='MAFN')
# plt.plot(x_data_ip, y_data6_ip, color='#FB6477', linewidth=2.0, marker='p', label='SVM')
# plt.title('(a) IP', y=-0.4)
# plt.xlabel('percentage of labelled samples for training (%)')
# plt.ylabel('OA (%)')
# ax = plt.gca()
# plt.legend()
# y_major_locator = plt.MultipleLocator(7)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.subplot(1,3,2)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_pu, y_data_pu, color='#D14081', linewidth=2.0, marker='*', label='SSACC')
# plt.plot(x_data_pu, y_data1_pu, color='#FFB758', linewidth=2.0, marker='o', label='DBDA')
# plt.plot(x_data_pu, y_data2_pu, color='#97D8C4', linewidth=2.0, marker='v', label='DBMA')
# plt.plot(x_data_pu, y_data3_pu, color='#E66B6E', linewidth=2.0, marker='d', label='FDSSC')
# plt.plot(x_data_pu, y_data4_pu, color='#0A719E', linewidth=2.0, marker='x', label='SSRN')
# plt.plot(x_data_pu, y_data5_pu, color='#7E2E85', linewidth=2.0, marker='h', label='MAFN')
# plt.plot(x_data_pu, y_data6_pu, color='#FB6477', linewidth=2.0, marker='p', label='SVM')
# plt.title('(b) PU', y=-0.4)
# plt.xlabel('percentage of labelled samples for training (%)')
# plt.ylabel('OA (%)')
# ax = plt.gca()
# plt.legend(loc='lower right')
# y_major_locator = plt.MultipleLocator(7)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.subplot(1,3,3)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_sv, y_data_sv, color='#D14081', linewidth=2.0, marker='*', label='SSACC')
# plt.plot(x_data_sv, y_data1_sv, color='#FFB758', linewidth=2.0, marker='o', label='DBDA')
# plt.plot(x_data_sv, y_data2_sv, color='#97D8C4', linewidth=2.0, marker='v', label='DBMA')
# plt.plot(x_data_sv, y_data3_sv, color='#E66B6E', linewidth=2.0, marker='d', label='FDSSC')
# plt.plot(x_data_sv, y_data4_sv, color='#0A719E', linewidth=2.0, marker='x', label='SSRN')
# plt.plot(x_data_sv, y_data5_sv, color='#7E2E85', linewidth=2.0, marker='h', label='MAFN')
# plt.plot(x_data_sv, y_data6_sv, color='#FB6477', linewidth=2.0, marker='p', label='SVM')
# plt.title('(c) SV', y=-0.4)
# plt.xlabel('percentage of labelled samples for training (%)')
# plt.ylabel('OA (%)')
# ax = plt.gca()
# plt.legend()
# y_major_locator = plt.MultipleLocator(7)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.show()


#
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = ['Times New Roman'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# x_data_ip = ['0.5', '1', '3', '5', '10']
# y_data_ip = [75.36, 90.01, 96.82, 97.91, 98.89]
# y_data1_ip = [72.22, 78.34, 93.46, 96.78, 97.94]
# y_data2_ip = [71.45, 88.71, 95.44, 96.97, 98.03]
# y_data3_ip = [68.17, 72.41, 89.81, 93.34, 94.59]
# y_data4_ip = [42.01, 45.61, 62.32, 65.96, 70.77]
#
# x_data_pu = ['0.25', '0.5', '1', '2', '5']
# y_data_pu = [90.78, 97.32, 97.69, 98.13, 99.05]
# y_data1_pu = [90.11, 95.41, 96.50, 97.80, 98.89]
# y_data2_pu = [87.51, 96.00, 97.46, 98.35, 98.66]
# y_data3_pu = [85.11, 95.59, 96.40, 97.30, 98.00]
# y_data4_pu = [80.00, 87.70, 90.21, 93.53, 96.12]
#
# x_data_sv = ['0.25', '0.5', '1', '2', '5']
# y_data_sv = [91.31, 95.98, 97.75, 98.98, 99.51]
# y_data1_sv = [90.41, 95.49, 97.01, 98.74, 99.19]
# y_data2_sv = [90.00, 95.44, 96.98, 98.60, 99.11]
# y_data3_sv = [83.50, 94.72, 95.81, 97.29, 98.00]
# y_data4_sv = [65.00, 77.79, 85.91, 94.11, 95.00]
#
# plt.figure(figsize=(15,3))
# plt.subplot(1,3,1)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_ip, y_data_ip, color='#D14081', linewidth=2.0, marker='*', label='NAIC')
# plt.plot(x_data_ip, y_data1_ip, color='#FFB758', linewidth=2.0, marker='o', label='3DOC-SSAN')
# plt.plot(x_data_ip, y_data2_ip, color='#97D8C4', linewidth=2.0, marker='v', label='DBDA')
# plt.plot(x_data_ip, y_data3_ip, color='#E66B6E', linewidth=2.0, marker='d', label='SSRN')
# plt.plot(x_data_ip, y_data4_ip, color='#0A719E', linewidth=2.0, marker='X', label='CDCNN')
#
# plt.title('(a) IP', y=-0.4)
# plt.xlabel('percentage of labelled samples for training (%)')
# plt.ylabel('OA (%)')
# ax = plt.gca()
# plt.legend()
# y_major_locator = plt.MultipleLocator(7)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.subplot(1,3,2)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_pu, y_data_pu, color='#D14081', linewidth=2.0, marker='*', label='NAIC')
# plt.plot(x_data_pu, y_data1_pu, color='#FFB758', linewidth=2.0, marker='o', label='3DOC-SSAN')
# plt.plot(x_data_pu, y_data2_pu, color='#97D8C4', linewidth=2.0, marker='v', label='DBDA')
# plt.plot(x_data_pu, y_data3_pu, color='#E66B6E', linewidth=2.0, marker='d', label='SSRN')
# plt.plot(x_data_pu, y_data4_pu, color='#0A719E', linewidth=2.0, marker='x', label='CDCNN')
#
# plt.title('(b) PU', y=-0.4)
# plt.xlabel('percentage of labelled samples for training (%)')
# plt.ylabel('OA (%)')
# ax = plt.gca()
# plt.legend(loc='lower right')
# y_major_locator = plt.MultipleLocator(7)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.subplot(1,3,3)##图包含1行3列子图，当前画在第一行第一列子图上
# plt.plot(x_data_sv, y_data_sv, color='#D14081', linewidth=2.0, marker='*', label='NAIC')
# plt.plot(x_data_sv, y_data1_sv, color='#FFB758', linewidth=2.0, marker='o', label='3DOC-SSAN')
# plt.plot(x_data_sv, y_data2_sv, color='#97D8C4', linewidth=2.0, marker='v', label='DBDA')
# plt.plot(x_data_sv, y_data3_sv, color='#E66B6E', linewidth=2.0, marker='d', label='SSRN')
# plt.plot(x_data_sv, y_data4_sv, color='#0A719E', linewidth=2.0, marker='x', label='CDCNN')
#
# plt.title('(c) SV', y=-0.4)
# plt.xlabel('percentage of labelled samples for training (%)')
# plt.ylabel('OA (%)')
# ax = plt.gca()
# plt.legend()
# y_major_locator = plt.MultipleLocator(7)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.show()

# 2.参数量
# x 方法 SSRN, FDSSC, DBMA, DBDA, PROPOSED
# y oa
# 圆的大小代表参数大小
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
x_data = ['CDCNN', 'SSRN', 'DBDA', 'NAIC']
y_data_ip = [62.32, 89.81, 95.38, 96.82]
txt_ip = ['1063824', '364168', '382347', '278172']
y_data_pu = [87.70, 95.59, 96.00, 97.32]
txt_pu = ['628361', '216537', '202772', '102197']
y_data_sv = [77.79, 94.72, 95.44, 95.98]
txt_sv = ['1081744', '370312', '389643', '285468']
s1 = [2120, 720, 760, 560]
s2 = [1260, 430, 400, 230]
s3 = [2160, 740, 780, 580]
c = ["#0A719E", "#FFB758", "#7E2E85", "#D14081"]
plt.figure(figsize=(15,3))
plt.subplot(1,3,1)##图包含1行3列子图，当前画在第一行第一列子图上
plt.scatter(x_data, y_data_ip, s=s1, c=c, marker='o')
for i in range(4):
    plt.annotate(txt_ip[i], xy=(x_data[i], y_data_ip[i]))
plt.title(u'(a)', y=-0.3)
plt.xlabel('methods')
plt.ylabel('OA (%)')
plt.subplot(1,3,2)##图包含1行3列子图，当前画在第一行第一列子图上
plt.scatter(x_data, y_data_pu, s=s2, c=c, marker='o')
for i in range(4):
    plt.annotate(txt_pu[i], xy=(x_data[i], y_data_pu[i]))
plt.title(u'(b)', y=-0.3)
plt.xlabel('methods')
plt.ylabel('OA (%)')
plt.subplot(1,3,3)##图包含1行3列子图，当前画在第一行第一列子图上
plt.scatter(x_data, y_data_sv, s=s3, c=c, marker='o')
for i in range(4):
    plt.annotate(txt_sv[i], xy=(x_data[i], y_data_sv[i]))
plt.title(u'(c)', y=-0.3)
plt.xlabel('methods')
plt.ylabel('OA (%)')
plt.legend()
plt.show()


# # # 3.patch字体新罗马， y改为OA， 图例IP:P1=11_
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = ['Times New Roman'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# x_data = ['1*1', '3*3', '5*5', '7*7', '9*9', '11*11']
# y_data_ip = [85.87, 90.56, 90.99, 91.10, 90.31, 89.59]
# y_data_pu = [94.33, 97.38, 97.57, 97.49, 97.13, 97.01]
# y_data_sv = [91.93, 95.75, 96.80, 97.04, 97.83, 96.97]
# plt.plot(x_data, y_data_ip, color='#F5D491', linewidth=2.0, marker='*', label='IP:p2=11')
# plt.plot(x_data, y_data_pu, color='#F9B9B7', linewidth=2.0, marker='d', label='PU:p2=13')
# plt.plot(x_data, y_data_sv, color='#96C9DC', linewidth=2.0, marker='v', label='SV:p2=11')
# plt.title('')
# plt.xlabel('patch sizes of top branch (p1)')
# plt.ylabel('OA (%)')
# plt.legend()
# plt.show()

# 4.siamese改表




# 5.方法表 改 ——
# 6.

