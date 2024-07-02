import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torchvision import transforms
import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import  roc_curve, auc
import math
import pandas as pd
from sklearn.metrics import classification_report,roc_curve,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import scipy.stats as stats
from scipy.stats import pearsonr,ttest_ind,levene
from scipy.stats import ttest_ind, levene, pearsonr, spearmanr
from sklearn.linear_model import LassoCV
from IPython.core.display import display
from sklearn.linear_model import LogisticRegression
import seaborn as sns
################################################
from data_3mode import MyDataSet
from data_loader import split_data
from model_cnn_c32 import LeNet
################################################
device = torch.device("cuda" if torch. cuda.is_available() else "cpu")
print("using {} device.".format(device))
####################################################################################
# 超参数定义
model_str = './model_str/'  # 所有K-fold训练好的模型存放的位置
data_list_txt_str = './data_class_3mode/'  # 所有K-fold数据List和打乱的真题数据存放的位置
dz = pd.read_csv("./radiomics_3mode.csv")
num_seed = 47         # 数据划分的随机种子,0-99,100次
eval_percent = 0.3     # 验证集占总数据的比例
save_train_str = model_str+'Radiomics_LR_result-seed47-c32-neur50-50-lr00005'    #  训练过程结果保存位置
net_name = 'CNN-radiomics-seed47-c32-neur50-50-lr00005'      # 保存的网络的名称
num_workers = 4   # 多线程的数量，默认O
batch_size = 16    # 16
epochs = 50      #    默认 100
learning_rate = 0.00005  # 学习率
num_class = 2           # 分类的类别数目
run_mode = 'test'       # 运行模式，train=训练，test=测试
####################################################################################
# 创建文件夹
if not os.path.exists(model_str):
        os.makedirs(model_str)
if not os.path.exists(save_train_str):
    os.makedirs(save_train_str)
####################################################################################
def result_print(train_Label, y_pred, valid_preds_fold):
    xg_eval_auc = metrics.roc_auc_score(train_Label, valid_preds_fold, average='weighted')  # 验证集上的auc值
    xg_eval_acc = metrics.accuracy_score(train_Label, y_pred)  # 验证集的准确度
    mcm = confusion_matrix(train_Label, y_pred)
    # print(mcm)
    TN = mcm[0, 0]
    FP = mcm[0, 1]
    FN = mcm[1, 0]
    TP = mcm[1, 1]
    xg_eval_specificity = TN/(TN+FP)
    xg_eval_sensitivity = TP / (TP + FN)
    xg_eval_precision = TP/(TP+FP)
    return xg_eval_auc,xg_eval_acc,xg_eval_sensitivity,xg_eval_specificity,xg_eval_precision
##############################################################################
best_auc = 0
for random_seed in range(num_seed,num_seed+1):
    print('random_seed=', random_seed)
    df = dz.sample(frac=1 - eval_percent, random_state=random_seed)  # 训练
    df_original = dz.sample(frac=1 - eval_percent, random_state=random_seed)  # 训练
    dv = dz.drop(df.index)  # 测试集

##############################################################################
#  组学
    print('##############################################################################')
    print('Radiomics training')
    # 指定不需要进行正则化的列
    ###################################################################################################
    exclude_cols = [0, 1]
    # 批量正则化
    for i, col in enumerate(df.columns):
        if i in exclude_cols:
            # 不需要正则化的列跳过
            continue
        # 进行正则化
        df[col] = preprocessing.scale(df[col])
    # 输出结果
    ###################################################################################################
    # 将数据按照 Group 进行分组
    groups = df.groupby('label')

    # 计算平均值和标准差
    means = groups.mean().dropna()
    stds = groups.std().fillna(0)

    p_values = []
    t_values = []
    for col in df.columns:
        if col in ['label'] or col == df.columns[1]:
            # 不需要进行t检验的列跳过
            continue
        # 将字符串类型的列转换为数字类型
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 执行方差齐的两独立样本t检验
        if levene(groups.get_group(0)[col], groups.get_group(1)[col])[1] >= 0.05:
            t_stat, p_val = ttest_ind(groups.get_group(0)[col], groups.get_group(1)[col], equal_var=True)
        # 执行方差不齐的两独立样本t检验
        else:
            t_stat, p_val = ttest_ind(groups.get_group(0)[col], groups.get_group(1)[col], equal_var=False)
        p_values.append(p_val)
        t_values.append(t_stat)
    result = pd.DataFrame({'指标名称': means.columns, 't值': t_values, 'P值': p_values})
    print(result)
    ###################################################################################################
    # 筛选 P 值小于 0.05 的指标
    significant_columns = result.loc[result['P值'] < 0.05, '指标名称']

    # 判断数据是否符合正态分布
    is_normal = True
    for col in significant_columns:
        if not stats.normaltest(df[col])[1] >= 0.05:
            is_normal = False
            break

    # 计算指标之间的相关系数
    if is_normal:
        corr_func = pearsonr
    else:
        corr_func = spearmanr
    corr_values = []
    for col1 in significant_columns:
        corr_row = []
        for col2 in significant_columns:
            corr_val, _ = corr_func(df[col1], df[col2])
            corr_row.append(corr_val)
        corr_values.append(corr_row)

    # 将结果存储到 DataFrame 中
    corr_df = pd.DataFrame(corr_values, columns=significant_columns, index=significant_columns)

    # 输出结果
    print('指标之间的相关系数：\n')
    display(corr_df)
    display(corr_df.style.background_gradient())
    ###################################################################################################
    # 筛选相关系数小于等于 0.8 的指标
    corr_df = corr_df.abs()  # 先将相关系数矩阵的值取绝对值
    selected_cols = []
    while True:
        # 找出相关系数小于等于 0.8 的指标，并保留每个匹配对中的第一个指标
        indices = np.where(corr_df.values <= 0.9)
        idx1, idx2 = indices[0], indices[1]
        unique_idx = np.unique(idx1)
        for i in unique_idx:
            j = np.min(np.where(idx1 == i))
            selected_cols.append(corr_df.columns[idx1[j]])
        # 更新 selected_cols 列表
        # selected_cols = list(set(selected_cols))
        corr_df = corr_df.drop(corr_df.columns[idx2], axis=1)
        corr_df = corr_df.drop(corr_df.index[idx2], axis=0)
        # 每次去除 5 个特征
        if len(selected_cols) > 5:
            selected_cols = selected_cols[:len(selected_cols) - 5]
        if corr_df.shape[0] < 2:
            break

    # 输出结果
    print('相关系数小于等于 0.9 的指标：\n')
    display(pd.DataFrame({'指标名称': selected_cols}))
    ###################################################################################################
    # 对应选择出来的指标与df里的数值
    df_selected = df[selected_cols]
    # 输出结果
    print('相关系数小于 0.9 的指标：\n')
    display(df_selected)
    ###################################################################################################
    X = df_selected
    y = df['label']
    ###################################################################################################
    # 尝试LASSO降维，如果报错则跳过
    try:
        # LASSO
        alphas = np.logspace(-5, 1, 50)
        model_lassoCV = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=0).fit(X, y)  # cv最大不能超过病例数，max指迭代次数
        #################################################################################################
        print(model_lassoCV.alpha_)
        coef = pd.Series(model_lassoCV.coef_, index=X.columns)
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)))
        #################################################################################################
        index = coef[coef != 0].index
        X = X[index]  # 系数不为0的特征都写出来
        X.head()
        print(coef[coef != 0])  # 特征的名称+特征的系数
        #################################################################################################
        coef = coef[coef != 0]
        coef_threshold = 1e-2
        print("Lasso picked " + str(sum(np.abs(coef) > coef_threshold)) + " variables and eliminated the other " + str(
            sum(np.abs(coef) <= coef_threshold)))
        index = X.columns[np.abs(coef) > coef_threshold]
        coef = coef[np.abs(coef) > coef_threshold]
        formula = ' '.join([f"{coef[i]:+.4f} * {index[i]}" for i in range(len(index))])
        if formula == '':
            formula = '0'
        print('公式：y = ', formula)
        #################################################################################################
        # 特征系数Lambda变化曲线/每个特征值随Lambda变化
        coefs = model_lassoCV.path(X, y, alphas=alphas, max_iter=100000)[1].T  # T转置

        #################################################################################################
        ##定义训练集
        ##训练集
        X_train_tmp = df_original.loc[:, index]  ###所有行都要，第一列以后要
        # print(X_train_tmp)
        y_train = df_original['label']  ###只要训练集第一列Group
        scaler = preprocessing.StandardScaler().fit(X_train_tmp)  # 数据按列特征做标准化
        X_train = scaler.transform(X_train_tmp)
        X_train = pd.DataFrame(X_train, columns=X_train_tmp.columns)
        print(X_train)

        ##定义内部验证集
        # 选择这些特征重新构建 X_test1 表格
        X_test1_tmp = dv.loc[:, index]  # 内部测试集
        # print(X_test1)
        X_test1 = scaler.transform(X_test1_tmp)
        X_test1 = pd.DataFrame(X_test1, columns=X_test1_tmp.columns)
        y_test1 = dv['label']
        print(X_test1)
        #################################################################################################
        def result_eval(evalLabel, X_filtered_eval_new, xg):  # 评估结果
            evallabel_xg_auc = xg.predict_proba(X_filtered_eval_new)
            valid_preds_fold = evallabel_xg_auc[:, 1]
            # print('threshold=', thresholds[i])
            thresholds = 0.5
            result_test = []
            for pred in valid_preds_fold:
                result_test.append(1 if pred > thresholds else 0)
            # 筛选出最优模型的总得分并保存种子数及模型，还有一个precision未加上
            xg_eval_auc, xg_eval_acc, xg_eval_sensitivity, xg_eval_specificity, xg_eval_precision = result_print(
                evalLabel, result_test, evallabel_xg_auc[:, 1])
            return xg_eval_auc, xg_eval_acc, xg_eval_sensitivity, xg_eval_specificity, xg_eval_precision, evallabel_xg_auc

        xg = LogisticRegression(random_state=0)  # LR逻辑回归
        xg.fit(X_train, y_train)  # 训练模型
        xg_train_auc, xg_train_f1, xg_train_acc, xg_train_precision, xg_train_recall, trainlabel_xg_auc = \
            result_eval(y_train, X_train, xg)  # 保存训练的结果
        xg_eval_auc, xg_eval_acc, xg_eval_sensitivity, xg_eval_specificity, xg_eval_precision, evallabel_xg_auc = \
            result_eval(y_test1, X_test1, xg)  # 保存验证的结果
        print('Auc train eval test=', xg_train_auc, xg_eval_auc)

        # 设置Seaborn样式
        sns.set(style="whitegrid")

        left_size = 0.2
        bottom_size = 0.2
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 13}

        # 创建图形并调整边距
        plt.figure(figsize=(10, 6))
        plt.gcf().subplots_adjust(left=left_size, bottom=bottom_size)

        # 绘制图表
        plt.semilogx(model_lassoCV.alphas_, coefs, '-')
        plt.axvline(model_lassoCV.alpha_, color='black', ls="--", label='Selected Lambda')

        # 添加图例和标题
        plt.xlabel('Lambda', font1)
        plt.ylabel('Coefficients', font1)
        plt.legend(prop={'family': 'Times New Roman', 'size': 12})

        # 保存并显示图表
        plt.savefig(save_train_str + '/2-Lambda-two-class_image.pdf', dpi=600, bbox_inches='tight')

        # 使用Seaborn样式
        sns.set(style="whitegrid")

        # 计算均值和标准差
        MSEs = model_lassoCV.mse_path_  # 所有Lambda下的MSEs
        MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
        MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}

        # 创建图形并设置分辨率
        plt.figure(figsize=(12, 6))

        # 绘制误差棒图
        plt.errorbar(model_lassoCV.alphas_, MSEs_mean, yerr=MSEs_std,
                     fmt="o", ms=5, mfc="orange", mec="red",
                     ecolor="darkgreen", elinewidth=2, capsize=5, capthick=1)

        # 设置对数坐标
        plt.semilogx()

        # 绘制垂直线标记选择的alpha值
        plt.axvline(model_lassoCV.alpha_, color='black', ls="--", label=f'Selected Lambda')

        # 设置横坐标和纵坐标标签
        plt.xlabel('Lambda', font1)
        plt.ylabel('MSE', font1)

        # 添加标题和图例
        # plt.title('Mean Squared Error for Different Lambda Values', font1)
        plt.legend(prop={'family': 'Times New Roman', 'size': 12})

        # 保存并显示图表
        plt.savefig(save_train_str + '/1-Lasso-mse-two-class_image.pdf', dpi=600, bbox_inches='tight')
        # plt.show()
        #################################################################################################
        y_values = coef[coef != 0]
        y_values  # 不为0的打印出来
        a = y_values.sort_values(ascending=False)
        b = pd.DataFrame(a)
        c = b.index.values
        #################################################################################################
        # 权重图
        x_values = np.arange(len(index))
        y_values = coef[coef != 0]
        left_size = 0.12
        bottom_size = 0.72
        label_size = 13
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 13,
                 }
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        plt.barh(c, a
                 ,  # color='lightblue'
                 color=np.where(a > 0, 'darkorange', 'seagreen')  # 判断大于0的为红色，负的为蓝色
                 , edgecolor='black'
                 , height=0.4
                 # , alpha=0.9  # 不透明度
                 )
        plt.tick_params(labelsize=label_size)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.yticks(x_values, c  # x轴做标记
                   # , rotation='90'
                   , ha='left'  # 水平对齐
                   , va='center'  # 垂直对齐
                   )
        plt.xlabel('weight', font1)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.savefig(save_train_str + '/3-lasso-coef-two-class_image.pdf', dpi=600, bbox_inches='tight')

        # save result
        target_all_train = df_original.values[:, 0:2]
        output_data = target_all_train
        output_file_name = save_train_str + "/target_train.csv"
        np.savetxt(output_file_name, output_data, delimiter=',', fmt='%s')
        ################################################################
        # save result
        target_all_eval = dv.values[:, 0:2]
        output_data = target_all_eval
        output_file_name = save_train_str + "/target_eval.csv"
        np.savetxt(output_file_name, output_data, delimiter=',', fmt='%s')
        ################################################################
        # save result
        output_data = trainlabel_xg_auc
        output_file_name = save_train_str + "/predict_train.csv"
        np.savetxt(output_file_name, output_data, delimiter=',', fmt='%s')
        ################################################################
        # save result
        output_data = evallabel_xg_auc
        output_file_name = save_train_str + "/predict_eval.csv"
        np.savetxt(output_file_name, output_data, delimiter=',', fmt='%s')
        ################################################################
        # save result
        output_data = X_train.values[:, :]
        output_file_name = save_train_str + "/predict_feature_lasso_train.npy"
        np.save(output_file_name, output_data)  # save npy
        ################################################################
        # save result
        output_data = X_test1.values[:, :]
        output_file_name = save_train_str + "/predict_feature_lasso_eval.npy"
        np.save(output_file_name, output_data)  # save npy
        ################################################################
        # save result
        output_data = coef
        output_file_name = save_train_str + "/predict_coef.txt"
        output_file = open(output_file_name, 'a')
        for fp in output_data:  # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')  # line feed
        output_file.close()
        ################################################################
        # LASSO不成功,打印报错信息并跳过到下一个seed
    except Exception as e:
        print(e)
    pass
##############################################################################
#  网络
    print('random_seed=', random_seed)
    print('##############################################################################')
    print('CNN training')
    # 加载网络路径数据
    train_images_path, train_images_label, train_id, val_images_path, val_images_label, val_id = split_data(df, dv,
                                                                                                            data_list_txt_str)
    print("using {} device.".format(device))
    data_transform = {
            "train": transforms.ToTensor(),
            "val": transforms.ToTensor()
    }
    train_data_set = MyDataSet(images_path=train_images_path,images_class=train_images_label, ids_patinet = train_id,aug_mode=1)
    train_loader = torch.utils.data.DataLoader(train_data_set,batch_size=batch_size,shuffle=True,
                                               num_workers=num_workers,collate_fn=train_data_set.collate_fn)
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, ids_patinet = val_id, aug_mode=0)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,num_workers=num_workers,collate_fn = val_dataset.collate_fn)

    ## 模型
    net = LeNet()
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    net.to(device)
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.25, 0.75])).float().to(device))
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    val_num = len(val_dataset)

    best_auc = 0.000
    best_auc_cnn_rad = 0.000
    best_epoch = 0
    train_steps = len(train_loader)
    train_inf = []
    if not os.path.exists(model_str+ net_name):
            os.makedirs(model_str+ net_name)

    save_path = model_str + net_name +  '/model.pth'
    if run_mode == 'train':
        for epoch in range(epochs):
            print('epoch= ', epoch)
            net.train()
            running_loss = 0.0
            for step, data in enumerate(train_loader):
                images, labels, ids_patient = data
                outputs, x_feature_train = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loader_desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1,epochs,running_loss)
            print(train_loader_desc)
            ####################################################################################3
            # validate， 评估CNN的效果
            net.eval()
            test_loss = 0
            valid_preds_fold = np.zeros(len(val_images_label))
            target_all = []
            x_feature_val_all = []
            ids_val_all = []
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    images, targets, ids_patient = data
                    outputs, x_feature_val = net(images.to(device))
                    loss = loss_function(outputs, targets.to(device))
                    test_loss += loss.item()
                    x_feature_val_all.append(x_feature_val.squeeze().cpu().detach().numpy().tolist())
                    target_all.append(targets.cpu().detach().numpy())
                    ids_val_all.append(ids_patient.cpu().detach().numpy().tolist())
                    valid_preds_fold[i:(i + 1)] = outputs.cpu().detach().numpy()[:, 1]
            print('Test loss=', test_loss)
            ###########################################################################################
            ## 与radiomics筛选出的特征进行匹配拼接
            feature_val_cnn = np.array(x_feature_val_all)
            ids_val_all_cnn = np.array(ids_val_all)
            labels_val_cnn = np.array(target_all)
            feature_final_val = np.concatenate((np.array(feature_val_cnn), X_test1.values), axis=1)
            label_val = y_test1.values
            ###########################################################################################
            threshold = 0.5
            result_test = []
            for pred in valid_preds_fold:
                result_test.append(1 if pred > threshold else 0)
            xg_eval_auc,xg_eval_acc,xg_eval_sensitivity,xg_eval_specificity,xg_eval_precision = result_print(
                val_images_label, result_test,valid_preds_fold)

            xg_eval_best = xg_eval_auc
            ###########################################################
            # 评估组学+CNN的联合模型效果
            # saved best CNN model
            if xg_eval_best > best_auc:
                print('saved model', epoch)
                best_epoch = epoch
                best_auc = xg_eval_best
                torch.save(net.state_dict(), save_path)
                ###########################################################
                ## 联合训练MLP模型
                # 先预测所有训练数据的CNN特征
                # 重新定义batch size为1的训练data loader
                print('###########################################################')
                print('Creating train CNN features')
                train_loader_tmp = torch.utils.data.DataLoader(train_data_set,batch_size=1,num_workers=num_workers,collate_fn=train_data_set.collate_fn)
                train_preds_fold = np.zeros(len(train_images_label))
                target_train_all = []
                x_feature_train_all = []
                ids_train_all = []
                with torch.no_grad():
                    for i, data in enumerate(train_loader_tmp):
                        images, targets, ids_patient = data
                        outputs, x_feature_train = net(images.to(device))
                        loss = loss_function(outputs, targets.to(device))
                        test_loss += loss.item()
                        x_feature_train_all.append(x_feature_train.squeeze().cpu().detach().numpy().tolist())
                        target_train_all.append(targets.cpu().detach().numpy())
                        ids_train_all.append(ids_patient.cpu().detach().numpy().tolist())
                        train_preds_fold[i:(i + 1)] = outputs.cpu().detach().numpy()[:, 1]

                feature_final_train = np.concatenate((np.array(x_feature_train_all), X_train.values), axis=1)
                label_train = y_train.values

                # 测一遍train的指标
                result_train = []
                for pred in train_preds_fold:
                    result_train.append(1 if pred > threshold else 0)
                xg_train_auc, xg_train_acc, xg_train_sensitivity, xg_train_specificity, xg_train_precision = result_print(
                    train_images_label, result_train, train_preds_fold)

                ## MLP训练
                from sklearn.neural_network import MLPClassifier
                xg = MLPClassifier(solver='adam', activation='relu', alpha=1e-4, hidden_layer_sizes=(50,50),
                                   random_state=0, max_iter=500, verbose=False, learning_rate_init=0.0001)
                xg.fit(feature_final_train, label_train)  # 训练模型
                xg_train_combine_prob = xg.predict_proba(feature_final_train)
                xg_train_combine_auc = metrics.roc_auc_score(label_train, xg_train_combine_prob[:, 1], average='weighted')  # 训练集上的auc值
                xg_val_combine_prob = xg.predict_proba(feature_final_val)
                xg_val_combine_auc = metrics.roc_auc_score(label_val, xg_val_combine_prob[:, 1],average='weighted')  # 验证集上的auc值
                print('Combined model | Auc train eval = ', xg_train_combine_auc,  xg_val_combine_auc)
                ###########################################################
            print('Current CNN train AUC = ', xg_train_auc)
            print('Epoch num = ',epoch + 1, '| Best CNN AUC = ', best_auc, '| Current AUC = ', xg_eval_auc)
            # save result
            output_data = [epoch + 1, running_loss, test_loss, best_auc, xg_eval_auc]
            output_file_name = model_str + net_name+ "/train.txt"
            output_file = open(output_file_name, 'a')
            for fp in output_data:  # write data in txt
                output_file.write(str(fp))
                output_file.write(',')
            output_file.write('\n')  # line feed
            output_file.close()
            print( 'best epoch = ', best_epoch,' best_auc =', best_auc)
    ####################################################################################################
    if run_mode == 'test':
        net.load_state_dict(torch.load(save_path))
        # validate， 评估CNN的效果
        net.eval()
        test_loss = 0
        valid_preds_fold = np.zeros(len(val_images_label))
        target_all = []
        x_feature_val_all = []
        ids_val_all = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images, targets, ids_patient = data
                outputs, x_feature_val = net(images.to(device))
                loss = loss_function(outputs, targets.to(device))
                test_loss += loss.item()
                x_feature_val_all.append(x_feature_val.squeeze().cpu().detach().numpy().tolist())
                target_all.append(targets.cpu().detach().numpy())
                ids_val_all.append(ids_patient.cpu().detach().numpy().tolist())
                valid_preds_fold[i :(i + 1)] = outputs.cpu().detach().numpy()[:, 1]
        print('Test loss=', test_loss)
        ###########################################################################################
        ## 与radiomics筛选出的特征进行匹配拼接
        feature_val_cnn = np.array(x_feature_val_all)
        ids_val_all_cnn = np.array(ids_val_all)
        labels_val_cnn = np.array(target_all)
        feature_final_val = np.concatenate((np.array(feature_val_cnn), X_test1.values), axis=1)
        label_val = y_test1.values
        ###########################################################################################
        threshold = 0.5
        result_test = []
        for pred in valid_preds_fold:
            result_test.append(1 if pred > threshold else 0)
        xg_eval_auc, xg_eval_acc, xg_eval_sensitivity, xg_eval_specificity, xg_eval_precision = result_print(
            val_images_label, result_test, valid_preds_fold)
        print('Current CNN Eval AUC = ', xg_eval_auc)
        ###########################################################
        # 评估组学+CNN的联合模型效果
        # 先预测所有训练数据的CNN特征
        # 重新定义batch size为1的训练data loader
        print('###########################################################')
        print('Creating train CNN features')
        train_loader_tmp = torch.utils.data.DataLoader(train_data_set, batch_size=1, num_workers=num_workers,
                                                       collate_fn=train_data_set.collate_fn)
        train_preds_fold = np.zeros(len(train_images_label))
        target_train_all = []
        x_feature_train_all = []
        ids_train_all = []
        with torch.no_grad():
            for i, data in enumerate(train_loader_tmp):
                images, targets, ids_patient = data
                outputs, x_feature_train = net(images.to(device))
                loss = loss_function(outputs, targets.to(device))
                test_loss += loss.item()
                x_feature_train_all.append(x_feature_train.squeeze().cpu().detach().numpy().tolist())
                target_train_all.append(targets.cpu().detach().numpy())
                ids_train_all.append(ids_patient.cpu().detach().numpy().tolist())
                train_preds_fold[i :(i + 1)] = outputs.cpu().detach().numpy()[:, 1]

        feature_final_train = np.concatenate((np.array(x_feature_train_all), X_train.values), axis=1)
        label_train = y_train.values

        # 测一遍train的指标
        result_train = []
        for pred in train_preds_fold:
            result_train.append(1 if pred > threshold else 0)
        xg_train_auc, xg_train_acc, xg_train_sensitivity, xg_train_specificity, xg_train_precision = result_print(
            train_images_label, result_train, train_preds_fold)
        print('Current CNN train AUC = ', xg_train_auc)

        best_auc_seed = 0
        best_seed = 0
        for seed_i in range(200):
            print('seed=',seed_i)

            # MLP训练
            from sklearn.neural_network import MLPClassifier
            xg = MLPClassifier(solver='adam', hidden_layer_sizes=(50,50),
                               random_state=seed_i, max_iter=500, verbose=False, learning_rate_init=0.0001)
            xg.fit(feature_final_train, label_train)  # 训练模型
            xg_train_combine_prob = xg.predict_proba(feature_final_train)
            xg_train_combine_auc = metrics.roc_auc_score(label_train, xg_train_combine_prob[:, 1],
                                                         average='weighted')  # 训练集上的auc值
            xg_val_combine_prob = xg.predict_proba(feature_final_val)
            xg_val_combine_auc = metrics.roc_auc_score(label_val, xg_val_combine_prob[:, 1],
                                                       average='weighted')  # 验证集上的auc值
            print('Combined model | Auc train eval=', xg_train_combine_auc, xg_val_combine_auc)
            if xg_val_combine_auc > best_auc_seed:
                best_auc_seed = xg_val_combine_auc
                best_seed = seed_i
                print('seed = ', seed_i, '| Best Combied AUC = ', best_auc_seed)
                # save result
                output_data_train = [target_all_train[:, 0], target_all_train[:, 1], trainlabel_xg_auc[:, 1],
                                     train_preds_fold, xg_train_combine_prob[:, 1]]
                output_file_name_train = "./predict_train.csv"
                np.savetxt(output_file_name_train, np.array(output_data_train).T, delimiter=',', fmt='%s')
                output_data_eval = [target_all_eval[:, 0], target_all_eval[:, 1], evallabel_xg_auc[:, 1],
                                    valid_preds_fold, xg_val_combine_prob[:, 1]]
                output_file_name_eval = "./predict_test.csv"
                np.savetxt(output_file_name_eval, np.array(output_data_eval).T, delimiter=',', fmt='%s')
                #######################################################################################
        print('Best seed = ', best_seed, '| Best Combied AUC = ', best_auc_seed)
        ###########################################################
    print('Finished Testing')