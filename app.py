from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from feature_extraction.seqtolist import seqtolist
app = Flask(__name__)
import numpy as np
import pandas as pd
import os
import zipfile

from feature_extraction.dna.DAC import dna_dac
from feature_extraction.dna.DACC import dna_dacc
from feature_extraction.dna.DCC import dna_dcc
from feature_extraction.dna.Kmer import dna_kmer
from feature_extraction.dna.revKmer import dna_revkmer
from feature_extraction.dna.Onehot import dna_onehot
from feature_extraction.dna.TAC import dna_tac
from feature_extraction.dna.TCC import dna_tcc
from feature_extraction.dna.TACC import dna_tacc


from feature_extraction.rna.DAC import rna_dac
from feature_extraction.rna.DACC import rna_dacc
from feature_extraction.rna.DCC import rna_dcc
from feature_extraction.rna.GAC import rna_gac
from feature_extraction.rna.Kmer import rna_kmer
from feature_extraction.rna.MAC import rna_mac
from feature_extraction.rna.mismatch import rna_mismatch
from feature_extraction.rna.NMBAC import rna_nmbac
from feature_extraction.rna.Onehot import rna_onehot
from feature_extraction.rna.PC_PseDNC_General import rna_pc_psednc_general
from feature_extraction.rna.SC_PseDNC_General import rna_sc_psednc_general
from feature_extraction.rna.subsequence import rna_subsequence
from feature_extraction.rna.revKmer import rna_revkmer

from feature_extraction.pro.Onehot import pro_onehot
from feature_extraction.pro.AC import pro_ac
from feature_extraction.pro.ACC import pro_acc
from feature_extraction.pro.CC import pro_cc
from feature_extraction.pro.SC_PseAAC import pro_SC_PseAAC
from feature_extraction.pro.PC_PseAAC import pro_PC_PseAAC
from feature_extraction.pro.PDT import pro_PDT
from feature_extraction.pro.PC_PseAAC_General import pro_PC_PseAACG
from feature_extraction.pro.SC_PseAAC_General import pro_SC_PseAACG
from feature_extraction.pro.AAC import pro_aac
from feature_extraction.pro.CHHAA import pro_chhaa
from feature_extraction.pro.DR import pro_dr
from feature_extraction.pro.DistancePair import pro_DistancePair
from feature_extraction.pro.k_mer import pro_kmer

# 主页
@app.route('/')
def home():
    return render_template('home.html')


# 结果页面
@app.route('/Result')
def result():
    return render_template('ResultShow.html')


# 下载示例文件函数
@app.route('/feature_extraction/<path:filename>')
def feature_extraction(filename):
    # 定义文件夹路径
    folder_path = os.path.join('feature_extraction', filename)

    # 定义压缩包路径
    zip_path = os.path.join('feature_extraction', f'{filename}.zip')

    # 将文件夹打包成 .zip 文件
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

    # 发送压缩文件给用户
    return send_file(zip_path, as_attachment=True)


# DNA页面
@app.route('/dna', methods=['GET', 'POST'])
def dna():
    return render_template('dna.html')


# DNA页面
@app.route('/rna', methods=['GET', 'POST'])
def rna():
    return render_template('rna.html')


# Protein页面
@app.route('/protein', methods=['GET', 'POST'])
def protein():
    return render_template('protein.html')


# 处理序列进行特征提取
@app.route('/process_seqs', methods=['POST'])
def process_seqs():
    # 1.收集数据，将数据变成四个list(train_data, train_label, test_data, test_label)
    text_area_content = request.form.get('Textarea')
    file = request.files.get('file_input')
    train_datao, train_label, test_datao, test_label = seqtolist(text_area_content, file)
    train_datao = [s.upper() for s in train_datao]
    test_datao = [s.upper() for s in test_datao]
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    # 获取选定的序列类型与特征提取算法
    type = request.form.get('seq_type')
    selected_algorithm = request.form.get('algorithm')
    print(f"type: {type}, algorithm:{selected_algorithm}")
    if type == 'dna':
        if selected_algorithm == 'DAC':
            train_data = dna_dac(train_datao, request)
            test_data = dna_dac(test_datao, request)

        elif selected_algorithm == 'DACC':
            train_data = dna_dacc(train_datao, request)
            test_data = dna_dacc(test_datao, request)

        elif selected_algorithm == 'DCC':
            train_data = dna_dcc(train_datao, request)
            test_data = dna_dcc(test_datao, request)

        elif selected_algorithm == 'TAC':
            train_data = dna_tac(train_datao, request)
            test_data = dna_tac(test_datao, request)

        elif selected_algorithm == 'Kmer':
            train_data = dna_kmer(train_datao, request)
            test_data = dna_kmer(test_datao, request)

        elif selected_algorithm == 'Revkmer':
            train_data = dna_revkmer(train_datao, request)
            test_data = dna_revkmer(test_datao, request)

        elif selected_algorithm == 'Onehot':
            train_data = dna_onehot(train_datao, request)
            test_data = dna_onehot(test_datao, request)

        elif selected_algorithm == 'TCC':
            train_data = dna_tcc(train_datao, request)
            test_data = dna_tcc(test_datao, request)

        elif selected_algorithm == 'TACC':
            train_data = dna_tacc(train_datao, request)
            test_data = dna_tacc(test_datao, request)
        else:
            print(f"{selected_algorithm} 没有实现")

    elif type == 'rna':
        print('rna')
        if selected_algorithm == 'DAC':
            train_data = rna_dac(train_datao, request)
            test_data = rna_dac(test_datao, request)

        elif selected_algorithm == 'DACC':
            train_data = rna_dacc(train_datao, request)
            test_data = rna_dacc(test_datao, request)

        elif selected_algorithm == 'DCC':
            train_data = rna_dcc(train_datao, request)
            test_data = rna_dcc(test_datao, request)

        elif selected_algorithm == 'GAC':
            train_data = rna_gac(train_datao, request)
            test_data = rna_gac(test_datao, request)

        elif selected_algorithm == 'Kmer':
            train_data = rna_kmer(train_datao, request)
            test_data = rna_kmer(test_datao, request)

        elif selected_algorithm == 'Revkmer':
            train_data = rna_revkmer(train_datao, request)
            test_data = rna_revkmer(test_datao, request)

        elif selected_algorithm == 'MAC':
            train_data = rna_mac(train_datao, request)
            test_data = rna_mac(test_datao, request)

        elif selected_algorithm == 'MisMatch':
            train_data = rna_mismatch(train_datao, request)
            test_data = rna_mismatch(test_datao, request)

        elif selected_algorithm == 'NMBAC':
            train_data = rna_nmbac(train_datao, request)
            test_data = rna_nmbac(test_datao, request)

        elif selected_algorithm == 'Onehot':
            train_data = rna_onehot(train_datao, request)
            test_data = rna_onehot(test_datao, request)

        elif selected_algorithm == 'PC_PseDNC_General':
            train_data = rna_pc_psednc_general(train_datao, request)
            test_data = rna_pc_psednc_general(test_datao, request)

        elif selected_algorithm == 'SC_PseDNC_General':
            train_data = rna_sc_psednc_general(train_datao, request)
            test_data = rna_sc_psednc_general(test_datao, request)

        elif selected_algorithm == 'SubSequence':
            train_data = rna_subsequence(train_datao, request)
            test_data = rna_subsequence(test_datao, request)

        else:
            print(f"{selected_algorithm} 没有实现")

    else:
        print('pro')
        if selected_algorithm == 'OneHot':
            train_data = pro_onehot(train_datao, request)
            test_data = pro_onehot(test_datao, request)
        elif selected_algorithm == 'AC':
            train_data = pro_ac(train_datao, request)
            test_data = pro_ac(test_datao, request)
        elif selected_algorithm == 'AAC':
            train_data = pro_aac(train_datao, request)
            test_data = pro_aac(test_datao, request)
        elif selected_algorithm == 'DR':
            train_data = pro_dr(train_datao, request)
            test_data = pro_dr(test_datao, request)
        elif selected_algorithm == 'CHHAA':
            train_data = pro_chhaa(train_datao, request)
            test_data = pro_chhaa(test_datao, request)
        elif selected_algorithm == 'ACC':
            train_data = pro_acc(train_datao, request)
            test_data = pro_acc(test_datao, request)
        elif selected_algorithm == 'CC':
            train_data = pro_cc(train_datao, request)
            test_data = pro_cc(test_datao, request)
        elif selected_algorithm == 'PDT':
            train_data = pro_PDT(train_datao, request)
            test_data = pro_PDT(test_datao, request)
        elif selected_algorithm == 'PCPseAAC':
            train_data = pro_PC_PseAAC(train_datao, request)
            test_data = pro_PC_PseAAC(test_datao, request)
        elif selected_algorithm == 'SCPseAAC':
            train_data = pro_SC_PseAAC(train_datao, request)
            test_data = pro_SC_PseAAC(test_datao, request)
        elif selected_algorithm == 'PCPseAACGeneral':
            train_data = pro_PC_PseAACG(train_datao, request)
            test_data = pro_PC_PseAACG(test_datao, request)
        elif selected_algorithm == 'SCPseAACGeneral':
            train_data = pro_SC_PseAACG(train_datao, request)
            test_data = pro_SC_PseAACG(test_datao, request)
        elif selected_algorithm == 'DistancePair':
            train_data = pro_DistancePair(train_datao, request)
            test_data = pro_DistancePair(test_datao, request)
        elif selected_algorithm == 'Kmer':
            train_data = pro_kmer(train_datao, request)
            test_data = pro_kmer(test_datao, request)
        else:
            train_data = train_datao
            test_data = test_datao
            print(f"{selected_algorithm} 没有实现")

    # 保存train_data, train_label, test_data, test_label进入特定目录。
    train_label_df = pd.DataFrame(train_label)
    test_label_df = pd.DataFrame(test_label)
    train_data_df = pd.DataFrame(train_data)
    test_data_df = pd.DataFrame(test_data)

    # 保存为 CSV 文件
    save_path = 'feature_extraction/after_feature_data/'
    train_label_df.to_csv(save_path + 'train_labels.csv', index=False, header=False)
    test_label_df.to_csv(save_path + 'test_labels.csv', index=False, header=False)
    train_data_df.to_csv(save_path + 'train_data.csv', index=False, header=False)
    test_data_df.to_csv(save_path + 'test_data.csv', index=False, header=False)
    # 定向到 sampling 页面
    return redirect(url_for('sampling'))


# 处理特征提取后的文件
@app.route('/process_files', methods=['POST'])
def process_files():
    train_data = request.files.get('file_input_traindata')
    train_labels = request.files.get('file_input_trainlables')
    test_data = request.files.get('file_input_testdata')
    test_labels = request.files.get('file_input_testlables')

    # 检查文件是否都上传
    if not all([train_data, train_labels, test_data, test_labels]):
        return jsonify({"success": False, "message": "All four files are required."})
        # 定义文件检查结果
    errors = {}

    # 加载文件数据
    try:
        train_data_df = pd.read_csv(train_data)
        train_labels_df = pd.read_csv(train_labels)
        test_data_df = pd.read_csv(test_data)
        test_labels_df = pd.read_csv(test_labels)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error reading files: {str(e)}"})

    # 1. 检查标签文件：确保只有1列，数据为整数，且train_labels有至少2种不同的标签
    def validate_labels(label_df, file_name):
        if label_df.shape[1] != 1:
            return f"{file_name} must contain exactly one column."
        if not pd.api.types.is_integer_dtype(label_df.iloc[:, 0]):
            return f"{file_name} must contain integer values only."
        return ""

    label_errors = {
        "train_labels": validate_labels(train_labels_df, "train_labels"),
        "test_labels": validate_labels(test_labels_df, "test_labels")
    }

    # 检查train_labels至少包含两种不同的标签
    if len(train_labels_df.iloc[:, 0].unique()) < 2:
        label_errors["train_labels"] = "train_labels must have at least two distinct labels."
    errors.update({k: v for k, v in label_errors.items() if v})

    # 2. 检查数据文件维度是否一致
    if train_data_df.shape[1] != test_data_df.shape[1]:
        errors["data_dimension"] = "train_data and test_data must have the same number of columns."

    # 3. 检查数据和标签行数是否匹配
    if train_data_df.shape[0] != train_labels_df.shape[0]:
        errors["train_data_labels"] = "The number of rows in train_data must match train_labels."
    if test_data_df.shape[0] != test_labels_df.shape[0]:
        errors["test_data_labels"] = "The number of rows in test_data must match test_labels."

    # 如果存在任何错误，返回错误信息
    if errors:
        return jsonify({"success": False, "errors": errors})

    # 保存为 CSV 文件
    save_path = 'feature_extraction/after_feature_data/'
    train_labels_df.to_csv(save_path + 'train_labels.csv', index=False, header=False)
    test_labels_df.to_csv(save_path + 'test_labels.csv', index=False, header=False)
    train_data_df.to_csv(save_path + 'train_data.csv', index=False, header=False)
    test_data_df.to_csv(save_path + 'test_data.csv', index=False, header=False)

    # return jsonify({"success": True, "message": "All files passed validation."})
    return redirect(url_for('sampling'))


# 跳转到重采样界面
@app.route('/sampling')
def sampling():
    data = get_data_after_Feature_extraction()
    return render_template('sampling.html', data=data)


# 根据选择的重采样算法，跳转到不同的采样输入参数界面上
@app.route('/load_algorithm/<algorithm_name>')
def load_algorithm(algorithm_name):
    data = get_data_after_Feature_extraction()

    if algorithm_name == 'OverRandom':
        return render_template('resampling_alg/OverRandom.html', data=data)
    elif algorithm_name == 'SMOTE':
        return render_template('resampling_alg/SMOTE.html', data=data)
    elif algorithm_name == 'UnderRandom':
        return render_template('resampling_alg/UnderRandom.html', data=data)
    elif algorithm_name == 'EPDCC':
        return render_template('resampling_alg/EPDCC.html', data=data)
    elif algorithm_name == 'SMOTE':
        return render_template('resampling_alg/SMOTE.html', data=data)
    elif algorithm_name == 'ClusterCentroids':
        return render_template('resampling_alg/ClusterCentroids.html', data=data)
    elif algorithm_name == 'NearMiss':
        return render_template('resampling_alg/NearMiss.html', data=data)
    elif algorithm_name == 'TomekLinks':
        return render_template('resampling_alg/TomekLinks.html', data=data)
    elif algorithm_name == 'One_sided_selection':
        return render_template('resampling_alg/OneSidedSelection.html', data=data)
    elif algorithm_name == 'NCR':
        return render_template('resampling_alg/NCR.html', data=data)
    elif algorithm_name == 'KSU':
        return render_template('resampling_alg/KSU.html', data=data)
    elif algorithm_name == 'ADASYN':
        return render_template('resampling_alg/ADASYN.html', data=data)
    elif algorithm_name == 'KPCA':
        return render_template('resampling_alg/KPCA.html', data=data)
    elif algorithm_name == 'ENN':
        return render_template('resampling_alg/ENN.html', data=data)
    elif algorithm_name == 'MDO':
        return render_template('resampling_alg/MDO.html', data=data)
    elif algorithm_name == 'MDNDO':
        return render_template('resampling_alg/MDNDO.html', data=data)
    elif algorithm_name == 'Interpolation':
        return render_template('resampling_alg/Interpolation.html', data=data)
    elif algorithm_name == 'BorderlineSMOTE':
        return render_template('resampling_alg/BorderlineSMOTE.html', data=data)
    elif algorithm_name == 'SVMSMOTE':
        return render_template('resampling_alg/SVMSMOTE.html', data=data)
    elif algorithm_name == 'KMeansSMOTE':
        return render_template('resampling_alg/KMeansSMOTE.html', data=data)
    elif algorithm_name == 'SMOTEN':
        return render_template('resampling_alg/SMOTEN.html', data=data)
    elif algorithm_name == 'ProWSyn':
        return render_template('resampling_alg/ProWSyn.html', data=data)
    elif algorithm_name == 'RASLE':
        return render_template('resampling_alg/RASLE.html', data=data)
    elif algorithm_name == 'WGANBased':
        return render_template('resampling_alg/WGANBased.html', data=data)
    elif algorithm_name == 'NoiseSMOTE':
        return render_template('resampling_alg/NoiseSMOTE.html', data=data)
    elif algorithm_name == 'RENN':
        return render_template('resampling_alg/RENN.html', data=data)
    elif algorithm_name == 'AllKNN':
        return render_template('resampling_alg/AllKNN.html', data=data)
    elif algorithm_name == 'IHT':
        return render_template('resampling_alg/IHT.html', data=data)
    elif algorithm_name == 'CNN':
        return render_template('resampling_alg/CNN.html', data=data)
    elif algorithm_name == 'EasyEnsemble':
        return render_template('resampling_alg/EasyEnsemble.html', data=data)
    elif algorithm_name == 'NC':
        return render_template('resampling_alg/NC.html', data=data)
    elif algorithm_name == 'ERS':
        return render_template('resampling_alg/ERS.html', data=data)
    elif algorithm_name == 'Fuzzy':
        return render_template('resampling_alg/Fuzzy.html', data=data)
    else:
        print("无采样算法")


from models.train_by_model import train_by_model
from samplings.over.Random import OverRandom
from samplings.over.SMOTE import OverSmote
from samplings.over.KPCA import OverKPCA
from samplings.over.ADASYN import Overadasyn
from samplings.over.MDO import OverMDO
from samplings.over.MDNDO import OverMDNDO
from samplings.over.Interpolation import OverInterpolation
from samplings.over.BorderlineSMOTE import OverBorderlineSMOTE
from samplings.over.KMeansSMOTE import OverKMeansSMOTE
from samplings.over.SVMSMOTE import OverSVMSMOTE
from samplings.over.SMOTEN import OverSMOTEN
from samplings.over.NoiseSMOTE import OverNoiseSMOTE
from samplings.over.ProWSyn import OverProWSyn
from samplings.over.RASLE import OverRASLE
from samplings.over.WGAN_Based import OverWGAN

from samplings.under.Random import UnderRandom
from samplings.under.EPDCC import UnderEPCCC
from samplings.under.ClusterCentroids import UnderClusterCentroids
from samplings.under.NearMiss import UnderNearMiss
from samplings.under.TomekLinks import UnderTomekLinks
from samplings.under.OSS import UnderOSS
from samplings.under.NCR import UnderNCR
from samplings.under.KSU import UnderKSU
from samplings.under.EditedNearestNeighbours import UnderENN
from samplings.under.RENN import UnderRENN
from samplings.under.ALLKNN import UnderALLKNN
from samplings.under.IHT import UnderIHT
from samplings.under.CNN import UnderCNN
from samplings.under.NC import UnderNC
from samplings.under.ERS import UnderERS
from samplings.under.Fuzzy import UnderFuzzy


# 进行采样模型处理
@app.route('/data_resampling', methods=['POST'])
def data_resampling():
    train_data = pd.read_csv("feature_extraction/after_feature_data/train_data.csv", header=None).to_numpy()
    train_labels = pd.read_csv("feature_extraction/after_feature_data/train_labels.csv", header=None).to_numpy()
    selected_algorithm = request.form.get('algorithm')
    print(f"algorithm:{selected_algorithm}")
    if selected_algorithm == 'OverRandom':
        n_data, n_labels = OverRandom(request, train_data, train_labels)
    elif selected_algorithm == 'UnderRandom':
        n_data, n_labels = UnderRandom(request, train_data, train_labels)
    elif selected_algorithm == 'EPDCC':
        n_data, n_labels = UnderEPCCC(request, train_data, train_labels)
    elif selected_algorithm == 'SMOTE':
        n_data, n_labels = OverSmote(request, train_data, train_labels)
    elif selected_algorithm == 'ClusterCentroids':
        n_data, n_labels = UnderClusterCentroids(request, train_data, train_labels)
    elif selected_algorithm == 'NearMiss':
        n_data, n_labels = UnderNearMiss(request, train_data, train_labels)
    elif selected_algorithm == 'TomekLinks':
        n_data, n_labels = UnderTomekLinks(request, train_data, train_labels)
    elif selected_algorithm == 'One_sided_selection':
        n_data, n_labels = UnderOSS(request, train_data, train_labels)
    elif selected_algorithm == 'NCR':
        n_data, n_labels = UnderNCR(request, train_data, train_labels)
    elif selected_algorithm == 'KSU':
        n_data, n_labels = UnderKSU(request, train_data, train_labels)
    elif selected_algorithm == 'ADASYN':
        n_data, n_labels = Overadasyn(request, train_data, train_labels)
    elif selected_algorithm == 'KPCA':
        n_data, n_labels = OverKPCA(request, train_data, train_labels)
    elif selected_algorithm == 'ENN':
        n_data, n_labels = UnderENN(request, train_data, train_labels)
    elif selected_algorithm == 'MDO':
        n_data, n_labels = OverMDO(request, train_data, train_labels)
    elif selected_algorithm == 'MDNDO':
        n_data, n_labels = OverMDNDO(request, train_data, train_labels)
    elif selected_algorithm == 'Interpolation':
        n_data, n_labels = OverInterpolation(request, train_data, train_labels)
    elif selected_algorithm == 'BorderlineSMOTE':
        n_data, n_labels = OverBorderlineSMOTE(request, train_data, train_labels)
    elif selected_algorithm == 'KMeansSMOTE':
        n_data, n_labels = OverKMeansSMOTE(request, train_data, train_labels)
    elif selected_algorithm == 'SVMSMOTE':
        n_data, n_labels = OverSVMSMOTE(request, train_data, train_labels)
    elif selected_algorithm == 'SMOTEN':
        n_data, n_labels = OverSMOTEN(request, train_data, train_labels)
    elif selected_algorithm == 'NoiseSMOTE':
        n_data, n_labels = OverNoiseSMOTE(request, train_data, train_labels)
    elif selected_algorithm == 'ProWSyn':
        n_data, n_labels = OverProWSyn(request, train_data, train_labels)
    elif selected_algorithm == 'RASLE':
        n_data, n_labels = OverRASLE(request, train_data, train_labels)
    elif selected_algorithm == 'WGANBased':
        n_data, n_labels = OverWGAN(request, train_data, train_labels)
    elif selected_algorithm == 'RENN':
        n_data, n_labels = UnderRENN(request, train_data, train_labels)
    elif selected_algorithm == 'ALLKNN':
        n_data, n_labels = UnderALLKNN(request, train_data, train_labels)
    elif selected_algorithm == 'IHT':
        n_data, n_labels = UnderIHT(request, train_data, train_labels)
    elif selected_algorithm == 'NC':
        n_data, n_labels = UnderNC(request, train_data, train_labels)
    elif selected_algorithm == 'ERS':
        n_data, n_labels = UnderERS(request, train_data, train_labels)
    elif selected_algorithm == 'CNN':
        n_data, n_labels = UnderCNN(request, train_data, train_labels)
    elif selected_algorithm == 'Fuzzy':
        n_data, n_labels = UnderFuzzy(request, train_data, train_labels)
    else:
        print("无该采样算法")
    print(n_data.shape)
    # 进行模型处理
    # 获得的信息：样本结果，样本的分类结果，分类指标正确率.
    acc, model = train_by_model(n_data, n_labels, request)

    # 定向到 sampling 页面
    train_class_counts = dict(zip(*np.unique(n_labels, return_counts=True)))
    # 将数据传递给前端模板

    # 保存路径
    save_dir = "feature_extraction/after_sampling"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存n_data到CSV文件
    n_data_file = os.path.join(save_dir, "data.csv")
    pd.DataFrame(n_data).to_csv(n_data_file, index=False, header=False)
    # 保存n_labels到CSV文件
    n_labels_file = os.path.join(save_dir, "labels.csv")
    pd.DataFrame(n_labels).to_csv(n_labels_file, index=False, header=False)

    data = {
        'acc': acc,
        'model': model,
        'train_class_counts': train_class_counts,
        'selected_algorithm': selected_algorithm
    }

    return render_template('ResultShow.html', data=data)


# 处理特征提取csv文件，记录其信息
def get_data_after_Feature_extraction():
    train_data = pd.read_csv("feature_extraction/after_feature_data/train_data.csv", header=None).to_numpy()
    test_data = pd.read_csv("feature_extraction/after_feature_data/test_data.csv", header=None).to_numpy()
    train_labels = pd.read_csv("feature_extraction/after_feature_data/train_labels.csv", header=None).to_numpy()
    test_labels = pd.read_csv("feature_extraction/after_feature_data/test_labels.csv", header=None).to_numpy()

    # 计算所需信息
    num_classes = len(np.unique(train_labels))  # 类别数 (train_labels 中的唯一值个数)
    train_shape = train_data.shape  # 训练数据的输入维度
    test_shape = test_data.shape  # 测试数据的输入维度

    # 各类别的样本数
    train_class_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
    test_class_counts = dict(zip(*np.unique(test_labels, return_counts=True)))

    # 将数据传递给前端模板
    data = {
        'num_classes': int(num_classes),
        'train_shape': train_shape,
        'train_class_counts': train_class_counts,
        'test_shape': test_shape,
        'test_class_counts': test_class_counts,
        'num_classes_1': int(num_classes)-1,
    }
    return data


# 下载采样后的数据
@app.route('/download_sample_data')
def download_sample_data():
    # 需要打包的文件夹路径
    folder_path = 'feature_extraction/after_sampling'
    # 压缩文件的路径
    zip_file_path = 'feature_extraction/after_sampling.zip'

    # 如果ZIP文件已经存在，先删除旧文件
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)

    # 创建ZIP文件
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 将文件添加到ZIP文件中
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

    # 使用Flask的send_file返回ZIP文件
    return send_file(zip_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
