#Dont change this code
import os 
import re 
import time
import random 
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import KFold
import tensorflow as tf
def main():
   # np.random.seed(42) 
    seed_value=420
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    strt=time.time()
    #hypm={'m': 62, 'activation': 'relu', 'learning_rate': 5e-05, 'epochs': 150, 'batch_size': 160, 'loss': 'mse', 'lat': 0.025, 'regularization_type': 'l1', 'regularization_strength': 5e-05}    
    hypm={'m': 62, 'activation': 'relu', 'learning_rate': 5e-05, 'epochs': 150, 'batch_size': 128, 'loss': 'mse', 'lat': 0.025, 'regularization_type': 'l2', 'regularization_strength': 0.0001}
    i=hypm['m']
    activation=hypm['activation']
    lr=hypm['learning_rate']
    epochs=hypm['epochs']
    batch_size=hypm['batch_size']
    loss=hypm['loss']
    lat=hypm['lat']
    regularization_type=hypm['regularization_type']
    regularization_strength=hypm['regularization_strength']

    data = pd.read_csv('data.csv')
    data=data.dropna()
    data_old=data
    text_file= open("ftrs.txt", "r")
    feature_sorted=text_file.read()
    feat_columns=feature_sorted.split("\n")

    target1=[k for k in data['poisson_ratio']]

    data=data[feat_columns[0:174]+['poisson_ratio']]
    index1=data[data['poisson_ratio']<0.0].index
    negdf=data.loc[data['poisson_ratio'] < 0.0]
    data.drop(index1,inplace=True) #negetives dropped
    
    # K folds validation 
    nfoldss = 4
    scores = []
    counts = []
    precisions = []
    cutoffs = []
    thresholds=[]
    for cts in range(1):
        kf = KFold(n_splits=nfoldss, shuffle=True)
        fold_num = 1
        for train_idx, test_idx in kf.split(data):
            print(f"Fold {fold_num}")
            data1 = data.iloc[train_idx]
            data2 = data.iloc[test_idx]
            frames = [data2, negdf]
            hold = pd.concat(frames)
            del data1["poisson_ratio"]
            del data1["mat_id"]
            del data1["composition"]
            reconstruction_fold, target_fold, mat_id_fold = auto_enc(data1[feat_columns[2:i]], hold[feat_columns[0:i]+['poisson_ratio']], i-2, activation, lr, epochs, batch_size, loss, lat, regularization_type, regularization_strength)
            score, count, prec, cutoff,thresh = anomaly_score(reconstruction_fold, target_fold)
           # print(len(target_fold), len(mat_id_fold))
            plot_r(reconstruction_fold,target_fold,cutoff,prec)
            print("Precision :",prec)
            #print(score, count, prec, cutoff,thresh)
            scores.append(score)
            counts.append(count)
            precisions.append(prec)
            cutoffs.append(cutoff)
            thresholds.append(thresh)
            fold_num += 1
    avg_score = np.mean(scores)
    avg_count = np.mean(counts)
    avg_prec = np.mean(precisions)
    avg_cutoff = np.mean(cutoffs)
    avg_thresh=np.mean(thresholds)
    plt.hist(thresholds,len(thresholds))

    print("Average F1 score :", avg_score)
    print("Average Precision :", avg_prec)
    print('time taken',time.time()-strt)

def auto_enc(X, hold, m, activation, lr, epochs, batch_size, loss, lat, regularization_type, regularization_strength):
    hold1 = hold
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    input_dim = X.shape[1]

    reg = None  # Initialize regularization as None by default
    if regularization_type == 'l1':
        reg = keras.regularizers.l1(regularization_strength)
    elif regularization_type == 'l2':
        reg = keras.regularizers.l2(regularization_strength)

    autoencoder = Sequential([
        Dense(m, activation=activation, input_shape=(input_dim,), kernel_regularizer=reg),  # Apply regularization
        Dense(int(m*0.5), activation=activation),
        Dense(int(m*lat), activation=activation),
        Dense(int(m*0.5), activation=activation),
        Dense(m, activation='sigmoid')
    ])

    opt = keras.optimizers.Adam(learning_rate=lr)
    autoencoder.compile(loss=loss, optimizer=opt)
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)

    target = [k for k in hold["poisson_ratio"]]
    mat_id = [k for k in hold["mat_id"]]
    compo = [k for k in hold["composition"]]

    del hold1["poisson_ratio"]
    del hold1["mat_id"]
    del hold1["composition"]

    # Use the trained autoencoder to reconstruct the test data
    hold_sc = scaler.transform(hold1)
    X_reconstructed = autoencoder.predict(hold_sc)
    reconstruction_errors = np.mean(np.square(hold_sc - X_reconstructed), axis=1)
    return reconstruction_errors, target, mat_id
def plot_r(reconstruction_errors,target,threshold,prec):
    fig,ax = plt.subplots()
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = 20
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    plt.scatter(target,reconstruction_errors,s=12)
    plt.ylabel('Reconstruction Error',fontweight='bold',fontsize=22)
    plt.xlabel('Poisson’s Ratio', fontweight='bold', fontsize=22)
    #plt.yticks(np.arange(0,5,0.25))
    #plt.title('Truth Table',fontweight='bold',fontsize=14)
    threshold=threshold-0.001
    plt.axvline(x=0,color='red',lw=3)
    plt.axhline(y=threshold,color='red',lw=3)
    plt.xlim(-1.0,0.5)
    if prec==0.8:
        plt.savefig("truth_table.pdf", format="pdf", bbox_inches="tight")
    #plt.show();
    plt.clf() 
def patt_find(string):
    pattern = r'([A-Za-z]+)(\d*)'
    elements = re.findall(pattern, string)
    symbols = [element[0] for element in elements]
  #  counts = [int(element[1]) if element[1] else 1 for element in elements]
    return symbols
def anomaly_score(reconstruction_errors, target):
    
    # Compute the anomaly score based on the reconstruction errors and target values
    pos_perc=[]
#     for k in zip(reconstruction_errors,target):
#         if k[1]>0:pos_perc.append(k[0])
    target=np.array(target);    reconstruction_errors=np.array(reconstruction_errors)
    thresholds=np.linspace(97.1,99.9,140)
    best_f1=0;cores_prec=0;cores_cut=0;cores_tp=0;cores_thresh=0
    for k in thresholds:
        temp=np.percentile(np.array(reconstruction_errors),k)
        TP = sum((target < 0) & (reconstruction_errors > temp))
        FP= sum((target > 0) & (reconstruction_errors > temp))
        FN=15-TP  # 15 is number of auxetics(anomalies) 
        #print(num_anomalies)
        F1_score=2*TP/(2*TP+FP+FN)
        PREC=TP/(TP+FP)
        if F1_score>best_f1:
            best_f1=F1_score
            cores_prec=PREC
            cores_cut=temp
            cores_tp=TP
            cores_thresh=k

    return best_f1,cores_tp,cores_prec,cores_cut,cores_thresh
if __name__ == '__main__':
    main()