"""
HE2RNA: Predict MSI status using either transfer learning from transcriptome prediction, or WSIs directly
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Input
import keras.backend as K
from keras.initializers import RandomNormal
from constant import PATH_TO_TILES, PATH_TO_TRANSCRIPTOME, PATH_TO_MSI_LABELS


def charging_data_msi(cancer_types, type_slide='FFPE', msi_l=0):

    X_mean = np.zeros((1000, 2048))

    Y = []
    ids = []
    dict_ids = dict()
    dict_X = dict()
    dict_Y = dict()
    print('List of slides excluded because they have with less than 100 tiles: \n')
    for j in cancer_types:
        print('Charging the labels for ', j)

        if type_slide == 'FFPE':
            slide_directory = f'{PATH_TO_TILES}/TCGA_{j}/0.50_mpp'
            msi_df = pd.read_csv(f'{PATH_TO_MSI_LABELS}/msi_{j}.csv')
        elif type_slide == 'Frozen':
            slide_directory = f'{PATH_TO_TILES}/TCGA_CRC_frozen/0.50_mpp'
            msi_df = pd.read_csv(f'{PATH_TO_MSI_LABELS}/msi_{j}_KR.csv')

        for _, row in msi_df.iterrows():

            for filename in os.listdir(slide_directory):

                if filename[:12] in row['0'] and 'mask' not in filename:
                    low_pass = 0
                    val = np.load(os.path.join(slide_directory, filename))

                    #Skip if the slide contains less than 100 tiles
                    if val.shape[0] < 100:
                        print(filename)
                        continue

                    if row['1'] == 'MSI-H':
                        Y = 1
                        low_pass = 1
                    elif row['1'] == 'MSS':
                        Y = 0
                        low_pass = 1
                    elif row['1'] == 'MSI-L' and msi_l == 1:
                        Y = 0
                        low_pass = 1

                    if filename[:12] not in dict_ids and low_pass == 1:
                        dict_ids[filename[:12]] = 1
                        dict_X[filename[:12]] = np.reshape(
                            np.mean(val[:, 3:], axis=0),
                            (1, -1))
                        dict_Y[filename[:12]] = np.array([Y])

                    elif filename[:12] in dict_ids and low_pass == 1:

                        dict_X[filename[:12]] = np.concatenate((
                            dict_X[filename[:12]],
                            np.reshape(np.mean(val[:, 3:], axis=0), (1, -1))), axis=0)
                        dict_Y[filename[:12]] = np.concatenate((
                            dict_Y[filename[:12]],
                            np.array([Y])), axis=0)

    return dict_X,dict_Y



def get_model_msi_classification_128(n):

    input_tile = Input(shape=(n,))

    x = Dense(128, activation='sigmoid')(input_tile)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tile, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=["accuracy"])

    return model

def get_model_msi_classification_256(n):

    input_tile = Input(shape=(n,))

    x = Dense(256, activation='sigmoid')(input_tile)
    x = Dense(128, activation='sigmoid')(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_tile, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=["accuracy"])

    return model


def get_model_gene_prediction_sig(Ngenes, last_bias_mean, last_bias_std):

    input_tile = Input(shape=(2048,))

    x = Dense(1024, activation='sigmoid')(input_tile)
    x = Dense(256, activation='sigmoid')(x)

    x_ts = Dense(
        Ngenes, activation='linear',
        bias_initializer=RandomNormal(
            mean=last_bias_mean, stddev=last_bias_std, seed=None)
        )(x)

    model = Model(inputs=input_tile, outputs=x_ts)
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=["accuracy"])

    return model


def autoenc_256_2h():

    input_tile = Input(shape=(2048,))

    x = Dense(1024, activation='relu')(input_tile)

    x = Dense(256, activation='linear')(x)

    x = Dense(1024, activation='relu')(x)

    x_ts = Dense(2048, activation='linear')(x)

    model = Model(inputs=input_tile, outputs=x_ts)
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=["accuracy"])

    return model


def auc_all(dict_X,dict_Y,cancer_types,type_slide,msi_l=0,Ncval=500,n_internsplit=3,n_epoch=50):

    pat_ids_unique = np.ravel(np.array(list(dict_Y.keys())))

    lll = '-'
    for u in cancer_types:
        lll = lll + u + '-'
    filename_slides = 'auc_all_' + type_slide + '_' + lll + 'msi_l_' + str(msi_l) + '_slides.npy'

    filename_patients = 'auc_all_' + type_slide + '_' + lll + 'msi_l_' + str(msi_l) + '_patients.npy'

    k = 0
    for p in pat_ids_unique:
        if k == 0:
            k += 1
            X = dict_X[p]

            count = dict_X[p].shape[0]
            count_slides = np.repeat(np.array([count]), count)

            y_val = np.array([dict_Y[p][0]])
            y = np.repeat(np.array([y_val]), count)

            pat_ids = np.repeat(p, count)

        else:

            X_val = dict_X[p]
            X = np.concatenate((X, dict_X[p]), axis=0)

            count = dict_X[p].shape[0]
            count_slides = np.concatenate((count_slides, np.repeat(np.array([count]), count)))

            y_val = np.array([dict_Y[p][0]])
            y = np.concatenate((y, np.repeat(np.array([y_val]), count)))

            pat_ids = np.concatenate((pat_ids, np.repeat(p, count)))

    TOT_pat = pat_ids_unique.shape[0]

    X = scale(X)
    X_df = pd.DataFrame(data=X, index=pat_ids)
    y_df = pd.DataFrame(data=y, index=pat_ids)
    count_slides_df = pd.DataFrame(data=count_slides, index=pat_ids)

    auc_WSI_slides = []
    auc_WSI_patients = []

    for cval in range(Ncval):
        print('CV n.', cval)
        skf = StratifiedKFold(n_splits=n_internsplit, shuffle=True, random_state=42 * cval)
        y_df_B_unique = y_df.reset_index().drop_duplicates(subset='index').set_index('index').values

        for train_index, test_index in skf.split(np.zeros((y_df_B_unique.shape[0], 1)), y_df_B_unique):

            pat_B_train = pat_ids_unique[train_index]
            pat_B_test = pat_ids_unique[test_index]

            y_train = y_df.loc[pat_B_train].values
            y_train_patients = y_df.loc[pat_B_train].reset_index().drop_duplicates(
                subset='index').set_index('index').values

            X_train_resnet = X_df.loc[pat_B_train].values

            y_test = y_df.loc[pat_B_test].values
            y_test_patients = y_df.loc[pat_B_test].reset_index().drop_duplicates(
                subset='index').set_index('index').values

            X_test_resnet = X_df.loc[pat_B_test].values

            pat_B_test_ids = X_df.loc[pat_B_test].index.values


            # All  Images
            Nfeatures_resnet = X_train_resnet.shape[1]

            K.clear_session()
            model = get_model_msi_classification_256(Nfeatures_resnet)

            model.fit(X_train_resnet, y_train, epochs=n_epoch, batch_size=10, verbose=0)
            pred_test_slides = np.squeeze(model.predict(X_test_resnet))
            pred_test_slides_df = pd.DataFrame(data=pred_test_slides, index=pat_B_test_ids)
            pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                'index').mean().loc[pat_B_test].values)

            auc_WSI_slides.append(roc_auc_score(y_test, pred_test_slides))
            auc_WSI_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

    np.save(filename_slides, auc_WSI_slides)
    np.save(filename_patients, auc_WSI_patients)

    print()
    print()
    print('On all dataset, MSI-H is predicted (%d CV %d-fold): ' % (Ncval, n_internsplit))
    print('- at level of slides with AUC: %.3f +/- %.3f' % (np.mean(auc_WSI_slides),
                                                            np.std(auc_WSI_slides)))
    print('- at level of patients with AUC: %.3f +/- %.3f' % (np.mean(auc_WSI_patients),
                                                              np.std(auc_WSI_patients)))
    print()
    print('The results for slides of the %d folds are saved in %s' % (Ncval * n_internsplit,
                                                                      filename_slides))
    print()
    print('The results for patients of the %d folds are saved in %s' % (Ncval * n_internsplit,
                                                                        filename_patients))



def auc_thresh(dict_X, dict_Y, cancer_types, type_slide,
               msi_l=0, Nsplit=50, Ncval=10, n_internsplit_A=3,
               n_internsplit_B=3, n_epoch=50):

    pat_ids_unique = np.ravel(np.array(list(dict_Y.keys())))

    lll = '-'
    for u in cancer_types:
        lll = lll + u + '-'
    filename_slides = 'auc_thr_' + type_slide + '_' + lll + 'msi_l_' + str(msi_l)+'_slides.npy'
    filename_patients = 'auc_thr_' + type_slide + '_' + lll + 'msi_l_' + str(msi_l)+'_patients.npy'

    k = 0
    for p in pat_ids_unique:
        if k == 0:
            k += 1
            X = dict_X[p]

            count = dict_X[p].shape[0]
            count_slides = np.repeat(np.array([count]), count)

            y_val = np.array([dict_Y[p][0]])
            y = np.repeat(np.array([y_val]), count)

            pat_ids = np.repeat(p, count)

        else:

            X_val = dict_X[p]
            X = np.concatenate((X, dict_X[p]), axis=0)

            count = dict_X[p].shape[0]
            count_slides = np.concatenate((count_slides, np.repeat(np.array([count]), count)))

            y_val = np.array([dict_Y[p][0]])
            y = np.concatenate((y, np.repeat(np.array([y_val]), count)))

            pat_ids = np.concatenate((pat_ids, np.repeat(p, count)))

    df_trans = pd.read_csv(f'{PATH_TO_TRANSCRIPTOME}/all_transcriptomes.csv').drop_duplicates(subset='Case.ID')
    df_trans_red = df_trans.set_index('Case.ID').filter(regex='ENS').loc[pat_ids].values
    trans_medlog = np.log10(1 + df_trans_red[:, np.median(df_trans_red, axis=0) > 0])

    TOT_pat = pat_ids_unique.shape[0]
    Ngenes = trans_medlog.shape[1]

    X = scale(X)
    X_df = pd.DataFrame(data=X, index=pat_ids)
    y_df = pd.DataFrame(data=y, index=pat_ids)
    gene_df = pd.DataFrame(data=trans_medlog, index=pat_ids)
    count_slides_df = pd.DataFrame(data=count_slides, index=pat_ids)

    ind_testing = np.zeros((Nsplit, pat_ids_unique.shape[0]))
    for i in range(Nsplit):
        ind_testing[i, :] = np.random.permutation(pat_ids_unique.shape[0])
    ind_testing = ind_testing.astype(int)

    hopA_size = np.array([
        int(0.833 * TOT_pat), int(0.667 * TOT_pat), int(0.5 * TOT_pat),
        int(0.33 * TOT_pat), int(0.167 * TOT_pat)])

    result_thr_slides = np.zeros((hopA_size.shape[0], Nsplit, 2))
    result_thr_patients = np.zeros((hopA_size.shape[0], Nsplit, 2))

    for thr in range(hopA_size.shape[0]):
        print('Starting thr n. ', thr)
        print()
        print()
        hopA_s = hopA_size[thr]
        for rs in range(Nsplit):

            ind = ind_testing[rs, :]
            hopA_index = ind[:hopA_s]
            hopB_index = ind[hopA_s:]

            pat_A = pat_ids_unique[hopA_index]
            pat_B = pat_ids_unique[hopB_index]

            # HospitalA. Access to transcriptomic data and image features (not to the MSI status data)
            X_hospA = X_df.loc[pat_A]
            Y_gene_hospA_val_df = gene_df.loc[pat_A]

            pat_A_ids = Y_gene_hospA_val_df.index.values
            Y_gene_hospA_val = Y_gene_hospA_val_df.values
            Y_gene_hospA_val = (Y_gene_hospA_val - np.mean(Y_gene_hospA_val, axis=0)) / np.std(
                Y_gene_hospA_val, axis=0)

            Y_gene_hospA = pd.DataFrame(data=Y_gene_hospA_val, index=pat_A_ids)

            # HospitalB. Access to MSI status data and image features (not to the transcriptomic data)
            X_hospB_resnet = X_df.loc[pat_B]
            Y_MSI_hospB = y_df.loc[pat_B]
            pat_B_ids = X_hospB_resnet.index.values
            h256_predicted_hosp_B_resnet = 0

            # KFold to train the gene prediction model
            skf = KFold(n_splits=n_internsplit_A, shuffle=True)

            for train_index, test_index in skf.split(np.zeros((hopA_s,1))):

                pat_A_train = pat_A[train_index]

                X_hospA_train = X_hospA.loc[pat_A_train].values
                Y_gene_hospA_train = Y_gene_hospA.loc[pat_A_train].values

                K.clear_session()
                model_gene_prediction = get_model_gene_prediction_sig(Ngenes,
                                                                      last_bias_mean=np.mean(Y_gene_hospA_train,axis=0),
                                                                      last_bias_std=np.std(Y_gene_hospA_train,axis=0))
                model_gene_prediction.fit(X_hospA_train, Y_gene_hospA_train, epochs=50, verbose=0)
                get_hidden_256_layer_output = K.function(
                    [model_gene_prediction.layers[0].input],
                    [model_gene_prediction.layers[-2].output])

                h256_predicted_hosp_B_resnet += get_hidden_256_layer_output([X_hospB_resnet.values])[0] / n_internsplit_A

            h256_predicted_hosp_B_df = pd.DataFrame(data=h256_predicted_hosp_B_resnet, index=pat_B_ids)

            auc_WSI_slides = []
            auc_tl_slides = []

            auc_WSI_patients = []
            auc_tl_patients = []

            for cval in range(Ncval):

                skf = StratifiedKFold(n_splits=n_internsplit_B, shuffle=True, random_state=42*cval)
                y_df_B_unique = Y_MSI_hospB.reset_index().drop_duplicates(subset='index').set_index(
                    'index').values

                for train_index, test_index in skf.split(np.zeros((y_df_B_unique.shape[0], 1)), y_df_B_unique):

                    pat_B_train=pat_B[train_index]
                    pat_B_test = pat_B[test_index]

                    y_train = Y_MSI_hospB.loc[pat_B_train].values
                    y_train_patients = Y_MSI_hospB.loc[pat_B_train].reset_index().drop_duplicates(
                        subset='index').set_index('index').values

                    X_train_resnet = X_hospB_resnet.loc[pat_B_train].values

                    X_train_h256_resnet = h256_predicted_hosp_B_df.loc[pat_B_train].values

                    y_test = Y_MSI_hospB.loc[pat_B_test].values
                    y_test_patients = Y_MSI_hospB.loc[pat_B_test].reset_index().drop_duplicates(
                        subset='index').set_index('index').values

                    X_test_resnet = X_hospB_resnet.loc[pat_B_test].values

                    pat_B_test_ids = X_hospB_resnet.loc[pat_B_test].index.values

                    X_test_h256_resnet = h256_predicted_hosp_B_df.loc[pat_B_test].values

                    # Images
                    Nfeatures_resnet = X_train_resnet.shape[1]

                    K.clear_session()
                    model = get_model_msi_classification_256(Nfeatures_resnet)

                    model.fit(X_train_resnet, y_train, epochs=n_epoch, batch_size=10, verbose=0)
                    pred_test_slides = np.squeeze(model.predict(X_test_resnet))
                    pred_test_slides_df = pd.DataFrame(data=pred_test_slides, index=pat_B_test_ids)
                    pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                        'index').mean().loc[pat_B_test].values)

                    auc_WSI_slides.append(roc_auc_score(y_test, pred_test_slides))
                    auc_WSI_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

                    # With transcriptomic learning
                    K.clear_session()

                    model = get_model_msi_classification_128(256)

                    model.fit(X_train_h256_resnet, y_train, epochs=n_epoch, batch_size=10, verbose=0)
                    pred_test_slides = np.squeeze(model.predict(X_test_h256_resnet))
                    pred_test_slides_df = pd.DataFrame(data=pred_test_slides, index=pat_B_test_ids)
                    pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                        'index').mean().loc[pat_B_test].values)

                    auc_tl_slides.append(roc_auc_score(y_test, pred_test_slides))
                    auc_tl_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

            result_thr_slides[thr, rs, 0] = np.mean(auc_WSI_slides)
            result_thr_slides[thr, rs, 1] = np.mean(auc_tl_slides)

            result_thr_patients[thr, rs, 0] = np.mean(auc_WSI_patients)
            result_thr_patients[thr, rs, 1] = np.mean(auc_tl_patients)

            np.save(filename_slides, result_thr_slides)
            np.save(filename_patients, result_thr_patients)

        print()
        print()
        print('The results for slides of the analysis by threshold are being saved in %s' % filename_slides)
        print()
        print('The results for patients of the analysis by threshold are being saved in %s' % filename_patients)


def auc_75(dict_X, dict_Y, cancer_types, type_slide, msi_l=0,
           Nsplit=50, Ncval=10, n_internsplit_A=3, n_internsplit_B=3, n_epoch=50):

    pat_ids_unique = np.ravel(np.array(list(dict_Y.keys())))

    lll = '-'
    for u in cancer_types:
        lll = lll + u + '-'
    filename_slides = 'auc_75_' + type_slide + '_' + lll + 'msi_l_' + str(msi_l) + '_slides.npy'

    filename_patients = 'auc_75_' + type_slide + '_' + lll + 'msi_l_' + str(msi_l) + '_patients.npy'

    k = 0
    for p in pat_ids_unique:
        if k == 0:
            k += 1
            X = dict_X[p]

            count = dict_X[p].shape[0]
            count_slides = np.repeat(np.array([count]), count)

            y_val = np.array([dict_Y[p][0]])
            y = np.repeat(np.array([y_val]), count)

            pat_ids = np.repeat(p, count)

        else:

            X_val = dict_X[p]
            X = np.concatenate((X, dict_X[p]), axis=0)

            count = dict_X[p].shape[0]
            count_slides = np.concatenate((count_slides, np.repeat(np.array([count]), count)))

            y_val = np.array([dict_Y[p][0]])
            y = np.concatenate((y, np.repeat(np.array([y_val]), count)))

            pat_ids = np.concatenate((pat_ids, np.repeat(p, count)))

    df_trans = pd.read_csv(f'{PATH_TO_TRANSCRIPTOME}/all_transcriptomes.csv').drop_duplicates(subset='Case.ID')
    df_trans_red = df_trans.set_index('Case.ID').filter(regex='ENS').loc[pat_ids].values
    trans_medlog = np.log10(1 + df_trans_red[:, np.median(df_trans_red, axis=0) > 0])
    TOT_pat = pat_ids_unique.shape[0]
    Ngenes = trans_medlog.shape[1]

    X = scale(X)
    X_df = pd.DataFrame(data=X, index=pat_ids)
    y_df = pd.DataFrame(data=y, index=pat_ids)
    gene_df = pd.DataFrame(data=trans_medlog, index=pat_ids)
    count_slides_df = pd.DataFrame(data=count_slides, index=pat_ids)

    ind_testing = np.zeros((Nsplit, pat_ids_unique.shape[0]))
    for i in range(Nsplit):
        ind_testing[i, :] = np.random.permutation(pat_ids_unique.shape[0])
    ind_testing = ind_testing.astype(int)

    hopA_size = np.array([int(0.75 * TOT_pat)])

    result_thr_slides = np.zeros((hopA_size.shape[0], Nsplit, 4))
    result_thr_patients = np.zeros((hopA_size.shape[0], Nsplit, 4))

    for thr in range(hopA_size.shape[0]):
        print('Starting thr n. ', thr)
        print()
        print()
        hopA_s = hopA_size[thr]
        for rs in range(Nsplit):

            ind = ind_testing[rs, :]
            hopA_index = ind[:hopA_s]
            hopB_index = ind[hopA_s:]

            pat_A = pat_ids_unique[hopA_index]
            pat_B = pat_ids_unique[hopB_index]

            # HospitalA. Access to transcriptomic data and image features (not to the MSI status data)
            X_hospA = X_df.loc[pat_A]
            Y_gene_hospA_val_df = gene_df.loc[pat_A]

            pat_A_ids = Y_gene_hospA_val_df.index.values
            Y_gene_hospA_val = Y_gene_hospA_val_df.values
            Y_gene_hospA_val = (Y_gene_hospA_val-np.mean(Y_gene_hospA_val,axis=0))/np.std(Y_gene_hospA_val,axis=0)

            Y_gene_hospA = pd.DataFrame(data=Y_gene_hospA_val,index=pat_A_ids)

            # HospitalB. Access to MSI status data and image features (not to the transcriptomic data)
            X_hospB_resnet = X_df.loc[pat_B]

            Y_MSI_hospB = y_df.loc[pat_B]

            pat_B_ids = X_hospB_resnet.index.values

            # Training the autoencoder from Hosp.B dataset

            K.clear_session()
            model_auto_prediction = autoenc_256_2h()
            model_auto_prediction.fit(X_hospB_resnet.values, X_hospB_resnet.values, epochs=100, verbose=0)
            encoder = Model(model_auto_prediction.input, model_auto_prediction.layers[-3].output)
            auto_predicted_hosp_B_fromB = encoder.predict(X_hospB_resnet.values)
            print(auto_predicted_hosp_B_fromB.shape)
            auto_predicted_hosp_B_fromB_df = pd.DataFrame(data=auto_predicted_hosp_B_fromB, index=pat_B_ids)

            # KFold to train the gene prediction model
            skf = KFold(n_splits=n_internsplit_A, shuffle=True)

            h256_predicted_hosp_B_resnet = 0
            auto_predicted_hosp_B_fromA = 0

            for train_index, test_index in skf.split(np.zeros((hopA_s, 1))):

                pat_A_train = pat_A[train_index]

                X_hospA_train = X_hospA.loc[pat_A_train].values
                Y_gene_hospA_train = Y_gene_hospA.loc[pat_A_train].values

                # Training the autoencoder from Hosp.A dataset
                K.clear_session()
                model_gene_prediction = get_model_gene_prediction_sig(
                    Ngenes,
                    last_bias_mean=np.mean(Y_gene_hospA_train,axis=0),
                    last_bias_std=np.std(Y_gene_hospA_train,axis=0))
                model_gene_prediction.fit(X_hospA_train, Y_gene_hospA_train, epochs=50, verbose=0)
                get_hidden_256_layer_output = K.function(
                    [model_gene_prediction.layers[0].input],
                    [model_gene_prediction.layers[-2].output])

                h256_predicted_hosp_B_resnet += get_hidden_256_layer_output([X_hospB_resnet.values])[0]/n_internsplit_A

                K.clear_session()
                model_auto_prediction = autoenc_256_2h()
                model_auto_prediction.fit(X_hospA_train, X_hospA_train, epochs=100, verbose=0)
                encoder = Model(model_auto_prediction.input, model_auto_prediction.layers[-3].output)
                auto_predicted_hosp_B_fromA += encoder.predict(X_hospB_resnet.values)/n_internsplit_A
                print(auto_predicted_hosp_B_fromA.shape)

            h256_predicted_hosp_B_df = pd.DataFrame(data=h256_predicted_hosp_B_resnet,index=pat_B_ids)
            auto_predicted_hosp_B_fromA_df = pd.DataFrame(data=auto_predicted_hosp_B_fromA, index=pat_B_ids)

            auc_WSI_slides = []
            auc_tl_slides = []
            auc_auto_fromA_slides = []
            auc_auto_fromB_slides = []

            auc_WSI_patients = []
            auc_tl_patients = []
            auc_auto_fromA_patients = []
            auc_auto_fromB_patients = []

            for cval in range(Ncval):

                skf = StratifiedKFold(n_splits=n_internsplit_B, shuffle=True, random_state=42*cval)
                y_df_B_unique = Y_MSI_hospB.reset_index().drop_duplicates(subset='index').set_index(
                    'index').values

                for train_index, test_index in skf.split(np.zeros((y_df_B_unique.shape[0], 1)), y_df_B_unique):

                    pat_B_train = pat_B[train_index]
                    pat_B_test = pat_B[test_index]

                    y_train = Y_MSI_hospB.loc[pat_B_train].values
                    y_train_patients = Y_MSI_hospB.loc[pat_B_train].reset_index().drop_duplicates(
                        subset='index').set_index('index').values

                    X_train_resnet = X_hospB_resnet.loc[pat_B_train].values

                    X_train_h256_resnet = h256_predicted_hosp_B_df.loc[pat_B_train].values

                    X_train_auto_fromA = auto_predicted_hosp_B_fromA_df.loc[pat_B_train].values
                    X_train_auto_fromB = auto_predicted_hosp_B_fromB_df.loc[pat_B_train].values

                    y_test = Y_MSI_hospB.loc[pat_B_test].values
                    y_test_patients = Y_MSI_hospB.loc[pat_B_test].reset_index().drop_duplicates(
                        subset='index').set_index('index').values

                    X_test_resnet = X_hospB_resnet.loc[pat_B_test].values

                    pat_B_test_ids = X_hospB_resnet.loc[pat_B_test].index.values

                    X_test_h256_resnet = h256_predicted_hosp_B_df.loc[pat_B_test].values

                    X_test_auto_fromA = auto_predicted_hosp_B_fromA_df.loc[pat_B_test].values
                    X_test_auto_fromB = auto_predicted_hosp_B_fromB_df.loc[pat_B_test].values

                    # Images
                    Nfeatures_resnet = X_train_resnet.shape[1]

                    K.clear_session()
                    model = get_model_msi_classification_256(Nfeatures_resnet)

                    model.fit(X_train_resnet, y_train, epochs=n_epoch, batch_size=10, verbose=0)
                    pred_test_slides = np.squeeze(model.predict(X_test_resnet))
                    pred_test_slides_df = pd.DataFrame(data=pred_test_slides, index=pat_B_test_ids)
                    pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                        'index').mean().loc[pat_B_test].values)

                    auc_WSI_slides.append(roc_auc_score(y_test, pred_test_slides))
                    auc_WSI_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

                    # With transcriptomic learning
                    K.clear_session()

                    model = get_model_msi_classification_128(256)

                    model.fit(X_train_h256_resnet, y_train, epochs=n_epoch, batch_size=10, verbose=0)
                    pred_test_slides = np.squeeze(model.predict(X_test_h256_resnet))
                    pred_test_slides_df = pd.DataFrame(data=pred_test_slides,index=pat_B_test_ids)
                    pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                        'index').mean().loc[pat_B_test].values)

                    auc_tl_slides.append(roc_auc_score(y_test, pred_test_slides))
                    auc_tl_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

                    # Autoencoder from A
                    K.clear_session()

                    model = get_model_msi_classification_128(256)

                    model.fit(X_train_auto_fromA, y_train, epochs=n_epoch, batch_size=10, verbose=0)
                    pred_test_slides = np.squeeze(model.predict(X_test_auto_fromA))
                    pred_test_slides_df = pd.DataFrame(data=pred_test_slides, index=pat_B_test_ids)
                    pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                        'index').mean().loc[pat_B_test].values)

                    auc_auto_fromA_slides.append(roc_auc_score(y_test, pred_test_slides))
                    auc_auto_fromA_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

                    # Autoencoder from B
                    K.clear_session()

                    model = get_model_msi_classification_128(256)

                    model.fit(X_train_auto_fromB, y_train, epochs=n_epoch, batch_size=10, verbose=0)
                    pred_test_slides = np.squeeze(model.predict(X_test_auto_fromB))
                    pred_test_slides_df = pd.DataFrame(data=pred_test_slides, index=pat_B_test_ids)
                    pred_test_patients = np.squeeze(pred_test_slides_df.reset_index().groupby(
                        'index').mean().loc[pat_B_test].values)

                    auc_auto_fromB_slides.append(roc_auc_score(y_test, pred_test_slides))
                    auc_auto_fromB_patients.append(roc_auc_score(y_test_patients, pred_test_patients))

            result_thr_slides[thr, rs, 0] = np.mean(auc_WSI_slides)
            result_thr_slides[thr, rs, 1] = np.mean(auc_tl_slides)
            result_thr_slides[thr, rs, 2] = np.mean(auc_auto_fromA_slides)
            result_thr_slides[thr, rs, 3] = np.mean(auc_auto_fromB_slides)

            result_thr_patients[thr, rs, 0] = np.mean(auc_WSI_patients)
            result_thr_patients[thr, rs, 1] = np.mean(auc_tl_patients)
            result_thr_patients[thr, rs, 2] = np.mean(auc_auto_fromA_patients)
            result_thr_patients[thr, rs, 3] = np.mean(auc_auto_fromB_patients)

            np.save(filename_slides, result_thr_slides)
            np.save(filename_patients, result_thr_patients)
        print()
        print()
        print('The results for slides of the analysis on threshold 0.75 are being saved in %s' % filename_slides)
        print()
        print('The results for patients of the analysis on threshold 0.75 are being saved in %s' % filename_patients)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_types", help="list of cancer type, ex: 'COAD READ' for CRC case", nargs='+', type=str,
                        default=['COAD', 'READ'])
    parser.add_argument("--type_slide", help="FFPE or Frozen", type=str,
                        default='FFPE')
    parser.add_argument(
        "--msi_l", help="take (1) or not (0) MSI_low as MSS. ", type=int,
        default=0)
    parser.add_argument(
        "--Nsplit", help="number of different splittings between Hospital A and B", type=int,
        default=50)
    parser.add_argument(
        "--Ncval_all", help="number of different cross-validations for whole WSI dataset analysis", type=int,
        default=10)
    parser.add_argument(
        "--Ncval", help="number of different cross-validations per split for MSI prediction on hospital B", type=int,
        default=10)
    parser.add_argument(
        "--n_internsplit_A", help="number of folds for the cross_validation in Hospital A", type=int,
        default=3)
    parser.add_argument(
        "--n_internsplit_B", help="number of folds for the cross_validation in Hospital B", type=int,
        default=3)
    parser.add_argument(
        "--n_epoch", help="number of epochs for training all neural networks", type=int,
        default=50)
    args = parser.parse_args()

    return (args.cancer_types, args.type_slide, args.msi_l, args.Nsplit, args.Ncval_all,
            args.Ncval, args.n_internsplit_A, args.n_internsplit_B, args.n_epoch)


if __name__ == '__main__':

    cancer_types, type_slide, msi_l, Nsplit, Ncval_all, Ncval, n_internsplit_A, n_internsplit_B, n_epoch = main()
    dict_X, dict_Y = charging_data_msi(cancer_types, type_slide, msi_l)
    auc_thresh(dict_X, dict_Y, cancer_types, type_slide, msi_l, Nsplit, Ncval, n_internsplit_A, n_internsplit_B, n_epoch)
    auc_75(dict_X, dict_Y, cancer_types, type_slide, msi_l, Nsplit, Ncval, n_internsplit_A, n_internsplit_B, n_epoch)
    auc_all(dict_X, dict_Y, cancer_types, type_slide, msi_l, Ncval_all, n_internsplit_B, n_epoch)
