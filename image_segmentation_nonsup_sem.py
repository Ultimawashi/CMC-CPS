import numpy as np
import json
import os
import cv2 as cv
from tools import bruit_gauss, calc_erreur, peano_transform_img, transform_peano_in_img, line_transform_img, \
    transform_line_in_img, peano_to_neighbours, sigmoid_np
from markov_chain_neigh import *
from markov_chain import *
from sklearn.cluster import KMeans

m1 = 0
m2 = 1
sig1 = 1
sig2 = 1

w = np.array([0, 255])
max_val = 255
image = ['./images/beee2.bmp', './images/cible2.bmp', './images/promenade2.bmp', './images/zebre2.bmp']

iter = 100
results = []
params = []
resfolder = './results_non_sup3'
if not os.path.exists(resfolder):
    os.makedirs(resfolder)


for path in image:
    img = cv.imread(path, 0)
    img_name = (path.split('/')[-1]).split('.')[0]
    signal = peano_transform_img(img)

    # bruite image
    signal_noisy = bruit_gauss(img, w, m1, sig1, m2, sig2)
    cv.imwrite(os.path.join(resfolder, img_name + '_noisy.bmp'), sigmoid_np(signal_noisy) * max_val)

    # récupère les voisins
    neighboursh, neighboursv = peano_to_neighbours(signal_noisy)

    # parcours de peano sur image
    peano_noisy = peano_transform_img(signal_noisy)

    # kmeans pour estimer les paramètres à priori
    Y = peano_noisy.reshape(-1, 1)

    kmeans = KMeans(n_clusters=w.shape[0], max_iter=100, n_init=100).fit(Y)
    hidden = kmeans.labels_
    labels_name = np.unique(hidden)
    p_init, A_init = calc_probaprio_mc(hidden, labels_name)

    m1_init = peano_noisy[hidden==labels_name[0]].sum()/hidden.shape[0]
    sig1_init = np.sqrt(((peano_noisy[hidden==labels_name[0]] - m1_init)**2).sum()/hidden.shape[0])

    m2_init = peano_noisy[hidden==labels_name[1]].sum()/hidden.shape[0]
    sig2_init = np.sqrt(((peano_noisy[hidden==labels_name[1]] - m2_init)**2).sum()/hidden.shape[0])

    p_est1, A_est1, m1_est1, sig1_est1, m2_est1, sig2_est1 = estim_param_SEM_mc(iter, peano_noisy, p_init, A_init,
                                                                               m1_init,
                                                                               sig1_init, m2_init, sig2_init)

    p_est2, A_est2, m1_est2, sig1_est2, m2_est2, sig2_est2 = estim_param_SEM_mc_neigh(iter, peano_noisy, neighboursh,
                                                                                     neighboursv, p_init, A_init,
                                                                                     m1_init, sig1_init, m2_init,
                                                                                     sig2_init)
    params.append({'param_mc':{'p': p_est1.tolist(), 't': A_est1.tolist(), 'mu1': m1_est1.tolist(),
               'sig1': sig1_est1.tolist(), 'mu2': m2_est1.tolist(),
               'sig2': sig2_est1.tolist()},'param_mc_neigh':{'p': p_est2.tolist(), 't': A_est2.tolist(), 'mu1': m1_est2.tolist(),
                'sig1': sig1_est2.tolist(), 'mu2': m2_est2.tolist(),
                'sig2': sig2_est2.tolist()}})

    segmentation_peano_mc = mpm_mc(peano_noisy, w, p_est1, A_est1, m1_est1, sig1_est1, m2_est1,
                                   sig2_est1)
    cv.imwrite(os.path.join(resfolder, img_name + '_segmentation_peano_mc.bmp'), transform_peano_in_img(segmentation_peano_mc,256)*max_val)

    segmentation_peano_mc_neigh = mpm_mc_neigh(peano_noisy, neighboursh, neighboursv, w, p_est2, A_est2, m1_est2,
                                               sig1_est2, m2_est2,
                                               sig2_est2)
    cv.imwrite(os.path.join(resfolder, img_name + '_segmentation_peano_mc_neigh.bmp'), transform_peano_in_img(segmentation_peano_mc_neigh,256)*max_val)
    segmentation_peano_mc_neigh_est_mc = mpm_mc_neigh(peano_noisy, neighboursh, neighboursv, w, p_est1, A_est1, m1_est1, sig1_est1, m2_est1,
                                   sig2_est1)
    cv.imwrite(os.path.join(resfolder, img_name + '_segmentation_peano_mc_neigh_est_mc.bmp'), transform_peano_in_img(segmentation_peano_mc_neigh_est_mc,256)*max_val)

    results.append({'img': path, 'err_mc': calc_erreur(signal, segmentation_peano_mc),
           'err_mc_neigh': calc_erreur(signal, segmentation_peano_mc_neigh), 'err_mc_neigh_est_mc':calc_erreur(signal, segmentation_peano_mc_neigh_est_mc)})

with open(os.path.join(resfolder, 'results_nonsup.txt'), 'w') as f:
    json.dump(results, f, ensure_ascii=False)

with open(os.path.join(resfolder, 'params_nonsup.txt'), 'w') as f:
    json.dump(params, f, ensure_ascii=False)
