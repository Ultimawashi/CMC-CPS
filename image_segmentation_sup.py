import numpy as np
import json
import os
import cv2 as cv
from tools import bruit_gauss,calc_erreur, peano_transform_img, transform_peano_in_img, line_transform_img, transform_line_in_img, peano_to_neighbours, sigmoid_np, heaviside_np
from markov_chain_neigh import *
from markov_chain import *
from markov_field import *


m1 = 1
m2 = 2
sig1 = 1
sig2 = 1

w = np.array([0, 1])
max_val = 255
image = ['./images/beee2.bmp', './images/cible2.bmp', './images/promenade2.bmp', './images/zebre2.bmp']
iter_gibbs = 100
nb_simu = 10
results = []
params = []
resfolder = './results_sup'
resolution = (256,256)
if not os.path.exists(resfolder):
    os.makedirs(resfolder)

for path in image:

    img = cv.imread(path, 0)
    img_name = (path.split('/')[-1]).split('.')[0]
    img = cv.resize(img, resolution)
    img = heaviside_np(img)
    signal = peano_transform_img(img)

    # bruite image
    signal_noisy = bruit_gauss(img, w, m1, sig1, m2, sig2)
    cv.imwrite(os.path.join(resfolder, img_name + '_noisy.bmp'), sigmoid_np(signal_noisy) * max_val)

    # récupère les voisins
    neighboursh, neighboursv = peano_to_neighbours(signal_noisy)

    # parcours de peano sur image
    peano_noisy = peano_transform_img(signal_noisy)

    # kmeans pour estimer les paramètres à priori
    m1_est = m1
    sig1_est = sig1

    m2_est = m2
    sig2_est = sig2

    p_est, A_est = calc_probaprio_mc(signal, w)

    proba_est, _, _, _, _ = estim_param_pseudol_sup(img, signal_noisy)

    params.append({'param_all': {'proba': proba_est.tolist(),'p': p_est.tolist(), 't': A_est.tolist(), 'mu1': m1_est,
                                'sig1': sig1_est, 'mu2': m2_est,
                                'sig2': sig2_est}})

    segmentation_peano_mc = mpm_mc(peano_noisy, w, p_est, A_est, m1_est, sig1_est, m2_est,
                                   sig2_est)
    cv.imwrite(os.path.join(resfolder, img_name + '_segmentation_peano_mc.bmp'),
               transform_peano_in_img(segmentation_peano_mc, resolution[0]) * max_val)

    segmentation_peano_mc_neigh = mpm_mc_neigh(peano_noisy, neighboursh, neighboursv, w, p_est, A_est, m1_est, sig1_est, m2_est,
                                   sig2_est)
    cv.imwrite(os.path.join(resfolder, img_name + '_segmentation_peano_mc_neigh.bmp'),
               transform_peano_in_img(segmentation_peano_mc_neigh, resolution[0]) * max_val)

    segmentation_mf = mpm_mf(signal_noisy, proba_est, m1_est, sig1_est, m2_est, sig2_est, iter_gibbs, nb_simu)
    cv.imwrite(os.path.join(resfolder, img_name + '_segmentation_mf.bmp'),
               segmentation_mf.astype(np.uint8)*max_val)

    results.append({'img': path, 'err_mc': calc_erreur(signal, segmentation_peano_mc),
                    'err_mc_neigh': calc_erreur(signal, segmentation_peano_mc_neigh), 'err_mf':calc_erreur(img, segmentation_mf)})

with open(os.path.join(resfolder, 'results_sup.txt'), 'w') as f:
    json.dump(results, f, ensure_ascii=False)

with open(os.path.join(resfolder, 'params_sup.txt'), 'w') as f:
    json.dump(params, f, ensure_ascii=False)

