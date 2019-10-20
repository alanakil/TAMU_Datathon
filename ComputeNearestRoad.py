import numpy as np
import pandas as pd
import os
import itertools
from tqdm import tqdm
from glob import glob
import pickle
import gc


def pop_dens_map(map_path, pop_dens, i_unit, j_unit):
    df_map = pd.read_csv(map_path)
    row_ = df_map.iloc[0]

    x_0 = row_.latitude-i_unit*row_.pixel_i
    x_n = x_0 + 8192*i_unit
    y_0 = row_.longitude-j_unit*row_.pixel_j
    y_n = y_0 + 8192*j_unit
    cond = (pop_dens['latitude']>x_0)&(pop_dens['latitude']<x_n)&(pop_dens['longitude']>y_0)&(pop_dens['longitude']<y_n)
    idxs = pop_dens.index[cond].to_list()
    return idxs


def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


def get_dist_info(df_map, pop_dens, pop_idx, i, j, main_dir, threshold):
    cols = ['latitude', 'longitude']
    neighbors = df_map[max(0, i - 1): min(i + 2, 8), max(0, j - 1): min(j + 2, 8)].reshape((-1,))
    csv_paths = [os.path.join(main_dir, "ml_preds_csv", str(item) + '.csv') for item in neighbors]
    csv_files = [pd.read_csv(item) for item in csv_paths]
    csv_files = [item[item['val'] > threshold] for item in csv_files]
    csv_files = [item.values.astype(np.float32) for item in csv_files]
    lens = [len(item) for item in csv_files]
    cum_lens = np.cumsum(lens)
    arr = np.concatenate(csv_files, axis=0)
    pop = pop_dens.iloc[pop_idx]
    pop = pop[cols].values.astype(np.float32)

    dists = np.sum(np.abs(pop[:, None] - arr[:, 3:5]) ** 2, axis=2)
    # print("dists: ", dists.shape)
    min_dists = np.min(dists, axis=1)
    # print(min_dists)
    print(min_dists.shape)
    idxs = np.argmin(dists, axis=1)
    print(idxs.shape)    # print(arr[idxs, 0:2].shape)
    # print(arr[idxs, 0:2])
    pixel_i, pixel_j = arr[idxs, 0], arr[idxs, 1]
    filenames = [None] * len(idxs)
    for k, item in enumerate(idxs):
        # print(item)
        # print(np.where((cum_lens - item) >= 0)[0])
        print(np.where((cum_lens - item) >= 0)[0])
        filenames[k] = neighbors[np.where((cum_lens - item) >= 0)[0][0]-1]
    print("filenames: ", filenames)
    gc.collect()
    return min_dists, pixel_i, pixel_j, filenames


# Find the shortest distance
# main_dir = "/home/mozahid/PycharmProjects/fb/dataset"
# main_dir = "/home/mksafari/servers/storage/fb/dataset"
main_dir = "/home/alan/PycharmProjects/TAMU_Datathon/fb_datathon_dataset"
filenames = glob(os.path.join(main_dir, "ml_preds_csv", '*.csv'))
# pop_dens = pd.read_csv(os.path.join(main_dir, "tz_popdens_sample.csv"))
arr = pd.read_csv("destinations.csv")
units = np.load("units.npy")
df_map = pd.read_csv("df_map.csv")
i_unit, j_unit = units[0, 0], units[0, 1]
df_map = df_map.values

# pop_idxs = [None]*64
# for i, item in tqdm(enumerate(filenames)):
#     pop_idxs[i] = pop_dens_map(item, arr, i_unit, j_unit)
# print(pop_idxs)
with open('dest_pop_idxs.pkl', 'rb') as f:
    pop_idxs = pickle.load(f)
columns = ['distance', 'pixel_i', 'pixel_j', 'patch_loc']
final_arr = np.zeros((len(arr), len(columns)), dtype=np.float32)
for i in tqdm(range(8)):
    for j in tqdm(range(8)):
        loc = sub2ind((8, 8), i, j)
        pop_idx = pop_idxs[loc]
        if pop_idx is not None:
            if len(pop_idx) == 0:
                pass
            else:
                print("loc: ", loc)
                dists, pixel_i, pixel_j , patch_loc = get_dist_info(df_map, arr, pop_idx, i, j, main_dir,
                                                                    threshold=75)
                final_arr[pop_idx] = np.array([dists, pixel_i, pixel_j , patch_loc]).T
                # np.save("final_arr_%d_%d.npy" % (i, j), final_arr)


final_df = pd.DataFrame(final_arr, index=np.arange(len(arr)), columns=columns)
final_df.to_csv("./destination_dists.csv")


