#!/usr/bin/python
import argparse
import glob
import os
import pickle

import numpy as np
from coffea.lumi_tools import LumiData, LumiList
from tqdm import tqdm

# import pickle as pkl


"""
This script combines the pkl files produced by the lumi processor to a single pkl file
that holds a dictionary with "key=dataset_name" and "value=lumi_set" which has the form (run, lumi).
"""


def main(args):

    # this csv was made using brilcalc and the GoldenJson2017... refer to
    # https://github.com/CoffeaTeam/coffea/blob/52e102fce21a3e19f8c079adc649dfdd27c92075/coffea/lumi_tools/lumi_tools.py#L20
    lumidata = LumiData(f"lumi{args.year}.csv")

    dir_ = f"../eos/Lumi_July11_{args.year}/"
    datasets = os.listdir(dir_)

    for j, dataset in enumerate(datasets):
        if (args.j is not None) & (j != args.j):
            continue

        print(f"Processing dataset: {dataset}")

        pkl_files = glob.glob(dir_ + dataset + "/outfiles/*")

        combined_lumi_set = set()
        for i, pkl_file in tqdm(enumerate(pkl_files), total=len(pkl_files)):
            with open(pkl_file, "rb") as f:
                out = pickle.load(f)

            lumiset = out[dataset][args.year]["lumilist"]
            combined_lumi_set.update(lumiset)

            # break

        # output_file = os.path.join(dir_, f"{dataset}/lumi_set_{dataset}.pkl")
        # with open(output_file, "wb") as handle:
        #     pickle.dump({dataset: combined_lumi_set}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # convert the set to a numpy 2d-array

        print("Will form a numpy array")
        lumis = np.array(list(combined_lumi_set))
        # print(lumis.shape)

        # make LumiList object
        print("Will instantiate a LumiList object")
        lumi_list = LumiList(runs=lumis[:, 0], lumis=lumis[:, 1])

        print("Will get the Lumi")
        lumi = lumidata.get_lumi(lumi_list)

        print(f"{dataset}: {lumi}")

        print("-------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument("--j", dest="j", default=None, help="year", type=int)
    args = parser.parse_args()
    main(args)
