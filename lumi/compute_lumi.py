#!/usr/bin/python
import argparse
import os
import pickle as pkl

import numpy as np
from coffea.lumi_tools import LumiData, LumiList

"""
This script computes the total luminosity using
    (1) a single lumi_set.pkl file saved using combine_lumi.py.
    (2) a lumi.csv produced using the GoldenJson with the commands below.

    # noqa for 2016: brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /pb --byls --output-style csv -i ../boostedhiggs/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt > lumi2016.csv
    # noqa for 2017: brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /pb --byls --output-style csv -i ../boostedhiggs/data/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt > lumi2017.csv
    # noqa for 2018: brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /pb --byls --output-style csv -i ../boostedhiggs/data/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt > lumi2018.csv
"""


def main(args):

    dir_ = f"../eos/Lumi_July11_{args.year}/"
    datasets = os.listdir(dir_)

    for dataset in datasets:
        pklfile = dataset + f"{dataset}/lumi_set_{dataset}.pkl"

        with open(pklfile, "rb") as f:
            lumi_set = pkl.load(f)

        dataset = list(lumi_set.keys())[0]

        # convert the set to a numpy 2d-array
        lumis = np.array(lumi_set[dataset])

        # (run number, lumi block, event id)
        # make LumiList object
        lumi_list = LumiList(runs=lumis[:, 0], lumis=lumis[:, 1])

        # this csv was made using brilcalc and the GoldenJson2017... refer to
        # https://github.com/CoffeaTeam/coffea/blob/52e102fce21a3e19f8c079adc649dfdd27c92075/coffea/lumi_tools/lumi_tools.py#L20
        lumidata = LumiData(f"lumi{args.year}.csv")
        print(f"{dataset}: {lumidata.get_lumi(lumi_list)}")

        print("------------------------------------")


if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python compute_lumi.py

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)

    args = parser.parse_args()
    main(args)
