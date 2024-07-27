"""
Builds hist.Hist templates after adding systematics for all samples.

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import logging
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import utils
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


def get_common_sample_name(sample):
    # first: check if the sample is in one of combine_samples_by_name
    sample_to_use = None
    for key in utils.combine_samples_by_name:
        if key in sample:
            sample_to_use = utils.combine_samples_by_name[key]
            break

    # second: if not, combine under common label
    if sample_to_use is None:
        for key in utils.combine_samples:
            if key in sample:
                sample_to_use = utils.combine_samples[key]
                break
            else:
                sample_to_use = sample
    return sample_to_use


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path, add_fake=False):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess (e.g. ["2016APV", "2016"])
        ch [list]: channels to postprocess (e.g. ["ele", "mu"])
        samples [list]: samples to postprocess (e.g. ["ggF", "QCD", "Data"])
        samples_dir [dict]: points to the path of the parquets for each region
        regions_sel [dict]: key is the name of the region; value is the selection (e.g. `{"pass": (fj_ParT_score>0.97)}`)
        model_path [str]: path to the ParT finetuned model.onnx

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (samples, systematic, ptbin, mass_observable)

    """

    # add extra selections to preselection
    presel = {
        "mu": {
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
        },
        "ele": {
            "fj_mass": "fj_mass>40",
            "tagger>0.75": "THWW>0.75",
        },
    }

    mass_binning = 20

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable(
            list(range(55, 255, mass_binning)),
            name="mass_observable",
            label=r"Higgs reconstructed mass [GeV]",
            overflow=True,
        ),
        storage=hist2.storage.Weight(),
    )

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):

                sample_to_use = get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                if sample_to_use == "Data":
                    is_data = True
                else:
                    is_data = False

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # use hidNeurons to get the finetuned scores
                data["THWW"] = utils.get_finetuned_score(data, model_path)

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                # get the xsecweight
                xsecweight, sumgenweights, sumpdfweights, sumscaleweights = utils.get_xsecweight(
                    pkl_files, year, sample, sample_to_use, is_data, luminosity
                )

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                    df = data.copy()

                    logging.info(f"Applying {region} selection on {len(data)} events")
                    df = df.query(region_sel)
                    logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                    # ------------------- Nominal -------------------
                    if is_data:
                        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                    else:
                        nominal = df[f"weight_{ch}"] * xsecweight

                    ###################################
                    if sample_to_use == "EWKvjets":
                        threshold = 20
                        avg_good_weight = nominal[nominal < threshold].mean()
                        nominal[nominal > threshold] = avg_good_weight
                    ###################################

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="nominal",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=nominal,
                    )

    if add_fake:

        for variation in ["FR_Nominal", "FR_stat_Up", "FR_stat_Down", "EWK_SF_Up", "EWK_SF_Down"]:

            for year in years:

                data = pd.read_parquet(f"{samples_dir[year]}/fake_{year}_ele_{variation}.parquet")

                # apply selection
                for selection in presel["ele"]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel["ele"][selection])

                data["event_weight"] *= 0.6  # the closure test SF

                for region in hists.axes["Region"]:
                    df = data.copy()

                    logging.info(f"Applying {region} selection on {len(df)} events")
                    df = df.query(regions_sel[region])
                    logging.info(f"Will fill the histograms with the remaining {len(df)} events")

                    if variation == "FR_Nominal":
                        hists.fill(
                            Sample="Fake",
                            Systematic="nominal",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=df["event_weight"],
                        )
                    else:
                        hists.fill(
                            Sample="Fake",
                            Systematic=variation,
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=df["event_weight"],
                        )

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 1e-3
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 1e-3


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates_sig.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years,
        channels,
        config["samples"],
        config["samples_dir"],
        config["regions_sel"],
        config["model_path"],
        args.add_fake,
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates_sig.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1_sig --add-fake

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")
    parser.add_argument("--add-fake", dest="add_fake", action="store_true")

    args = parser.parse_args()

    main(args)
