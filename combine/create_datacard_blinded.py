"""
Creates "combine datacards" using hist.Hist templates.

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
"""

from __future__ import division, print_function

import argparse
import logging
import math
import os
import warnings

import pandas as pd
import rhalphalib as rl
from combine.datacard_systematics import systs_from_parquets, systs_not_from_parquets
from utils import (
    blindBins_diffbins,
    get_template_diffbins,
    labels,
    load_templates,
    samples,
    shape_to_num,
    sigs,
)

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


def create_datacard(hists_templates, years, lep_channels, add_ttbar_constraint=True, add_wjets_constraint=True):
    # define the systematics
    systs_dict, systs_dict_values = systs_not_from_parquets(years, lep_channels)
    sys_from_parquets = systs_from_parquets(years)

    # define the model
    model = rl.Model("testModel")

    # define the signal and control regions
    SIG_regions = ["VBF97", "ggF975pt250to300", "ggF975pt300to450", "ggF975pt450toInf"]
    CONTROL_regions = ["TopCR", "WJetsCR"]

    if add_ttbar_constraint:
        ttbarnormSF = rl.IndependentParameter("ttbarnormSF", 1.0, 0, 10)

    if add_wjets_constraint:
        wjetsnormSF = rl.IndependentParameter("wjetsnormSF", 1.0, 0, 10)

    # fill datacard with systematics and rates
    for ChName in SIG_regions + CONTROL_regions:

        if ChName in SIG_regions:
            h = blindBins_diffbins(hists_templates[ChName].copy())
        else:
            h = hists_templates[ChName].copy()
        # h = blindBins_diffbins(hists_templates[ChName].copy())
        # h = hists_templates[ChName].copy()

        Samples = samples.copy()

        ch = rl.Channel(ChName)
        model.addChannel(ch)

        for sName in Samples:

            if "CR" in ChName:  # remove signal from control regions
                if sName in sigs:
                    continue

            templ = get_template_diffbins(h, sName)
            if templ == 0:
                continue
            stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + labels[sName], stype, templ)

            if "CR" in ChName:
                sample.autoMCStats(lnN=True)

            # SYSTEMATICS NOT FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in systs_dict[syst_on_sample].items():
                    if systs_dict_values[syst_on_sample][sys_name][1] is None:  # if up and down are the same
                        sample.setParamEffect(sys_value, systs_dict_values[syst_on_sample][sys_name][0])
                    else:
                        sample.setParamEffect(
                            sys_value,
                            systs_dict_values[syst_on_sample][sys_name][0],
                            systs_dict_values[syst_on_sample][sys_name][1],
                        )

            # SYSTEMATICS FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in sys_from_parquets[syst_on_sample].items():
                    syst_up = h[{"Sample": sName, "Systematic": sys_name + "_up"}].values()
                    syst_do = h[{"Sample": sName, "Systematic": sys_name + "_down"}].values()
                    nominal = h[{"Sample": sName, "Systematic": "nominal"}].values()

                    if sys_value.combinePrior == "lnN":
                        eff_up = shape_to_num(syst_up, nominal)
                        eff_do = shape_to_num(syst_do, nominal)

                        if math.isclose(eff_up, eff_do, rel_tol=1e-2):  # if up and down are the same
                            sample.setParamEffect(sys_value, max(eff_up, eff_do))
                        else:
                            sample.setParamEffect(sys_value, max(eff_up, eff_do), min(eff_up, eff_do))

                    else:
                        nominal[nominal == 0] = 1  # to avoid invalid value encountered in true_divide in "syst_up/nominal"
                        sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))

            ch.addSample(sample)

        # add data
        data_obs = get_template_diffbins(h, "Data")

        ch.setObservation(data_obs)

        if "CR" not in ChName:
            ch.autoMCStats()

    if add_ttbar_constraint:
        failCh = model["TopCR"]

        ttbarfail = failCh["ttbar"]
        ttbarfail.setParamEffect(ttbarnormSF, 1 * ttbarnormSF)

        for sig_region in SIG_regions:

            passCh = model[sig_region]

            ttbarpass = passCh["ttbar"]
            ttbarpass.setParamEffect(ttbarnormSF, 1 * ttbarnormSF)

    if add_wjets_constraint:
        failCh = model["WJetsCR"]

        wjetsfail = failCh["wjets"]
        wjetsfail.setParamEffect(wjetsnormSF, 1 * wjetsnormSF)

        for sig_region in SIG_regions:
            passCh = model[sig_region]

            wjetspass = passCh["wjets"]
            wjetspass.setParamEffect(wjetsnormSF, 1 * wjetsnormSF)

    return model


def main(args):
    years = args.years.split(",")
    lep_channels = args.channels.split(",")

    hists_templates = load_templates(years, lep_channels, args.outdir)

    model = create_datacard(hists_templates, years, lep_channels)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python create_datacard_diffbins.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v7

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", default="templates/test", type=str, help="name of template directory")

    args = parser.parse_args()

    main(args)
