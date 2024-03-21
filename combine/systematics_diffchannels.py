from __future__ import division, print_function

import json
import logging
import warnings
from typing import List

import pandas as pd
import rhalphalib as rl
from utils import samples, sigs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


def systs_not_from_parquets(years: List[str], lep_channels: List[str]):
    """
    Define systematics that are NOT stored in the parquets.

    Args
        years: e.g. ["2018", "2017", "2016APV", "2016"]
        lep_channels: e.g. ["mu", "ele"]

    Returns
        systs_dict [dict]:        keys are systematics; values are rl.NuisanceParameters
        systs_dict_values [dict]: keys are same as above; values are tuples (up, down) and if (up, None) then down=up
    """
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in lep_channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(lep_channels)

    # get the LUMI covered in the templates
    full_lumi = 0
    for year in years:
        full_lumi += LUMI[year]

    systs_dict, systs_dict_values = {}, {}

    # COMMON SYSTEMATICS
    systs_dict["all_samples"], systs_dict_values["all_samples"] = {}, {}

    # branching ratio systematics
    systs_dict["all_samples"]["BR_hww"] = rl.NuisanceParameter("BR_hww", "lnN")
    systs_dict_values["all_samples"]["BR_hww"] = (1.0153, 0.9848)

    systs_dict["all_samples"]["muon_mini-isolation_SF"] = rl.NuisanceParameter(
        "muon_mini-isolation_SF", "lnN"
    )  # TODO: apply only for muon channel
    systs_dict_values["all_samples"]["muon_mini-isolation_SF"] = (1.02, 0.98)

    # lumi systematics
    if "2016" in years:
        systs_dict["all_samples"]["lumi_13TeV_2016"] = rl.NuisanceParameter("CMS_lumi_13TeV_2016", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2016"] = (1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi), None)
    if "2017" in years:
        systs_dict["all_samples"]["lumi_13TeV_2017"] = rl.NuisanceParameter("CMS_lumi_13TeV_2017", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2017"] = (1.02 ** (LUMI["2017"] / full_lumi), None)

    if "2018" in years:
        systs_dict["all_samples"]["lumi_13TeV_2018"] = rl.NuisanceParameter("CMS_lumi_13TeV_2018", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2018"] = (1.015 ** (LUMI["2018"] / full_lumi), None)

    if len(years) == 4:
        systs_dict["all_samples"]["lumi_13TeV_correlated"] = rl.NuisanceParameter("CMS_lumi_13TeV_corelated", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_correlated"] = (
            (
                (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
                * (1.009 ** (LUMI["2017"] / full_lumi))
                * (1.02 ** (LUMI["2018"] / full_lumi))
            ),
            None,
        )

        systs_dict["all_samples"]["lumi_13TeV_1718"] = rl.NuisanceParameter("CMS_lumi_13TeV_1718", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_1718"] = (
            (1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi)),
            None,
        )

    # PER SAMPLE SYSTEMATICS
    for sample in samples:
        systs_dict[sample], systs_dict_values[sample] = {}, {}

    # tagger eff
    n = rl.NuisanceParameter("taggereff", "lnN")
    for sample in sigs:
        systs_dict[sample]["taggereff"] = n
        systs_dict_values[sample]["taggereff"] = (1.30, None)

    # QCD scale
    n = rl.NuisanceParameter("QCD_scale", "lnN")
    systs_dict["ggF"]["QCD_scale"] = n
    systs_dict_values["ggF"]["QCD_scale"] = (1.039, 0.961)
    systs_dict["VBF"]["QCD_scale"] = n
    systs_dict_values["VBF"]["QCD_scale"] = (1.004, 0.997)
    systs_dict["ttH"]["QCD_scale"] = n
    systs_dict_values["ttH"]["QCD_scale"] = (1.058, 0.908)

    systs_dict["WH"]["QCD_scale"] = n
    systs_dict_values["WH"]["QCD_scale"] = (1.005, 0.993)
    systs_dict["ZH"]["QCD_scale"] = n
    systs_dict_values["ZH"]["QCD_scale"] = (1.038, 0.97)

    # PDF scale
    n = rl.NuisanceParameter("PDF_scale", "lnN")
    systs_dict["ggF"]["PDF_scale"] = n
    systs_dict_values["ggF"]["PDF_scale"] = (1.019, 0.981)
    systs_dict["VBF"]["PDF_scale"] = n
    systs_dict_values["VBF"]["PDF_scale"] = (1.021, 0.979)
    systs_dict["ttH"]["PDF_scale"] = n
    systs_dict_values["ttH"]["PDF_scale"] = (1.03, 0.97)

    systs_dict["WH"]["PDF_scale"] = n
    systs_dict_values["WH"]["PDF_scale"] = (1.017, 0.983)
    systs_dict["ZH"]["PDF_scale"] = n
    systs_dict_values["ZH"]["PDF_scale"] = (1.013, 0.987)

    return systs_dict, systs_dict_values


def systs_from_parquets(years):
    """
    Specify systematics that ARE stored in the parquets
    """
    if len(years) == 4:
        year = "Run2"
    else:
        year = years[0]

    systs_from_parquets = {
        "ele": {
            "all_samples": {
                "weight_ele_pileup": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PU_{year}", "shape"),
                "weight_ele_pileupIDSFDown": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PUIDSF_{year}", "shape"),
                "weight_ele_isolation_electron": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_isolation_{year}", "lnN"),
                "weight_ele_id_electron": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_identification", "lnN"),
                "weight_ele_reco_electron": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_reconstruction", "lnN"),
                "weight_ele_L1Prefiring": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_L1Prefiring_{year}", "lnN"),
                "weight_ele_trigger_electron": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_ele_trigger_{year}", "lnN"),
            },
            # signal
            "ggF": {
                "weight_ele_PSFSR_weight": rl.NuisanceParameter("PSFSR_ggF", "shape"),
                "weight_ele_PSISR_weight": rl.NuisanceParameter("PSISR_ggF", "shape"),
                "rec_higgs_mUES": rl.NuisanceParameter("AK8_UES", "shape"),
                "rec_higgs_mJES": rl.NuisanceParameter("AK8_JES", "shape"),
                "rec_higgs_mJER": rl.NuisanceParameter("AK8_JER", "shape"),
                "rec_higgs_mJMS": rl.NuisanceParameter("AK8_JMS", "shape"),
                "rec_higgs_mJMR": rl.NuisanceParameter("AK8_JMR", "shape"),
            },
            "VBF": {
                "weight_ele_PSFSR_weight": rl.NuisanceParameter("PSFSR_VBF", "shape"),
                "weight_ele_PSISR_weight": rl.NuisanceParameter("PSISR_VBF", "shape"),
                "rec_higgs_mUES": rl.NuisanceParameter("AK8_UES", "shape"),
                "rec_higgs_mJES": rl.NuisanceParameter("AK8_JES", "shape"),
                "rec_higgs_mJER": rl.NuisanceParameter("AK8_JER", "shape"),
                "rec_higgs_mJMS": rl.NuisanceParameter("AK8_JMS", "shape"),
                "rec_higgs_mJMR": rl.NuisanceParameter("AK8_JMR", "shape"),
            },
            "ttH": {},
            "WH": {},
            "ZH": {},
            # bkgs
            "QCD": {},
            "WJetsLNu": {
                "weight_ele_d1K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO_{year}", "lnN"),
                "weight_ele_d2K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO_{year}", "lnN"),
                "weight_ele_d3K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO_{year}", "lnN"),
                "weight_ele_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW_{year}", "lnN"),
                "weight_ele_W_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW_{year}", "lnN"),
                "weight_ele_W_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW_{year}", "lnN"),
            },
            "TTbar": {},
            "SingleTop": {},
            "Diboson": {},
            "EWKvjets": {},
            # "DYJets": {
            #     "weight_ele_d1K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d1kappa_EW_{year}", "lnN"),
            #     "weight_ele_Z_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW_{year}", "lnN"),
            #     "weight_ele_Z_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW_{year}", "lnN"),
            # },
            # "WZQQ": {},
            "WZQQorDYJets": {},
        },
        "mu": {
            "all_samples": {
                "weight_mu_pileup": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PU_{year}", "shape"),
                "weight_mu_pileupIDSFDown": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PUIDSF_{year}", "shape"),
                "weight_mu_isolation_muon": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_iso_{year}", "lnN"),
                "weight_mu_id_muon": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_id_{year}", "lnN"),
                "weight_mu_L1Prefiring": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_L1Prefiring_{year}", "lnN"),
                "weight_mu_trigger_iso_muon": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_iso_{year}_mu", "lnN"),
                "weight_mu_trigger_noniso_muon": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_{year}_mu", "lnN"),
            },
            # signal
            "ggF": {
                "weight_mu_PSFSR": rl.NuisanceParameter("PSFSR_ggF", "shape"),
                "weight_mu_PSISR": rl.NuisanceParameter("PSISR_ggF", "shape"),
                "rec_higgs_mUES": rl.NuisanceParameter("AK8_UES", "shape"),
                "rec_higgs_mJES": rl.NuisanceParameter("AK8_JES", "shape"),
                "rec_higgs_mJER": rl.NuisanceParameter("AK8_JER", "shape"),
                "rec_higgs_mJMS": rl.NuisanceParameter("AK8_JMS", "shape"),
                "rec_higgs_mJMR": rl.NuisanceParameter("AK8_JMR", "shape"),
            },
            "VBF": {
                "weight_mu_PSFSR": rl.NuisanceParameter("PSFSR_VBF", "shape"),
                "weight_mu_PSISR": rl.NuisanceParameter("PSISR_VBF", "shape"),
                "rec_higgs_mUES": rl.NuisanceParameter("AK8_UES", "shape"),
                "rec_higgs_mJES": rl.NuisanceParameter("AK8_JES", "shape"),
                "rec_higgs_mJER": rl.NuisanceParameter("AK8_JER", "shape"),
                "rec_higgs_mJMS": rl.NuisanceParameter("AK8_JMS", "shape"),
                "rec_higgs_mJMR": rl.NuisanceParameter("AK8_JMR", "shape"),
            },
            "ttH": {},
            "WH": {},
            "ZH": {},
            # bkgs
            "QCD": {},
            "WJetsLNu": {
                "weight_mu_d1K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO_{year}", "lnN"),
                "weight_mu_d2K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO_{year}", "lnN"),
                "weight_mu_d3K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO_{year}", "lnN"),
                "weight_mu_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW_{year}", "lnN"),
                "weight_mu_W_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW_{year}", "lnN"),
                "weight_mu_W_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW_{year}", "lnN"),
            },
            "TTbar": {},
            "SingleTop": {},
            "Diboson": {},
            "EWKvjets": {},
            # "DYJets": {
            #     "weight_mu_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d1kappa_EW_{year}", "lnN"),
            #     "weight_mu_Z_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW_{year}", "lnN"),
            #     "weight_mu_Z_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW_{year}", "lnN"),
            # },
            # "WZQQ": {},
            "WZQQorDYJets": {},
        },
    }

    # add btag SF
    for lepch in ["ele", "mu"]:
        for year in years:
            systs_from_parquets[lepch]["all_samples"][f"weight_btagSFlight{year}"] = rl.NuisanceParameter(
                f"{CMS_PARAMS_LABEL}_btagSFlight_{year}", "lnN"
            )
            systs_from_parquets[lepch]["all_samples"][f"weight_btagSFbc{year}"] = rl.NuisanceParameter(
                f"{CMS_PARAMS_LABEL}_btagSFbc_{year}", "lnN"
            )

        systs_from_parquets[lepch]["all_samples"]["weight_btagSFlightCorrelated"] = rl.NuisanceParameter(
            f"{CMS_PARAMS_LABEL}_btagSFlightCorrelated", "lnN"
        )
        systs_from_parquets[lepch]["all_samples"]["weight_btagSFbcCorrelated"] = rl.NuisanceParameter(
            f"{CMS_PARAMS_LABEL}_btagSFbcCorrelated", "lnN"
        )

    return systs_from_parquets