import importlib.resources
import json
import warnings

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate
from coffea.processor import ProcessorABC, column_accumulator

from boostedhiggs.corrections import (
    add_lepton_weight,
    add_pileup_weight,
    add_VJets_kFactors,
)
from boostedhiggs.utils import match_H

# we suppress ROOT warnings where our input ROOT tree has duplicate branches - these are handled correctly.
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


class TriggerEfficienciesProcessor(ProcessorABC):
    """Accumulates yields from all input events: 1) before triggers, and 2) after triggers"""

    def __init__(self, year="2017", yearmod=""):
        super(TriggerEfficienciesProcessor, self).__init__()

        self._year = year
        self._yearmod = yearmod
        self._channels = ["ele"]

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

    def pad_val(
        self,
        arr: ak.Array,
        target: int,
        value: float,
        axis: int = 0,
        to_numpy: bool = True,
    ):
        """pads awkward array up to `target` index along axis `axis` with value `value`,
        optionally converts to numpy array"""
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=True), value)
        return ret.to_numpy() if to_numpy else ret

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger histograms from input NanoAOD events"""
        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        sumgenweight = ak.sum(events.genWeight) if self.isMC else nevents

        self.weights = Weights(nevents, storeIndividual=True)
        self.weights_per_ch = {}

        def pad_val_nevents(arr: ak.Array):
            """pad values with the length equal to the number of events"""
            return self.pad_val(arr, nevents, -1)

        # skimmed events for different channels
        out = {}
        for ch in self._channels:
            out[ch] = {}
            out[ch]["triggers"] = {}

        for ch in self._channels:
            HLT_triggers = {}
            for trigger in self._HLTs[ch]:

                if trigger in events.HLT.fields:
                    HLT_triggers["HLT_" + trigger] = np.array(events.HLT[trigger])
                else:
                    HLT_triggers["HLT_" + trigger] = np.zeros(nevents, dtype="bool")

            out[ch]["triggers"] = {**out[ch]["triggers"], **HLT_triggers}

        ######################
        # Trigger
        ######################

        trigger = np.zeros(nevents, dtype="bool")
        for t in self._HLTs["mu"]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]

        ######################
        # METFLITERS
        ######################

        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        ######################
        # OBJECT DEFINITION
        ######################

        # OBJECT: taus
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        # OBJECT: muons
        muons = ak.with_field(events.Muon, 0, "flavor")

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | (muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2))
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.02)
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        goodmuons = muons[good_muons]
        goodmuons = goodmuons[ak.argsort(goodmuons.pt, ascending=False)]  # sort by pt
        candidate_mu = ak.firsts(goodmuons)

        # OBJECT: electrons
        electrons = ak.with_field(events.Electron, 1, "flavor")

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WP90)
            & (((electrons.pfRelIso03_all < 0.15) & (electrons.pt < 120)) | (electrons.pt >= 120))
            # additional cuts
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        goodelectrons = electrons[good_electrons]
        goodelectrons = goodelectrons[ak.argsort(goodelectrons.pt, ascending=False)]  # sort by pt
        candidate_ele = ak.firsts(goodelectrons)

        # OBJECT: candidate lepton
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        # OBJECT: AK8 fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = fatjets.msoftdrop
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        # OBJECT: candidate fatjet
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        met = events.MET

        NumFatjets = ak.num(good_fatjets)

        # delta R between AK8 jet and lepton
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        # delta phi MET and higgs candidate
        met_fj_dphi = candidatefj.delta_phi(met)

        ######################
        # Baseline weight
        ######################

        if self.isMC:
            self.weights.add("genweight", events.genWeight)
            self.weights.add(
                "L1Prefiring",
                events.L1PreFiringWeight.Nom,
                events.L1PreFiringWeight.Up,
                events.L1PreFiringWeight.Dn,
            )
            add_pileup_weight(self.weights, self._year, "", nPU=ak.to_numpy(events.Pileup.nPU))
            add_VJets_kFactors(self.weights, events.GenPart, dataset, events)

        ######################
        # Baseline selection
        ######################

        for ch in ["ele"]:
            add_lepton_weight(self.weights, candidatelep, self._year, "electron")

            selection = PackedSelection()
            selection.add("MuonTrigger", trigger)
            selection.add("METFilters", (metfilters))
            selection.add(
                "AtLeatOneTightMuon",
                (n_good_muons >= 1),
            )
            selection.add(
                "AtLeatOneTightElectron",
                (n_good_electrons >= 1),
            )
            selection.add("NoTaus", (n_loose_taus_ele == 0))
            selection.add("AtLeastOneFatJet", (NumFatjets >= 1))
            selection.add("CandidateJetpT", (candidatefj.pt > 250))
            selection.add("LepInJet", (lep_fj_dr < 0.8))
            selection.add("JetLepOverlap", (lep_fj_dr > 0.03))
            selection.add("dPhiJetMET", (np.abs(met_fj_dphi) < 1.57))
            selection.add("MET", (met.pt > 0))
            selection.add("CandidateJetSoftdropMass", (candidatefj.msdcorr > 40))

            ######################
            # variables to store
            ######################
            out[ch]["vars"] = {}
            out[ch]["vars"]["fj_pt"] = pad_val_nevents(candidatefj.pt)
            out[ch]["vars"]["fj_eta"] = pad_val_nevents(candidatefj.eta)
            out[ch]["vars"]["fj_msoftdrop"] = pad_val_nevents(candidatefj.msoftdrop)
            out[ch]["vars"]["met_pt"] = pad_val_nevents(met.pt)
            out[ch]["vars"]["lep_pt"] = pad_val_nevents(candidatelep.pt)
            out[ch]["vars"]["lep_eta"] = pad_val_nevents(candidatelep.eta)

            out[ch]["vars"]["ele_pt"] = pad_val_nevents(candidate_ele.pt)
            out[ch]["vars"]["ele_eta"] = pad_val_nevents(candidate_ele.eta)

            out[ch]["vars"]["mu_pt"] = pad_val_nevents(candidate_mu.pt)
            out[ch]["vars"]["mu_eta"] = pad_val_nevents(candidate_mu.eta)

            if "HToWW" in dataset:
                genVars, _ = match_H(events.GenPart, candidatefj)
                out[ch]["vars"]["fj_genH_pt"] = pad_val_nevents(genVars["fj_genH_pt"]).data

            out[ch]["weights"] = {}
            for key in self.weights._weights.keys():
                # store the individual weights (ONLY for now until we debug)
                out[ch]["weights"][f"weight_{key}"] = self.weights.partial_weight([key])
                if ch in self.weights_per_ch.keys():
                    self.weights_per_ch[ch].append(key)

            # use column accumulators
            for key_ in out[ch].keys():
                for key, value in out[ch][key_].items():
                    out[ch][key_][key] = column_accumulator(value[selection.all(*selection.names)])

        return {
            self._year + self._yearmod: {dataset: {"nevents": nevents, "sumgenweight": sumgenweight, "skimmed_events": out}}
        }

    def postprocess(self, accumulator):
        for year, datasets in accumulator.items():
            for dataset, output in datasets.items():
                for ch in output["skimmed_events"].keys():
                    for key_ in output["skimmed_events"][ch].keys():
                        output["skimmed_events"][ch][key_] = {
                            key: value.value for (key, value) in output["skimmed_events"][ch][key_].items()
                        }

        return accumulator
