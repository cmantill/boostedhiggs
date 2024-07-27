#!/usr/bin/python

"""
Explores unproduced files due to condor job errors.
"""
import argparse
import glob
import json
import os
from math import ceil

from condor.file_utils import loadFiles


def main(args):
    # username = os.environ["USER"]

    # condordir = "/uscms/home/cmantill/nobackup/hww/boostedhiggs/condor/"
    condordir = "condor/"
    homedir = f"/store/user/{args.username}/boostedhiggs/"
    outdir = "/eos/uscms/" + homedir + args.tag + "_" + args.year + "/"

    # check only specific samples
    slist = args.slist.split(",") if args.slist is not None else None
    print(slist)

    metadata = f"{condordir}/{args.tag}_{args.year}/metadata_{args.configkey}.json"

    print(metadata)

    try:
        with open(metadata, "r") as f:
            files = json.load(f)
    except KeyError:
        raise Exception(f"Could not open file {metadata}")

    config = f"{condordir}/{args.tag}_{args.year}/{args.config}"
    splitname = f"{condordir}/{args.tag}_{args.year}/pfnano_splitting.yaml"

    print(f"Loading files from {splitname}")
    _, nfiles_per_job = loadFiles(config, args.configkey, args.year, args.pfnano, slist, splitname)

    samples = slist if args.slist is not None else files.keys()
    nfailed = 0
    # submit a cluster of jobs per sample
    for sample in samples:
        tot_files = len(files[sample])

        if sample not in nfiles_per_job.keys():
            continue

        njobs = ceil(tot_files / nfiles_per_job[sample])

        njobs_produced = len(glob.glob1(f"{outdir}/{sample}/outfiles", "*.pkl"))

        id_failed = []
        if njobs_produced != njobs:  # debug which pkl file wasn't produced
            print(f"-----> SAMPLE {sample} HAS RAN INTO ERROR, #jobs produced: {njobs_produced}, # jobs {njobs}")
            for i, x in enumerate(range(0, njobs * nfiles_per_job[sample], nfiles_per_job[sample])):
                fname = f"{x}-{x+nfiles_per_job[sample]}"
                print(f"{outdir}/{sample}/outfiles/{fname}.pkl")
                if not os.path.exists(f"{outdir}/{sample}/outfiles/{fname}.pkl"):
                    print(f"file {fname}.pkl wasn't produced which means job_idx {i} failed..")
                    id_failed.append(i)
                    nfailed = nfailed + 1
        else:
            print(f"Sample {sample} produced {njobs_produced} files - as expected")

        if args.resubmit and len(id_failed) > 0:
            fname = f"condor/{args.tag}_{args.year}/{sample}.jdl"
            condor_file = open(fname)

            resub_name = f"condor/{args.tag}_{args.year}/{sample}_resubmit.txt"
            iresub = 0
            while iresub < 100:
                if not os.path.exists(resub_name):
                    break
                iresub = iresub + 1
                resub_name = resub_name.replace(".txt", f"_{iresub}.txt")

            tfile = open(resub_name, "w")
            for i in id_failed:
                tfile.write(f"{i}\n")
            tfile.close()

            if iresub == 0:
                f_fail = fname.replace(".jdl", "_resubmit.jdl")
            else:
                f_fail = fname.replace(".jdl", f"_resubmit_{iresub}.jdl")
            condor_new = open(f_fail, "w")
            for line in condor_file:
                if "queue" in line:
                    line = f"queue jobid from {resub_name}"
                condor_new.write(line)
            condor_new.close()
            condor_file.close()

            print("Submit ", f_fail)
            os.system(f"condor_submit {f_fail}")

    print("-----------------------------------------------------------------------------------------")
    print(f"Number of jobs that failed: {nfailed}")


if __name__ == "__main__":
    """
    e.g.
    python check_jobs.py --username fmokhtar --config samples_inclusive.yaml --tag Sep12_QCD --key mc --year 2017
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2017", help="year", type=str)
    parser.add_argument(
        "--username",
        dest="username",
        default="cmantill",
        help="user who submitted the jobs",
        type=str,
    )
    parser.add_argument("--tag", dest="tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="path to datafiles, e.g. samples_inclusive.yaml",
        type=str,
    )
    parser.add_argument(
        "--key",
        dest="configkey",
        default=None,
        help="config key: [data, mc, ... ]",
        type=str,
    )
    parser.add_argument(
        "--slist",
        dest="slist",
        default=None,
        help="give sample list separated by commas",
    )
    parser.add_argument(
        "--pfnano",
        dest="pfnano",
        type=str,
        default="v2_2",
        help="pfnano version",
    )
    parser.add_argument("--resubmit", action="store_true")
    parser.add_argument("--no-resubmit", dest="resubmit", action="store_false")
    parser.set_defaults(resubmit=False)

    args = parser.parse_args()

    main(args)
