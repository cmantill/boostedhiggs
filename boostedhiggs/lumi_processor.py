import os
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


class LumiProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        output_location="./outfiles/",
    ):
        self._year = year
        self._output_location = output_location

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def process(self, events: ak.Array):
        """Returns a set holding 3 values: (run number, lumi block, even number)."""
        dataset = events.metadata["dataset"]

        nevents = len(events)

        # # store as a parquet filec
        # lumilist = set(
        #     zip(
        #         events.run,
        #         events.luminosityBlock,
        #         events.event,
        #     )
        # )
        output = {}
        output["run"] = events.run
        output["luminosityBlock"] = events.luminosityBlock
        output["event"] = events.event

        # convert arrays to pandas
        if not isinstance(output, pd.DataFrame):
            output = self.ak_to_pandas(output)

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        if not os.path.exists(self._output_location + "/parquet"):
            os.makedirs(self._output_location + "/parquet")
        self.save_dfs_parquet(fname, output)

        # return dictionary with cutflows
        return {
            dataset: {
                self._year: {
                    "nevents": nevents,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator
