# Making stacked histograms

We build stacked histograms after processing first the parquets into an `events_dict` object.

To make the `events_dict` object, run the command `python make_stacked_hists.py --make-events-dict` but make sure to first edit the `config_make_events_dict.yaml`.

To plot the stacked histograms after building the `events_dict` object, run the command `python make_stacked_hists.py --plot-stacked-hists` but make sure to first edit the `config_plot_stacked_hists.yaml`.