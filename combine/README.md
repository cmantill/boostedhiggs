# Making combine templates
Use the script `make_templates.py` to produce the templates before creating the datacard but first make sure to edit the `config_make_templates.yaml`.
```
python make_templates.py --years 2018,2017,2016APV,2016 --channels mu,ele --outdir templates/v1
```

Sample templates produced can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v6/) in `hists_templates_Run2.pkl`.

# Create the datacard

Use the script `create_datacard.py` to build a rhalphalib Model from the `hist` templates stored under `templates/v6` and create the datacard.
```
python create_datacard.py --years 2018,2017,2016APV,2016 --channels mu,ele --outdir templates/v6
```
Sample datacards can be found [here](https://github.com/farakiko/boostedhiggs/tree/main/combine/templates/v6/datacards).

# Run combine

Refer to `run.sh` for some useful commands.
