######## Script to combine datacards and run asimov significance to study optimal working points.

cards_dir="/uscms/home/fmokhtar/nobackup/boostedhiggs/combine/templates/v6_sig/datacards"

####################################################################################################
# Combine cards
####################################################################################################

# outdir is the combined directory with the combine.txt datafile
outdir=${cards_dir}/combined
mkdir -p ${outdir}
chmod +x ${outdir}

combined_datacard=${outdir}/combined.txt
ws=${outdir}/workspace.root

# # ADD REGIONS
sr1="ggF90pt500toInf"
# sr2="ggF93pt250to350"
# sr3="ggF93pt350to500"
# sr4="ggF93pt500toInf"
ccargs="SR1=${cards_dir}/${sr1}.txt"
# ccargs="SR1=${cards_dir}/${sr1}.txt SR2=${cards_dir}/${sr2}.txt"
# ccargs="SR1=${cards_dir}/${sr1}.txt SR2=${cards_dir}/${sr2}.txt SR3=${cards_dir}/${sr3}.txt SR4=${cards_dir}/${sr4}.txt"

cr1="TopCR"
cr2="WJetsCR"
ccargs+=" CR1=${cards_dir}/${cr1}.txt CR2=${cards_dir}/${cr2}.txt"

####################################################################################################
# Combine datacards
####################################################################################################

echo "Combining cards:"
for file in $ccargs; do
  echo "  ${file##*/}"
done
echo "-------------------------"
combineCards.py $ccargs > $combined_datacard

####################################################################################################
# Run asimov significance
####################################################################################################

echo "Running text2workspace"
text2workspace.py $combined_datacard --channel-masks -o $ws 2>&1

echo "Expected significance"
combine -M Significance -d $ws -m 125 --expectSignal=1 --rMin -1 --rMax 5 -t -1
