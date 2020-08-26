# HiChew

### What is it?

HiChew is the command-line tool to find optimal TADs or TADs boundaries segmentation and perform clustering based on D-scores or insulation scores on it. 
First and last assumption to use our tool is that you have embryo developmental stages stores in **balanced** coolfiles.
provided by access url (to ArrayExpress website) or E-MTAB .txt file or directly coolfiles (.cool).

### Why HiChew is useful?

HiChew makes your head free of choosing `gamma` parameter in segmentation methods Armatus and Modularity, 
or `window` parameter in TAD boundaries calling insulation score based segmentation method. Tool searches 
optimal `gamma` or `window` value itself! You just pass a grid of `gamma` or `window` parameter search (start, end, step).

HiChew also makes clustering of optimal segmentation. It is useful because recent researches shed a light upon the 
biological meaning of TADs clustering based on D-scores or insulation scores.
Thus, an easy-to-use tool should make the process of searching the best clustering faster.

### How to use it?

Just launch docker container and then one of the provided scripts `run_segmentation.py` and `run_clustering.py` 
inside the container (scripts have parameters -- please go inside the code, there are some insights and 
documentation).

Example for command-line usage (modularity or armatus):
```bash
git clone https://github.com/aence/hichew
cd hichew && make build -f Makefile
docker run -it --rm  -p 9990:9990 -v $(pwd):/hichew --name hichew-bash hichew-bash
# inside the container:
cd hichew/src && \
python3 run_segmentation.py -it coolfiles -ip ../data/coolfiles/E-MTAB-4918.sdrf -e DEMO_MODULARITY_60kb -eps 1e-1 -s 3-4h_repl_merged_5kb -res 5000 -chr X,2L,2R,3L,3R -m modularity -g 0,200.0,0.1 -e_mts 60000 -mis 2 -mts 1000 -pcnt 99.9 -vbc 1000
python3 run_clustering.py -sp ../data/experiments/DEMO_MODULARITY_60kb/opt_tads_modularity_60kb_5kb.csv -it coolfiles -ip ../data/coolfiles/E-MTAB-4918.sdrf -e DEMO_MODULARITY_60kb -mode range -m kmeans -nc 15 -s nuclear_cycle_12_repl_merged_5kb,nuclear_cycle_13_repl_merged_5kb,nuclear_cycle_14_repl_merged_5kb,3-4h_repl_merged_5kb -chr X,2L,2R,3L,3R -pcnt 99.9 -rs 42 -res 5000
python3 run_clustering.py -sp ../data/experiments/DEMO_MODULARITY_60kb/opt_tads_modularity_60kb_5kb.csv -it coolfiles -ip ../data/coolfiles/E-MTAB-4918.sdrf -e DEMO_MODULARITY_60kb -mode certain -m kmeans -nc 7 -s nuclear_cycle_12_repl_merged_5kb,nuclear_cycle_13_repl_merged_5kb,nuclear_cycle_14_repl_merged_5kb,3-4h_repl_merged_5kb -chr X,2L,2R,3L,3R -pcnt 99.9 -vbc 1000 -rs 42 -vs 3-4h_repl_merged_5kb -res 5000
# see all flags description and other documentation in code!
```

Example for command-line usage (insulation):
```bash
git clone https://github.com/aence/hichew
cd hichew && make build -f Makefile
docker run -it --rm  -p 9990:9990 -v $(pwd):/hichew --name hichew-bash hichew-bash
# inside the container:
cd hichew/src && \
python3 run_segmentation.py -it coolfiles -ip ../data/coolfiles/E-MTAB-4918.sdrf -e DEMO_INSULATION_60kb -eps 0.05 -s 3-4h_repl_merged_5kb -res 5000 -chr X,2L,2R,3L,3R -m insulation -g 0,200,1 -e_mts 60000 -mis 3 -mts 1000 -pcnt 99.9 -vbc 100
python3 run_clustering.py -sp ../data/experiments/DEMO_INSULATION_60kb/opt_tads_insulation_60kb_5kb.csv -it coolfiles -ip ../data/coolfiles/E-MTAB-4918.sdrf -e DEMO_INSULATION_60kb -mode range -m kmeans -nc 15 -s nuclear_cycle_12_repl_merged_5kb,nuclear_cycle_13_repl_merged_5kb,nuclear_cycle_14_repl_merged_5kb,3-4h_repl_merged_5kb -chr X,2L,2R,3L,3R -pcnt 99.9 -rs 42 -ins True -res 5000
python3 run_clustering.py -sp ../data/experiments/DEMO_INSULATION_60kb/opt_tads_insulation_60kb_5kb.csv -it coolfiles -ip ../data/coolfiles/E-MTAB-4918.sdrf -e DEMO_INSULATION_60kb -mode certain -m kmeans -nc 7 -s nuclear_cycle_12_repl_merged_5kb,nuclear_cycle_13_repl_merged_5kb,nuclear_cycle_14_repl_merged_5kb,3-4h_repl_merged_5kb -chr X,2L,2R,3L,3R -pcnt 99.9 -vbc 100 -rs 42 -vs 3-4h_repl_merged_5kb -ins True -res 5000
# see all flags description and other documentation in code!
```

For jupyter usage:
```bash
git clone https://github.com/aence/hichew
cd hichew && make build -f Makefile
docker run -it --rm  -p 9999:9999 -v $(pwd):/hichew --name hichew-jupyter hichew-jupyter
```

File `api.py` contains useful functions that you can use in jupyter or python environment (both for segmentation and clustering).
File `utils.py` contatins third-level-functions that `api.py` uses.

**Pool requests and issues are welcome!**

### Citing

Bykov N.S., Sigalova O.M., Gelfand M.S., Galitsyna A.A. (2020) HiChew: a Tool for TAD Clustering in Embryogenesis. In: Cai Z., Mandoiu I., Narasimhan G., Skums P., Guo X. (eds) Bioinformatics Research and Applications. ISBRA 2020. Lecture Notes in Computer Science, vol 12304. Springer, Cham. https://doi.org/10.1007/978-3-030-57821-3_37
