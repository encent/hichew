# HiChew

### What is it?

HiChew is the command-line tool to find optimal TADs segmentation and perform clustering based on D-scores on it. 
First and last assumption to use our tool is that you have embryo developmental stages stores in coolfiles, 
provided by access url (to ArrayExpress website) or E-MTAB .txt file or directly coolfiles (.cool).

### Why HiChew is useful?

HiChew makes your head free of choosing `gamma` parameter in segmentation methods Armatus and Modularity. Tool searches 
optimal `gamma` value itself! You just pass a grid of `gamma` parameter search (start, end, step).

HiChew also makes clustering of optimal segmentation. It is useful because recent researches shed a light upon the 
biological meaning of TADs clustering based on D-scores. Thus, an easy-to-use tool should make the process of searching 
the best clustering faster.

### How to use it?

Just launch docker container and then one of the provided scripts `run_segmentation.py` and `run_clustering.py` 
inside the container (scripts have parameters -- please go inside the code, threre are some insights and 
documentation).

For command-line usage:
```bash
git clone https://github.com/aence/hichew
cd hichew && make build -f Makefile
docker run -it --rm  -p 9990:9990 -v $(pwd):/hichew --name hichew-bash hichew-bash
# inside the container:
cd hichew && \
python3 run_segmentation.py -it url -ip https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-4918/E-MTAB-4918.sdrf.txt -eps 1e-1 -m modularity -g 0,200.0,0.1 -e_mts 60000 -mis 2 -exp_name DEMO_EXPERIMENT 
python3 run_clustering.py -exp_name DEMO_EXPERIMENT -sp ../data/experiments/E-MTAB-4918.sdrf/DEMO_EXPERIMENT/opt_tads_modularity_60kb_5kb.csv -mode range -m kmeans
python3 run_clustering.py -exp_name DEMO_EXPERIMENT -sp ../data/experiments/E-MTAB-4918.sdrf/DEMO_EXPERIMENT/opt_tads_modularity_60kb_5kb.csv -mode certain -m kmeans -nc 7
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

### That is it?

No. That is the first version of this tool (even probably alpha-version). Yet not founded bug-fixes and some additional 
functions will be provided as soon as possible. Watch for updates!
