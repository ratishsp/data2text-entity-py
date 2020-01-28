
# data2text-entity-py

This repo contains code for [Data-to-text Generation with Entity Modeling](https://www.aclweb.org/anthology/P19-1195) (Puduppully, R., Dong, L., & Lapata, M.; ACL 2019); this code is based on an earlier release (0.1) of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/tree/v0.1). The Pytorch version is 0.3.1.

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```
Note that the Pytorch version is 0.3.1 and Python version is 2.7.
The path to Pytorch wheel in ```requirements.txt``` is configured with CUDA 8.0. You may change it to the desired CUDA version.


## MLB
The code for training with MLB dataset is available on branch ```mlb```.
Scripts to create the MLB dataset are available at [mlb-data-scripts](https://github.com/ratishsp/mlb-data-scripts).


## Preprocessing
Assuming the OpenNMT-py input files reside at `~/boxscore-data`, the following command will preprocess the data

```
BASE=~/boxscore-data

mkdir $BASE/entity_preprocess
python preprocess.py -train_src $BASE/mlb/src_train.txt -train_tgt $BASE/mlb/tgt_train.txt -valid_src $BASE/mlb/src_valid.txt -valid_tgt $BASE/mlb/tgt_valid.txt -save_data $BASE/entity_preprocess/mlb -src_seq_length 10000 -tgt_seq_length 10000 -dynamic_dict -max_shard_size 26214400 -tgt_words_min_frequency 2
```

## Training (and Downloading Trained Models)
The command for training the Entity model is as follows:
```
BASE=~/boxscore-data
IDENTIFIER=cc
GPUID=0

python train.py -data $BASE/entity_preprocess/mlb -save_model $BASE/gen_model/$IDENTIFIER/mlb -encoder_type brnn -input_feed 1 -layers 1 -batch_size 12 -feat_merge mlp -seed 1234 -report_every 100 -gpuid $GPUID -start_checkpoint_at 4 -epochs 25 -copy_attn -truncated_decoder 100 -feat_vec_size 300 -word_vec_size 300 -rnn_size 600 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -entity_memory_size 300 -valid_batch_size 5
```
The Entity model can be downloaded from  https://drive.google.com/open?id=1oZSofzqYO6q8Egl-T4rmo13rM9V3N5Tf

## Generation
During inference, we execute the following command:

```
MODEL_PATH=<path to model>

python translate.py -model $MODEL_PATH -src $BASE/mlb/src_valid.txt -output $BASE/gen/mlb_$IDENTIFIER-beam5_gens.txt -batch_size 5 -max_length 1000 -min_length 300 -gpu $GPUID
```

## Automatic evaluation using IE metrics
Metrics of RG, CS, CO are computed following the commands in the [mlb-ie repo](https://github.com/ratishsp/mlb-ie).

## Evaluation using BLEU script
The BLEU perl script can be obtained from  https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
Command to compute BLEU score:
```
~/multi-bleu.perl $BASE/mlb/tgt_valid.txt < $BASE/gen/mlb_$IDENTIFIER-beam5_gens.txt
```


