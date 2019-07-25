
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
The code for training with MLB dataset will be soon available on branch ```mlb```.
Scripts to create the MLB dataset are available at [mlb-data-scripts](https://github.com/ratishsp/mlb-data-scripts).

## Dataset

The boxscore-data json files can be downloaded from the [boxscore-data repo](https://github.com/harvardnlp/boxscore-data).

The input dataset for data2text-plan-py can be created by running the script ```create_dataset.py``` in ```scripts``` folder.
The dataset so obtained is available at link https://drive.google.com/open?id=1GvFBVvOa2YPy_X9aJ6KYLoz_CnqZN796

## Preprocessing
Assuming the OpenNMT-py input files reside at `~/boxscore-data`, the following command will preprocess the data

```
BASE=~/boxscore-data

mkdir $BASE/entity_preprocess
python preprocess.py -train_src $BASE/rotowire/src_train.txt -train_tgt $BASE/rotowire/tgt_train.txt -valid_src $BASE/rotowire/src_valid.txt -valid_tgt $BASE/rotowire/tgt_valid.txt -save_data $BASE/entity_preprocess/roto -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict
```

## Training (and Downloading Trained Models)
The command for training the Entity model is as follows:
```
BASE=~/boxscore-data
IDENTIFIER=cc
GPUID=0

python train.py -data $BASE/entity_preprocess/roto -save_model $BASE/gen_model/$IDENTIFIER/roto -encoder_type mean -input_feed 1 -layers 2 -batch_size 5 -feat_merge mlp -seed 1234 -report_every 100 -gpuid $GPUID -start_checkpoint_at 4 -epochs 25 -copy_attn -truncated_decoder 100 -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -entity_memory_size 300 -valid_batch_size 5
```
The Entity model can be downloaded from  https://drive.google.com/open?id=1vOGtTty57QJqjWAfW1tw0P2gHhpCUBAY

## Generation
During inference, we execute the following command:

```
MODEL_PATH=<path to model>

python translate.py -model $MODEL_PATH -src $BASE/rotowire/src_valid.txt -output $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt -batch_size 5 -max_length 850 -min_length 150 -gpu $GPUID
```

## Automatic evaluation using IE metrics
Metrics of RG, CS, CO are computed using the below commands.
```
python data_utils.py -mode prep_gen_data -gen_fi $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt -dict_pfx "roto-ie" -output_fi $BASE/transform_gen/roto_$IDENTIFIER-beam5_gens.h5 -input_path "/boxcore-json/rotowire"

th extractor.lua -gpuid  $GPUID -datafile roto-ie.h5 -preddata $BASE/transform_gen/roto_$IDENTIFIER-beam5_gens.h5 -dict_pfx "roto-ie" -just_eval 

python non_rg_metrics.py $BASE/transform_gen/roto-gold-val-beam5_gens.h5-tuples.txt $BASE/transform_gen/roto_$IDENTIFIER-beam5_gens.h5-tuples.txt 
```

## Evaluation using BLEU script
The BLEU perl script can be obtained from  https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
Command to compute BLEU score:
```
~/multi-bleu.perl $BASE/rotowire/tgt_valid.txt < $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt
```

## IE models
For training the IE models, follow the updated code in https://github.com/ratishsp/data2text-1 which contains bug fixes for number handling. The repo contains the downloadable links for IE models too.

