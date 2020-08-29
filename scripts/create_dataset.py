# -*- coding: utf-8 -*-
import codecs, json
import argparse

RECORD_DELIM = " "
DELIM = u"￨"
NUM_PLAYERS = 13


parser = argparse.ArgumentParser(description='Script to create dataset')
parser.add_argument('-json_root', type=str, default="",
                    help="path to json input")
parser.add_argument('-output_fi', type=str, default="",
                    help="desired path to output file")
parser.add_argument('-gen_fi', type=str, default="",
                    help="path to file containing generated summaries")
parser.add_argument('-dict_pfx', type=str, default="roto-ie",
                    help="prefix of .dict and .labels files")
parser.add_argument('-mode', type=str, default='train',
                    choices=['train', 'valid'],
                    help="create output dataset for train or valid")

args = parser.parse_args()


JSON_ROOT = args.json_root
ROOT = args.output_fi

if args.mode == 'train':
    JSON = JSON_ROOT + "train.json"
    SRC_FILE = ROOT + 'src_train.txt'
    TRAIN_TGT_FILE = ROOT + "tgt_train.txt"
else:
    JSON = JSON_ROOT +"valid.json"
    SRC_FILE = ROOT + 'src_valid.txt'
    TRAIN_TGT_FILE = ROOT + "tgt_valid.txt"


HOME = "HOME"
AWAY = "AWAY"

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

bs_keys = ["PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
     "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
     "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
     "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
     "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
    "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
    "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]


def box_preproc2(entry):
    records = []

    home_players, vis_players = get_player_idxs(entry)
    for ii, player_list in enumerate([home_players, vis_players]):
        for j in xrange(NUM_PLAYERS):
            player_key = player_list[j] if j < len(player_list) else None
            player_name = entry["box_score"]['PLAYER_NAME'][player_key] if player_key is not None else "N/A"
            for k, key in enumerate(bs_keys):
                rulkey = key.split('-')[1]
                val = entry["box_score"][rulkey][player_key] if player_key is not None else "N/A"
                record = []
                record.append(val.replace(" ","_"))
                record.append(player_name.replace(" ","_"))
                record.append(rulkey)
                record.append(HOME if ii == 0 else AWAY)
                records.append(DELIM.join(record))

    for k, key in enumerate(ls_keys):
        record = []
        record.append(entry["home_line"][key].replace(" ","_"))
        record.append(entry["home_line"]["TEAM-NAME"].replace(" ","_"))
        record.append(key)
        record.append(HOME)
        records.append(DELIM.join(record))
    for k, key in enumerate(ls_keys):
        record = []
        record.append(entry["vis_line"][key].replace(" ","_"))
        record.append(entry["vis_line"]["TEAM-NAME"].replace(" ","_"))
        record.append(key)
        record.append(AWAY)
        records.append(DELIM.join(record))

    return records


def get_player_idxs(entry):
    nplayers = 0
    home_players, vis_players = [], []
    for k, v in entry["box_score"]["PTS"].iteritems():
        nplayers += 1

    num_home, num_vis = 0, 0
    for i in xrange(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        if player_city == entry["home_city"]:
            if len(home_players) < NUM_PLAYERS:
                home_players.append(str(i))
                num_home += 1
        else:
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append(str(i))
                num_vis += 1

    if entry["home_city"] == entry["vis_city"] and entry["home_city"] == "Los Angeles":
        home_players, vis_players = [], []
        num_home, num_vis = 0, 0
        for i in xrange(nplayers):
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append(str(i))
                num_vis += 1
            elif len(home_players) < NUM_PLAYERS:
                home_players.append(str(i))
                num_home += 1

    return home_players, vis_players


with codecs.open(JSON, "r", "utf-8") as f:
    trdata = json.load(f)

summaries = []
src_instances = []
src_instance = ''
summary = ''

for entry in trdata:
    records = box_preproc2(entry)
    src_instance = " ".join(records)
    summary = entry['summary']
    src_instances.append(src_instance)
    summaries.append(summary)

summary_file = open(TRAIN_TGT_FILE,'w')
for summary in summaries:
    summary = [word.encode("utf-8") for word in summary]
    summary_file.write(" ".join(summary))
    summary_file.write("\n")
summary_file.close()

src_file = open(SRC_FILE, 'w')
for src_instance in src_instances:
    src_file.write(src_instance.encode("utf-8"))
    src_file.write("\n")
src_file.close()
