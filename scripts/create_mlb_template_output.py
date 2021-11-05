import json, sys, os
INPUT = '<Path to valid/test json>'
OUTPUT = '<Output text file>'
NUM_PLAYERS  = 46
team_line = "The <team1> defeated the <team2> <pts1> - <pts2> ."
pitcher_line = '<pitcher> ( p_w - p_l ) allowed p_r runs , p_h hits and p_bb walks in p_ip1 p_ip2 innings .'
rbi_line = '<batter> hit <rbi> RBI <single> in the <inning> .'
home_run_line = '<batter> hit <rbi> RBI homer in the <inning> .'
wild_pitch_line = '<scorers> scored on wild pitch by <pitcher> in the <inning> .'
field_error_line = '<scorers> scored on field error by <fielder_error> in the <inning> .'
inning_map = {"1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth", "6": "sixth", "7": "seventh",
              "8": "eighth", "9": "ninth", "10": "tenth"}

pitching_attrib = ["bb", "h", "l", "r", "w", "ip1", "ip2"]
pitching_attrib = ["p_"+entry for entry in pitching_attrib]

def replace_all(text, dic):
    for i, j in list(dic.items()):
        text = text.replace(i, str(j))
    return text

file_outputs = []
with open(INPUT, encoding='utf-8') as json_data:
    d = json.load(json_data)
    for entry in d:
        outputs = []
        if int(entry["home_line"]["team_runs"]) > int(entry["vis_line"]["team_runs"]):
            team1_id = "home_line"
            team2_id = "vis_line"
        else:
            team1_id = "vis_line"
            team2_id = "home_line"
        team1 = entry[team1_id]["team_city"] + " "+ entry[team1_id]["team_name"]
        team2 = entry[team2_id]["team_city"] + " " + entry[team2_id]["team_name"]
        team1_pts = entry[team1_id]["team_runs"]
        team2_pts = entry[team2_id]["team_runs"]
        dic = {}
        dic["<team1>"] = team1
        dic["<team2>"] = team2
        dic["<pts1>"] = team1_pts
        dic["<pts2>"] = team2_pts
        output = replace_all(team_line, dic)
        outputs.append(output)


        home_players = list(range(46))
        vis_players = list(range(46,92))
        for ii, player_list in enumerate([home_players, vis_players]):
            for j in range(NUM_PLAYERS):
                player_key = str(player_list[j])
                #print "entry['box_score']['last_name']", json.dumps(entry['box_score']['last_name'], indent=4)
                player_name = entry['box_score']['full_name'][player_key]
                dic = {}
                for k, key in enumerate(pitching_attrib):
                    rulkey = key
                    if player_key not in entry["box_score"][rulkey]:
                        continue
                    val = entry["box_score"][rulkey][player_key]
                    if val == 'N/A':
                        continue
                    if key in ['p_ip1'] and int(val) == 0:
                        continue
                    dic[rulkey] = val
                if len(dic) > 0:
                    dic['<pitcher>'] = player_name
                    output = replace_all(pitcher_line, dic)
                    output = output.replace('p_ip1 ', '')
                    output = output.replace('p_ip2 ', '')
                    outputs.append(output)

        plays = entry["play_by_play"]
        for inning in range(1, len(entry['home_line']['innings'])+1):
            for top_bottom in ["top", "bottom"]:
                inning_plays = plays[str(inning)][top_bottom]
                play_index = 0
                for inning_play in inning_plays:
                    if inning_play["runs"] == 0:
                        continue
                    if inning_play["event"] in ['Single', 'Double', 'Triple']: #'Home Run',
                        dic = {}
                        dic['<batter>'] = inning_play['batter']
                        if 'rbi' in inning_play:
                            dic['<rbi>'] = inning_play['rbi']
                        if inning <= 10:
                            dic['<inning>'] = inning_map[str(inning)]
                        else:
                            dic['<inning>'] = str(inning) + "th"
                        dic['<single>'] = inning_play["event"].lower()
                        output = replace_all(rbi_line, dic)
                        output = output.replace('<rbi>', 'a')
                        outputs.append(output)
                    elif inning_play["event"] == 'Home Run':
                        dic = {}
                        dic['<batter>'] = inning_play['batter']
                        dic['<rbi>'] = inning_play['rbi']
                        if inning <= 10:
                            dic['<inning>'] = inning_map[str(inning)]
                        else:
                            dic['<inning>'] = str(inning) + "th"
                        output = replace_all(home_run_line, dic)
                        outputs.append(output)
                    elif inning_play["event"] == 'Wild Pitch':
                        dic = {}
                        if 'pitcher' in inning_play:
                            dic['<pitcher>'] = inning_play['pitcher']
                        if inning <= 10:
                            dic['<inning>'] = inning_map[str(inning)]
                        else:
                            dic['<inning>'] = str(inning) + "th"
                        if len(inning_play['scorers']) > 1:
                            dic['<scorers>'] = " , ".join(inning_play['scorers'])
                        else:
                            dic['<scorers>'] = inning_play['scorers'][0]
                        output = replace_all(wild_pitch_line, dic)
                        outputs.append(output)
                    elif inning_play["event"] == 'Field Error':
                        dic = {}
                        if 'fielder_error' in inning_play:
                            dic['<fielder_error>'] = inning_play['fielder_error']
                        if inning <= 10:
                            dic['<inning>'] = inning_map[str(inning)]
                        else:
                            dic['<inning>'] = str(inning) + "th"
                        if len(inning_play['scorers']) > 1:
                            dic['<scorers>'] = " , ".join(inning_play['scorers'])
                        else:
                            dic['<scorers>'] = inning_play['scorers'][0]
                        output = replace_all(field_error_line, dic)
                        outputs.append(output)
        file_outputs.append(" ".join(outputs))

output_file = open(OUTPUT, mode='w', encoding='utf-8')
output_file.write("\n".join(file_outputs))
output_file.write("\n")
output_file.close()
