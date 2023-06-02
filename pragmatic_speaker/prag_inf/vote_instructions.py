import argparse
import json
import numpy as np
import logging
import sys
from collections import defaultdict


def vote_instructions(input_file_list, output_file, result_sample, output_duplicate_instrs, output_all_instrs,
                      key="ndtw", metric="avg", no_prob=0, speaker_weight=0.0, speaker_file=None, speaker_model=None):
    instr2speaker_score = {}
    print("Speaker weight= ", speaker_weight)
    if speaker_weight > 0.0:
        if not speaker_file:
            sys.exit("Error: Speaker weight={}, but there is no speaker file".format(speaker_weight))
        else:
            with open(speaker_file) as f:
                tmp_data = json.load(f)
                for instr_id, item in tmp_data.items():
                    instr2speaker_score[instr_id] = item["speaker_result"][speaker_model]

    instrid2scores = defaultdict(list)
    path2instrids = defaultdict(list)

    instrid2scores_list = defaultdict(lambda: defaultdict(list))
    if no_prob:
        metrics = ['score', 'spl', 'ndtw', 'sdtw']
    else:
        metrics = ['score', 'spl', 'ndtw', 'sdtw', 'prob']

    count_scores = 0
    for input_file in input_file_list:
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                if not result_sample:
                    count_scores += 1
                    score = item['result'][key]
                    instrid2scores[instr_id].append(score)
                    for metric_key in metrics:
                        metric_score = item['result'][metric_key]
                        instrid2scores_list[instr_id][metric_key].append(float(metric_score))
                else:
                    scores = []
                    metric2scores = defaultdict(list)
                    for k in range(result_sample):
                        result_key = "result_sample_{}".format(k)
                        score = item[result_key][key]
                        scores.append(score)
                        count_scores += 1
                        for metric_key in metrics:
                            metric_score = item[result_key][metric_key]
                            metric2scores[metric_key].append(float(metric_score))
                    instrid2scores[instr_id].append(np.average(scores))  # TODO: change for product metric
                    for metric_key in metrics:
                        instrid2scores_list[instr_id][metric_key].append(np.average(metric2scores[metric_key]))

                path_id = instr_id.split("_")[0]
                if instr_id not in path2instrids[path_id]:
                    path2instrids[path_id].append(instr_id)

    print("Number of scores counted:", count_scores)

    best_instructions = []
    print("Agent scores metric: ", metric)
    if metric == "avg":
        best_instructions = best_avg(instrid2scores, path2instrids, output_duplicate_instrs,
                                     speaker_weight=speaker_weight, instr2speaker_score=instr2speaker_score)
    # elif metric == "median":
    #     best_instructions = best_median(instrid2scores, path2instrids)
    # elif metric == "mean-std":
    #     best_instructions = best_mean_std(instrid2scores, path2instrids)
    elif metric == "product":
        best_instructions = best_product(instrid2scores, path2instrids, output_duplicate_instrs)

    if output_all_instrs:
        best_instructions = list(instrid2scores.keys())

    # get instruction items by best ids
    all_preds = {}
    count = 0
    with open(input_file_list[0]) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            if instr_id in best_instructions:
                if result_sample:
                    for k in range(result_sample):
                        result_key = "result_sample_{}".format(k)
                        path_key = "pred_path_sample_{}".format(k)
                        del item[result_key]
                        del item[path_key]
                else:
                    del item["result"]
                    del item["pred_path"]
                metric_scores_list = instrid2scores_list[instr_id]
                overall_scores = {}
                for key, scores in metric_scores_list.items():
                    if metric == "avg":
                        overall_score = np.average(scores)
                    elif metric == "product":
                        # overall_score = sum(np.log(scores))
                        overall_score = np.product(scores)
                    overall_scores[key] = overall_score
                item['overall_voting_result'] = overall_scores
                all_preds[item['instr_id']] = item
                count += 1

    print("Number of output instructions: ", count)

    with open(output_file, 'w') as f:
        json.dump(all_preds, f, indent=2)
    print('Saved eval info to %s' % output_file)


def best_avg(instrid2scores, path2instrids, output_duplicate_instrs, speaker_weight=0.0, instr2speaker_score=None):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = np.array([np.average(instrid2scores[instr_id]) for instr_id in instr_ids])
        if speaker_weight:
            instr_speaker_scores = [instr2speaker_score[instr_id] for instr_id in instr_ids]
            instr_scores = np.array([pow(instr_scores[i], 1.0-speaker_weight) * pow(instr_speaker_scores[i], speaker_weight) for i in range(len(instr_speaker_scores))])
        max_score = max(instr_scores)
        if not output_duplicate_instrs:
            max_instr_idx = np.argmax(instr_scores)
            best_instructions.append(instr_ids[max_instr_idx])
        else:
        #elif max_score > 0.0:
            max_instr_indices = [i for i, j in enumerate(instr_scores) if j == max_score]
            best_instructions += [instr_ids[x] for x in max_instr_indices]

    return best_instructions


def best_product(instrid2scores, path2instrids, output_duplicate_instrs):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = np.array([sum(np.log(instrid2scores[instr_id])) for instr_id in instr_ids])
        max_score = max(instr_scores)
        if not output_duplicate_instrs:
            max_instr_idx = np.argmax(instr_scores)
            best_instructions.append(instr_ids[max_instr_idx])
        else:
        #elif max_score != 0.0:
            max_instr_indices = [i for i, j in enumerate(instr_scores) if j == max_score]
            best_instructions += [instr_ids[x] for x in max_instr_indices]

    return best_instructions


def best_median(instrid2scores, path2instrids):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = []
        for instr_id in instr_ids:
            scores = instrid2scores[instr_id]
            instr_scores.append(np.median(scores))
        instr_scores = np.array(instr_scores)
        max_instr_idx = np.argmax(instr_scores)
        best_instructions.append(instr_ids[max_instr_idx])
    return best_instructions


def best_mean_std(instrid2scores, path2instrids):
    best_instructions = []
    for path_id, instr_ids in path2instrids.items():
        instr_scores = []
        for instr_id in instr_ids:
            scores = instrid2scores[instr_id]
            instr_scores.append(np.average(scores) - np.std(scores))
        instr_scores = np.array(instr_scores)
        max_instr_idx = np.argmax(instr_scores)
        best_instructions.append(instr_ids[max_instr_idx])
    return best_instructions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_exp', help='output exp dir')
    parser.add_argument('--input_path', help='input file path')
    parser.add_argument('--output_duplicate_instrs', type=int, default=0)
    parser.add_argument('--output_all_instrs', type=int, default=0)
    parser.add_argument('--metric', type=str, default="avg")
    parser.add_argument('--no_prob', type=int, default=0)
    parser.add_argument('-input_exps', '--list', nargs='+', help='input exps list', required=True)
    parser.add_argument('--result_sample', type=int, default=0)  # 0, 10
    parser.add_argument('--speaker_weight', type=float, default=0.0)
    parser.add_argument('--speaker_file', type=str, default=None)
    parser.add_argument('--speaker_model', type=str, default=None)
    args = parser.parse_args()

    # metric = "avg"
    score_metric = "sdtw"
    print("Choose by best ", score_metric)
    print("Allow duplicate instrs: ", args.output_duplicate_instrs)
    print("Outputing all instrs: ", args.output_all_instrs)

    # input_file_list = ["snap/"+agent+"_pi_vote_speaker-clip/val_seen_sampled.json" for agent in args.list]
    #input_file_list = ["snap/" + agent + "_pi-vote-sample_speaker-gpt/speaker11_val_seen_eval.json" for agent in args.list]
    #input_file_list = ["snap/" + agent + "_pi_vote_speaker-perturb_ref/swap_entity_direction_val_unseen.json" for agent in args.list]
    #input_file_list = ["snap/" + agent + "_pi_vote_speaker-9models/9models_50routes_val_seen.json" for agent in args.list]
    #input_file_list = ["snap/" + agent + "_pi-vote-sample_speaker-clip-10/val_seen_sampled.json" for agent in args.list]
    #input_file_list = ["snap/" + agent + "_pi-vote_speaker-clip-10_test-sample/voted_best_avg_val_seen_eval.json" for agent in args.list]
    # input_file_list = ["snap/" + agent + "_pi_vote-sample_speaker-gpt_best10ila_beam/speaker-gpt_beam_vote_10ila_combined_val_seen.json" for agent in args.list]
    input_file_list = ["snap/" + agent + "_" + args.input_path for agent in args.list]
    print("Input file list: ", input_file_list)
    if args.output_all_instrs:
        output_file = args.output_exp + "voted_all_" + args.metric + "_val_seen_eval.json"
    else:
        output_file = args.output_exp + "voted_best_" + args.metric + "_val_seen_eval.json"
    print("Output file: ", output_file)
    vote_instructions(input_file_list, output_file, args.result_sample, args.output_duplicate_instrs, args.output_all_instrs,
                      metric=args.metric, key=score_metric, no_prob=args.no_prob,
                      speaker_weight=args.speaker_weight, speaker_file=args.speaker_file, speaker_model=args.speaker_model)
    #vote_instructions(input_file_list, output_file, args.result_sample, metric=metric, key="score")

    # input_file_list = ["snap/"+agent+"_pi_vote/speaker11_val_unseen_eval.json" for agent in args.list]
    # print("Input file list: ", input_file_list)
    # output_file = args.output_exp + "voted_best_" + metric + "_val_unseen_eval.json"
    # print("Output file: ", output_file)
    # vote_instructions(input_file_list, output_file, metric=metric)
