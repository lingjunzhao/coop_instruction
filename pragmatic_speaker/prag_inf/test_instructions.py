import argparse
import json
import numpy as np
import logging
from collections import defaultdict


def test_instructions(input_files, output_file):
    instrid2scores_list = defaultdict(lambda: defaultdict(list))
    metrics = ['dist', 'path_len', 'score', 'spl', 'ndtw', 'sdtw']
    metric2avg_scores, metric2std_scores = defaultdict(list), defaultdict(list)

    count_input = 0
    for input_file in input_files:
        with open(input_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                count_input += 1
                for key in metrics:
                    score = item['result'][key]
                    instrid2scores_list[instr_id][key].append(float(score))

    print("Number of input instructions: ", count_input)

    # get instruction items by best ids
    all_preds = {}
    count_output = 0
    with open(input_files[0]) as f:
        tmp_data = json.load(f)
        for instr_id, item in tmp_data.items():
            scores_list = instrid2scores_list[instr_id]
            avg_scores = {}
            for key, scores in scores_list.items():
                avg_score = np.average(scores)
                std_score = np.std(scores)
                avg_scores[key] = avg_score
                metric2avg_scores[key].append(avg_score)
                metric2std_scores[key].append(std_score)
            item['avg_result'] = avg_scores
            all_preds[item['instr_id']] = item
            count_output += 1

    print("Number of output instructions: ", count_output)

    with open(output_file, 'w') as f:
        json.dump(all_preds, f, indent=2)
    logging.info('Saved eval info to %s' % output_file)

    for metric, avg_scores in metric2avg_scores.items():
        overall_avg_score = np.average(avg_scores)
        overall_std_score = np.average(metric2std_scores[metric])
        print("Metric {}: avg={}".format(metric, overall_avg_score))
        print("Metric {}: std={}".format(metric, overall_std_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_exp', help='output exp dir')
    parser.add_argument('-input_files', '--list', nargs='+', help='input files list', required=True)
    args = parser.parse_args()

    input_file_list = args.list
    print("Input file list: ", input_file_list)
    output_file = args.output_exp + "epi_test_val_seen_eval.json"
    print("Output file: ", output_file)
    test_instructions(input_file_list, output_file)
