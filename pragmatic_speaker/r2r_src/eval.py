''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import math
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs, load_speaker_outputs
from agent import BaseAgent


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok, speaker_outputs=False):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.speaker_outputs = speaker_outputs

        if speaker_outputs:
            self.data, self.scans = load_speaker_outputs(splits, None)
            for item in self.data:
                self.gt[item['instr_id']] = item
                self.instr_ids.append(item['instr_id'])

        else:
            for split in splits:
                for item in load_datasets([split]):
                    if scans is not None and item['scan'] not in scans:
                        continue
                    self.gt[str(item['path_id'])] = item
                    self.scans.append(item['scan'])
                    self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]

        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def compute_ndtw(self, scan, pred_path, gold_path):
         r = gold_path
         q = pred_path
         c = [[1e9] * (len(q) + 1) for _ in range(len(r) + 1)]
         c[0][0] = 0

         for i in range(1, len(r) + 1):
             for j in range(1, len(q) + 1):
                 d = self.distances[scan][r[i - 1]][q[j - 1]]
                 c[i][j] = min(c[i - 1][j], c[i][j - 1], c[i - 1][j - 1]) + d

         return math.exp(-c[len(r)][len(q)] / (len(r) * self.error_margin))

    def compute_sdtw(self, scan, pred_path, gold_path):
        d = self.distances[scan][pred_path[-1]][gold_path[-1]]
        if d > self.error_margin:
            return 0
        return self.compute_ndtw(scan, pred_path, gold_path)

    def _score_item(self, instr_id, path, sample_idx=None):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        result = {}
        if self.speaker_outputs:
            gt = self.gt[instr_id]
        else:
            gt = self.gt[instr_id.split('_')[-2]]

        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        dist = self.distances[gt['scan']][final_position][goal]
        self.scores['nav_errors'].append(dist)
        result['dist'] = dist
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        result['path_len'] = distance
        shortest_length = self.distances[gt['scan']][start][goal]
        self.scores['shortest_lengths'].append(
            shortest_length
        )
        result['score'] = dist <= self.error_margin
        result['spl'] = float(result['score']) * shortest_length / max(shortest_length, distance)

        pred_path = [x[0] for x in path]
        ndtw = self.compute_ndtw(gt['scan'], pred_path, gt['path'])
        sdtw = self.compute_sdtw(gt['scan'], pred_path, gt['path'])
        self.scores['ndtws'].append(ndtw)
        self.scores['sdtws'].append(sdtw)
        result['ndtw'] = ndtw
        result['sdtw'] = sdtw

        # save pred_path and scores
        if self.speaker_outputs:
            if sample_idx is None:
                self.gt[instr_id]["pred_path"] = pred_path
                self.gt[instr_id]["result"] = result
            else:
                self.gt[instr_id]["pred_path_sample_" + str(sample_idx)] = pred_path
                self.gt[instr_id]["result_sample_" + str(sample_idx)] = result


    def score(self, output_file, sample_idx=None):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file

        print('number of result: ', len(results))
        print('number of instr ids: ', len(instr_ids))
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'], sample_idx=sample_idx)
        print('number of remaining instr ids: ', len(instr_ids))
        # print(instr_ids)

        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in results'\
                            % (len(instr_ids), len(self.instr_ids), ",".join(self.splits))
            assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'ndtw': np.average(self.scores['ndtws']),
            'sdtw': np.average(self.scores['sdtws']),
        }
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)
        return score_summary, self.gt

    def bleu_score(self, path2inst, require_decode=True):
        from bleu import compute_bleu
        refs = []
        candidates = []
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            refs.append([self.tok.split_sentence(sent) for sent in self.gt[path_id]['instructions']])
            if require_decode:
                candidates.append([self.tok.index_to_word[word_id] for word_id in inst])
            else:
                candidates.append(self.tok.split_sentence(inst))

        tuple = compute_bleu(refs, candidates, smooth=False)
        bleu_score = tuple[0]
        precisions = tuple[1]

        return bleu_score, precisions


def format_results(result_dict):

    result_strings = []
    result_strings.append('score %.1f' % (result_dict['success_rate'] * 100))
    result_strings.append('spl %.1f' % (result_dict['spl'] * 100))
    result_strings.append('dist %.2f' % result_dict['nav_error'])
    result_strings.append('ndtw %.1f' % (result_dict['ndtw'] * 100))
    result_strings.append('sdtw %.1f' % (result_dict['sdtw'] * 100))
    result_strings.append('path_len %.1f' % result_dict['lengths'])

    return ', '.join(result_strings)

RESULT_DIR = 'tasks/R2R/results/'

def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen', 'test']:
        env = R2RBatch(None, batch_size=1, splits=[split])
        ev = Evaluation([split])

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    eval_simple_agents()






