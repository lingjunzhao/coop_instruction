
import torch

import os
import time
import json
import sys
import numpy as np
import copy
from collections import defaultdict
from speaker import Speaker
from speaker_t5 import SpeakerT5

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation, format_results
from param import args

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter


log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

feedback_method = args.feedback # teacher or sample
decode_temp = args.decode_temp
decode_top_p = args.decode_top_p

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}, model='lstm'):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    if model == 'lstm':
        speaker = Speaker(train_env, listner, tok)
        require_decode = True
    elif model == 't5' or model == 't5_attention':
        speaker = SpeakerT5(train_env, listner, tok, model)
        require_decode = False
    else:
        sys.exit("Unknown speaker model: {}".format(model))

    if args.fast_train:
        log_every = 40

    print("val_envs:", val_envs)

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency  # TODO: comment out
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, path2log_prob, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            if require_decode:
                print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            else:
                print("Inference: ", path2inst[path_id])
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst, require_decode=require_decode)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)
            print('Current model %s env bleu %0.4f' % (env_name, bleu_score))
            print('Current model %s env loss %0.4f' % (env_name, loss))

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction, seed=args.seed)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    if args.fast_train:
        log_every = 40
    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:     # The default training process
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)   # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(feedback_method)
                    listner.env = aug_env

                    # Train with Back Translation
                    args.ml_weight = 0.6        # Sem-Configuration
                    listner.accumulate_gradient(feedback_method, speaker=speaker)
                    listner.optim_step()
            else:
                for _ in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    args.ml_weight = 0.2
                    listner.train(1, feedback=feedback_method)

                    # Train with Back Translation
                    listner.env = aug_env
                    args.ml_weight = 0.6
                    listner.train(1, feedback=feedback_method, speaker=speaker)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction, seed=args.seed)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            loss_str += format_results(score_summary)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1-alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric,val in score_summary.items():
                            if metric in ['success_rate']:
                                print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                      (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                   key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                  )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


def evaluate_with_outputs(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    feedback = args.decode_feedback
    sample_size = 10

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))
    print("Feedback method: ", feedback)
    print("Sample size: ", sample_size)

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        if feedback == 'argmax':
            iters = None
            agent.test(use_dropout=False, feedback=feedback, iters=iters)
            result = agent.get_results()

            score_summary, all_preds = evaluator.score(result)
            loss_str = "Env name: %s, " % env_name
            loss_str += format_results(score_summary)
            print(loss_str)

        elif feedback == 'sample':
            for k in range(sample_size):
                iters = None
                agent.test(use_dropout=False, feedback=feedback, iters=iters)
                result = agent.get_results()

                score_summary, all_preds = evaluator.score(result, sample_idx=k)
                evaluator.gt = all_preds
                loss_str = "Env name: %s, " % env_name
                loss_str += "Sample index: %s, " % str(k)
                loss_str += format_results(score_summary)
                print(loss_str)

        else:
            print("Unknown decode feedback method: ", feedback)
            sys.exit()

        if args.submit:
            filename = os.path.basename(env_name).split(".")[0]
            file_path = os.path.join(log_dir, "%s.json" % filename)
            with open(file_path, 'w') as f:
                json.dump(all_preds, f, indent=2)

            print("Saved eval info to ", file_path)


def setup(seed):
    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)
    print("Random seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup(args.seed)
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(args.features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok, train_sampling=args.train_sampling)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    #val_env_names = ['train']  # TODO: change back
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    #if not args.beam:
    #    val_env_names.append("train")

    use_generated_instr = False
    speaker_outputs = False
    if args.train == 'validspeakerdecodeprob':
        use_generated_instr = True
        val_env_names = [args.input_decode_json_file]
    elif args.train == 'validspeakerdecode':
        val_env_names = [args.input_decode_json_file]
    elif args.train == "eval_listener_outputs":
        speaker_outputs = True
        val_env_names = args.speaker_output_files
        print(args.speaker_output_files)

    print("Use generated instr as reference: ", use_generated_instr)
    print("Use speaker outputs: ", speaker_outputs)

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok, use_generated_instr=use_generated_instr, speaker_outputs=speaker_outputs),
           Evaluation([split], featurized_scans, tok, speaker_outputs=speaker_outputs))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == "eval_listener_outputs":
        evaluate_with_outputs(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs, model=args.speaker_model, log_every=args.log_every)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs, train_env, model=args.speaker_model)
    elif args.train == 'validspeakerdecode':
        valid_speaker_decode(tok, val_envs, train_env, args.input_decode_json_file, model=args.speaker_model)
    elif args.train == 'validspeakerdecodeprob':
        valid_speaker_decode_probs(tok, val_envs, train_env, args.input_decode_json_file, model=args.speaker_model,
                                   word_prob=args.word_prob)
    else:
        assert False


def valid_speaker(tok, val_envs, train_env, model='lstm'):
    import tqdm
    # listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    # speaker = Speaker(None, listner, tok)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    # speaker = Speaker(train_env, listner, tok)
    if model == 'lstm':
        speaker = Speaker(train_env, listner, tok)
        require_decode = True
    elif model == 't5' or model == 't5_attention':
        speaker = SpeakerT5(train_env, listner, tok, model)
        require_decode = False
    else:
        sys.exit("Unknown speaker model: {}".format(model))
    # speaker.load(args.load)
    path = os.path.join(log_dir, 'state_dict', 'best_val_unseen_bleu')
    print("Loaded the listener model at iter %d from %s" % (speaker.load(path), path))

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':  # TODO: uncomment
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, path2log_prob, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference prob: ", path2log_prob[path_id])
        if require_decode:
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        else:
            print("Inference: ", path2inst[path_id])
        print("GT: ", evaluator.gt[str(path_id)]['instructions'])
        bleu_score, precisions = evaluator.bleu_score(path2inst, require_decode=require_decode)
        print('BLEU score for %s env: bleu %0.4f' % (env_name, bleu_score))


def valid_speaker_decode(tok, val_envs, train_env, input_decode_json_file, model='lstm'):
    import tqdm
    # listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    # speaker = Speaker(None, listner, tok)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    # speaker = Speaker(train_env, listner, tok)
    if model == 'lstm':
        speaker = Speaker(train_env, listner, tok)
        require_decode = True
    elif model == 't5' or model == 't5_attention':
        speaker = SpeakerT5(train_env, listner, tok, model)
        require_decode = False
    else:
        sys.exit("Unknown speaker model: {}".format(model))
    # speaker.load(args.load)
    path = os.path.join(log_dir, 'state_dict', 'best_val_unseen_bleu')
    print("Loaded the listener model at iter %d from %s" % (speaker.load(path), path))
    decode_sampling_size = 10

    decoded_dir = log_dir + "/decoded_outputs"
    os.makedirs(decoded_dir, exist_ok=True)

    for env_name, (env, evaluator) in val_envs.items():
        #if env_name != 'val_seen':
        #    continue
        output_name, ext = os.path.splitext(env_name)
        greedy_output_file = os.path.join(decoded_dir, '%s_greedy_prob.pred' % output_name)
        sampled_output_file = os.path.join(decoded_dir,
                                           '%s_topP%.1f_temp%.1f_sampled.pred' % (
                                           output_name, decode_top_p, decode_temp))
        print("saving greedy outputs to %s" % greedy_output_file)
        print("saving sampled outputs to %s" % sampled_output_file)

        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2sampled_sents = defaultdict(list)

        # Greedy
        path2inst, path2log_prob, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference prob: ", path2log_prob[path_id])
        if require_decode:
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        else:
            print("Inference: ", path2inst[path_id])
        print("GT: ", evaluator.gt[str(path_id)]['instructions'])
        bleu_score, precisions = evaluator.bleu_score(path2inst, require_decode=require_decode)
        print('BLEU score for %s env: bleu %0.4f' % (env_name, bleu_score))

        #print("Average Length %0.4f" % utils.average_length(path2inst))
        for path_id, instr in path2inst.items():
            if require_decode:
                decoded_sent = tok.decode_sentence(path2inst[path_id])
            else:
                decoded_sent = path2inst[path_id]
            path2sampled_sents[path_id].append(decoded_sent)

        # Sampling
        for i in range(decode_sampling_size):
            path2inst, _, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm, sampling=True,
                                                                     decode_temp=decode_temp, decode_top_p=decode_top_p)
            for path_id, instr in path2inst.items():
                if require_decode:
                    decoded_sent = tok.decode_sentence(path2inst[path_id])
                else:
                    decoded_sent = path2inst[path_id]
                path2sampled_sents[path_id].append(decoded_sent)

        # print(path2sampled_sents[6757])
        print("Reading input decoding file from: ", input_decode_json_file)
        count_greedy, count_sampled = 0, 0
        instrid2greedy = {}
        instrid2sampled = {}
        with open(input_decode_json_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                # path_id = int(instr_id.split("_")[0])
                path_id = instr_id.split("_")[0]
                if path_id not in path2sampled_sents:
                    print("Path id not in decoded sents: ", path_id)
                    continue

                item.pop('result', None)
                item.pop('pred_path', None)

                instrid = "{}_0".format(path_id)
                new_item = copy.deepcopy(item)
                new_item["generated_instr"] = path2sampled_sents[path_id][0]
                new_item["instr_id"] = instrid
                new_item["generated_instr_sum_log_prob"] = np.sum(path2log_prob[path_id])
                new_item["generated_instr_avg_log_prob"] = np.average(path2log_prob[path_id])
                new_item["path_id"] = str(path_id)
                instrid2greedy[instrid] = new_item
                count_greedy += 1

                for k, instruction in enumerate(path2sampled_sents[path_id]):
                    instrid = "{}_{}".format(path_id, k)
                    new_item = copy.deepcopy(item)
                    new_item["generated_instr"] = instruction
                    new_item["instr_id"] = instrid
                    new_item["path_id"] = str(path_id)
                    instrid2sampled[instrid] = new_item
                    count_sampled += 1

        print("Number of greedy decoded sents: ", count_greedy)
        print("Number of sampled decoded sents: ", count_sampled)

        with open(greedy_output_file, 'w') as f:
            json.dump(instrid2greedy, f, indent=2)
        with open(sampled_output_file, 'w') as f:
            json.dump(instrid2sampled, f, indent=2)
        print('Saved eval info to %s' % greedy_output_file)
        print('Saved eval info to %s' % sampled_output_file)


def valid_speaker_decode_probs(tok, val_envs, train_env, input_decode_json_file, model='lstm', word_prob=False):
    import tqdm
    # listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    # speaker = Speaker(None, listner, tok)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    if model == 'lstm':
        speaker = Speaker(train_env, listner, tok)
        require_decode = True
        speaker_name = "clip"
        pytorch_total_params = sum(p.numel() for p in speaker.encoder.parameters())
        pytorch_total_params += sum(p.numel() for p in speaker.decoder.parameters())
    elif model == 't5' or model == 't5_attention':
        speaker = SpeakerT5(train_env, listner, tok, model)
        require_decode = False
        speaker_name = "t5"
        pytorch_total_params = sum(p.numel() for p in speaker.model.parameters())
    else:
        sys.exit("Unknown speaker model: {}".format(model))
    #speaker = Speaker(train_env, listner, tok)
    # speaker.load(args.load)
    path = os.path.join(log_dir, 'state_dict', 'best_val_unseen_bleu')
    print("Loaded the listener model at iter %d from %s" % (speaker.load(path), path))
    print("Number of parameters: ", pytorch_total_params)

    print("Decode word level prob: ", word_prob)

    for env_name, (env, evaluator) in val_envs.items():
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        instr2marginal_prob_list = None
        instr2lm_ratio_list = None
        if word_prob:
            instr2prob, instr2marginal_prob_list, instr2lm_ratio_list = speaker.get_instrs_word_probs(wrapper=tqdm.tqdm)
        else:
            instr2prob = speaker.get_instrs_probs(wrapper=tqdm.tqdm)

        decoded_dir = log_dir + "/decoded_outputs"
        os.makedirs(decoded_dir, exist_ok=True)

        print("Reading input decoding file from: ", input_decode_json_file)
        count = 0
        result_json = {}
        with open(input_decode_json_file) as f:
            tmp_data = json.load(f)
            for instr_id, item in tmp_data.items():
                pred_score = instr2prob[instr_id]

                if "speaker_result" not in item:
                    item["speaker_result"] = {speaker_name: pred_score}
                else:
                    item["speaker_result"][speaker_name] = pred_score

                if instr2marginal_prob_list:
                    item["speaker_result"][speaker_name + "_margin"] = instr2marginal_prob_list[instr_id]
                if instr2lm_ratio_list:
                    item["speaker_result"][speaker_name + "_lm_ratio"] = instr2lm_ratio_list[instr_id]

                result_json[instr_id] = item
                count += 1

        print("Number of decoded sents: ", count)
        filename = os.path.basename(env_name).split(".")[0]
        output_file = os.path.join(decoded_dir, '%s.json' % filename)
        with open(output_file, 'w') as f:
            json.dump(result_json, f, indent=2)
        print('Saved eval info to %s' % output_file)


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup(args.seed)

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(args.features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok, train_sampling=args.train_sampling)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug', train_sampling=args.train_sampling)

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker', 'validspeakerdecode', 'validspeakerdecodeprob',
                      'listener', 'validlistener', 'eval_listener_outputs']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

