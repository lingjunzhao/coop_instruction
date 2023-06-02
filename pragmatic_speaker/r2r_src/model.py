import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from transformers import T5Tokenizer, T5EncoderModel, AdamW, get_linear_schedule_with_warmup, \
    T5ForConditionalGeneration, AutoConfig


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class MLP(nn.Module):

    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.ReLU, device="cuda"):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias).to(device))
            if act:
                layers.append(act().to(device))
        self.model = nn.Sequential(*layers)
        print(self.model)


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class SpeakerT5(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, transformer_dropout_ratio, num_transformer_layers, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.dropout_ratio = dropout_ratio
        self.transformer_dropout_ratio = transformer_dropout_ratio
        self.device = 'cuda'

        # self.load_model(transformer_dropout_ratio)
        self.load_model(transformer_dropout_ratio, num_transformer_layers)

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional).to(self.device)
        self.drop = nn.Dropout(p=dropout_ratio).to(self.device)
        self.drop3 = nn.Dropout(p=args.featdropout).to(self.device)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size).to(self.device)

        self.max_instruction_length = 200

    def load_model(self, transformer_dropout_ratio, num_transformer_layers):  # without pretrained
        self.special_tokens = {'bos': '<pad>',
                               'eos': '</s>',
                               'pad': '<pad>'}
        model_name = "t5-small"  # "t5-base" "t5-small"
        self.emb_size = 512  # 768 512
        self.t5_configuration = AutoConfig.from_pretrained(model_name)
        self.t5_configuration.dropout_rate = transformer_dropout_ratio
        self.t5_configuration.num_layers = num_transformer_layers
        self.t5_configuration.num_decoder_layers = num_transformer_layers
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # self.transformer = T5ForConditionalGeneration.from_pretrained(model_name, config=self.t5_configuration).to(self.device)
        self.transformer = T5ForConditionalGeneration(config=self.t5_configuration).to(self.device)
        # self.encoder = T5EncoderModel.from_pretrained(model_name, config=self.t5_configuration).to(self.device)
        self.encoder = T5EncoderModel(config=self.t5_configuration).to(self.device)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_seq_length = 1024
        print("Transformer dropout rate: ", transformer_dropout_ratio)
        print("Transformer num_transformer_layers: ", num_transformer_layers)
        print(
            "The end of sequence token {} has the id {}".format(self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id),
                                                                self.tokenizer.eos_token_id))
        print("The padding token {} has the id {}".format(self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id),
                                                          self.tokenizer.pad_token_id))

    def load_model_pretrain(self, transformer_dropout_ratio, num_transformer_layers):
        self.special_tokens = {'bos': '<pad>',
                               'eos': '</s>',
                               'pad': '<pad>'}
        model_name = "t5-large"  # "t5-base"
        self.emb_size = 1024  # 768
        self.t5_configuration = AutoConfig.from_pretrained(model_name)
        self.t5_configuration.dropout_rate = transformer_dropout_ratio
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.transformer = T5ForConditionalGeneration.from_pretrained(model_name, config=self.t5_configuration).to(self.device)
        self.encoder = T5EncoderModel.from_pretrained(model_name, config=self.t5_configuration).to(self.device)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.max_seq_length = 1024
        print("Transformer dropout rate: ", transformer_dropout_ratio)
        print(
            "The end of sequence token {} has the id {}".format(self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id),
                                                                self.tokenizer.eos_token_id))
        print("The padding token {} has the id {}".format(self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id),
                                                          self.tokenizer.pad_token_id))

    def get_instruction_tensor(self, instructions):
        original_instruction_tokens_seqs =[]
        decoder_instruction_tokens_seqs = []
        for instruction in instructions:
            input_ids = self.tokenizer(instruction, max_length=self.max_instruction_length, padding="max_length",
                                       truncation=True,  return_tensors="pt").input_ids.to(self.device)
            decoder_ids = self.tokenizer(instruction, max_length=self.max_instruction_length, padding="max_length",
                                         truncation=True, return_tensors="pt").input_ids.to(self.device)
            decoder_ids = self.transformer._shift_right(decoder_ids)
            original_instruction_tokens_seqs.append(input_ids)
            decoder_instruction_tokens_seqs.append(decoder_ids)

        original_instruction_tokens_seqs = torch.stack(original_instruction_tokens_seqs)
        decoder_instruction_tokens_seqs = torch.stack(decoder_instruction_tokens_seqs)

        return original_instruction_tokens_seqs, decoder_instruction_tokens_seqs

    def prepare_inputs(self, action_embeds, feature, already_dropfeat):
        x = action_embeds.to(self.device)
        feature = feature.to(self.device)
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        # print("max_length: ", max_length)
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        return x

    def predict(self, action_embeds, feature, lengths, instructions, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = self.prepare_inputs(action_embeds, feature, already_dropfeat)

        original_instruction_tokens_seqs, decoder_instruction_tokens_seqs = self.get_instruction_tensor(instructions)
        original_instruction_tokens_seqs = original_instruction_tokens_seqs.squeeze()
        if len(original_instruction_tokens_seqs.size()) == 1:
            original_instruction_tokens_seqs = original_instruction_tokens_seqs[None, :]
        # print("x shape: ", x.size())
        # print("tokens shape: ", original_instruction_tokens_seqs.size())
        # transformer layer
        outputs = self.transformer(inputs_embeds=x, labels=original_instruction_tokens_seqs)

        return outputs, original_instruction_tokens_seqs

    def compute_word_probs(self, action_embeds, feature, lengths, instructions, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        string_punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        sp_prefix = "▁".encode('utf-8')

        with torch.no_grad():
            x = self.prepare_inputs(action_embeds, feature, already_dropfeat)

            original_instruction_tokens_seqs, decoder_instruction_tokens_seqs = self.get_instruction_tensor(instructions)
            original_instruction_tokens_seqs = original_instruction_tokens_seqs.squeeze()
            if len(original_instruction_tokens_seqs.size()) == 1:
                original_instruction_tokens_seqs = original_instruction_tokens_seqs[None, :]
            # print("x shape: ", x.size())
            # print("tokens shape: ", original_instruction_tokens_seqs.size())
            # transformer layer
            # shape = list(x.size())
            # fake_generated = torch.zeros(shape, dtype=torch.double).double().to(self.device)

            # outputs = self.transformer(inputs_embeds=x, labels=original_instruction_tokens_seqs)
            # probs = outputs.logits.softmax(-1)

            batch_word_prob_list, batch_marginal_word_prob_list = [], []
            batch_word_prob_lm_ratio_list = []

            for j, instruction in enumerate(instructions):
                obs_emb = x[j][None, :].to(self.device)
                tokens_seq = original_instruction_tokens_seqs[j]
                tokens_seq = tokens_seq[None, :]
                outputs = self.transformer(inputs_embeds=obs_emb, labels=tokens_seq)
                obs_shape = list(obs_emb.size())
                lm_obs = torch.zeros(obs_shape, dtype=torch.float32).to(self.device)
                lm_outputs = self.transformer(inputs_embeds=lm_obs, labels=tokens_seq)

                logits = outputs.logits[0]
                probs = F.softmax(logits, -1)
                lm_logits = lm_outputs.logits[0]
                lm_probs = F.softmax(lm_logits, -1)
                input_ids = self.tokenizer(instruction, max_length=self.max_instruction_length,
                                           truncation=True, return_tensors="pt").input_ids.to(self.device)

                ids = input_ids.tolist()[0]
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                tokens = [str(x).encode('utf-8') for x in tokens[:-1]]

                word_prob_list = []
                marginal_word_prob_list = []
                word_prob_lm_ratio_list = []
                for i, id in enumerate(ids):
                    p = probs[i, id].item()
                    token = self.tokenizer.decode(id)
                    # word_prob = f'{token} ({round(p, 3)})'
                    word_prob = (token, p)
                    word_prob_list.append(word_prob)

                    list_probs = probs[i, :].tolist()
                    # list_probs = probs[j, i, :].tolist()
                    sorted_prob = sorted(list_probs, reverse=True)
                    # print(word_prob)
                    # print(sorted_prob[:10])
                    # max_idx = list_probs.index(max(list_probs))
                    # max_token = self.tokenizer.decode(max_idx)
                    # print("max token: ", max_token)
                    next_id = sorted_prob.index(p) + 1
                    next_prob = sorted_prob[next_id]
                    # print("next prob: ", next_prob)
                    marginal_prob = p - next_prob
                    marginal_word_prob_list.append(f'{token} ({round(marginal_prob, 3)})')

                    lm_p = lm_probs[i, id].item()
                    lm_ratio = np.log(p / lm_p)
                    word_prob_lm_ratio_list.append(f'{token} ({round(lm_ratio, 3)})')

                # combine subword prob into word prob
                new_word_prob_list = []
                cached_subwords = []
                cached_probs = []
                for k, (token, p) in enumerate(word_prob_list[:-1]):
                    if ((sp_prefix in tokens[k]) and cached_subwords) or (token in string_punctuation):
                        new_word = "".join(cached_subwords)
                        # new_word = new_word.replace("▁", '')
                        new_prob = np.prod(cached_probs)
                        new_word_prob_list.append(f'{new_word} ({round(new_prob, 3)})')
                        cached_subwords = []
                        cached_probs = []

                    if token not in string_punctuation:
                        cached_subwords.append(token)
                        cached_probs.append(p)
                    else:
                        new_word_prob_list.append(f'{token} ({round(p, 3)})')

                if cached_subwords:
                    new_word = "".join(cached_subwords)
                    # new_word = new_word.replace("▁", '')
                    new_prob = np.prod(cached_probs)
                    new_word_prob_list.append(f'{new_word} ({round(new_prob, 3)})')

                word_prob_list = " ".join(new_word_prob_list)
                marginal_word_prob_list = " ".join(marginal_word_prob_list)
                word_prob_lm_ratio_list = " ".join(word_prob_lm_ratio_list)
                batch_word_prob_list.append(word_prob_list)
                batch_marginal_word_prob_list.append(marginal_word_prob_list)
                batch_word_prob_lm_ratio_list.append(word_prob_lm_ratio_list)

        return outputs, batch_word_prob_list, batch_marginal_word_prob_list, batch_word_prob_lm_ratio_list

    def compute_word_probs1(self, action_embeds, feature, lengths, instructions, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        with torch.no_grad():
            x = self.prepare_inputs(action_embeds, feature, already_dropfeat)
            obs_emb = x[0]
            # obs_emb = x
            stop_token_index = self.tokenizer.eos_token_id
            word_prob_list = []
            marginal_word_prob_list = []

            log_probs = []
            tokens = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
            generated = obs_emb[None, :].to(self.device)

            print("instruction: ", instructions)

            for i in range(self.max_instruction_length):
                outputs = self.transformer(inputs_embeds=generated, decoder_input_ids=tokens)
                logits = outputs.logits  # [batch size, seq length, vocab size]
                logits = logits[:, -1, :]
                next_token = torch.argmax(logits, -1).unsqueeze(0).to(self.device)
                next_token_embed = self.transformer.encoder.embed_tokens(next_token)
                tokens = torch.cat((tokens, next_token), dim=1)
                # generated = torch.cat((generated, next_token_embed), dim=1)
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                log_prob = m.log_prob(next_token).detach()
                log_probs.append(log_prob)
                token_string = self.tokenizer.decode(next_token[0][0])

                list_probs = probs[0].tolist()
                p = list_probs[next_token]
                word_prob = f'{token_string} ({round(p, 3)})'
                word_prob_list.append(word_prob)

                sorted_prob = sorted(list_probs, reverse=True)
                print(word_prob)
                print(sorted_prob[:10])
                # max_idx = list_probs.index(max(list_probs))
                # max_token = self.tokenizer.decode(max_idx)
                # print("max token: ", max_token)
                next_id = sorted_prob.index(p) + 1
                next_prob = sorted_prob[next_id]
                print("next prob: ", next_prob)
                marginal_prob = p - next_prob
                marginal_word_prob_list.append(f'{token_string} ({round(marginal_prob, 3)})')

                if stop_token_index == next_token.item():
                    break

            word_prob_list = " ".join(word_prob_list)
            marginal_word_prob_list = " ".join(marginal_word_prob_list)

        return outputs, word_prob_list, marginal_word_prob_list

    def decode_greedy(self, action_embeds, feature, already_dropfeat=False, temperature=1.0):
        with torch.no_grad():
            x = self.prepare_inputs(action_embeds, feature, already_dropfeat)

            #self.model.eval()
            stop_token_index = self.tokenizer.eos_token_id
            remove_tokens = list(self.special_tokens.values())
            generated_list = []
            batch_log_probs = []

            for obs_emb in x:
                log_probs = []
                tokens = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
                generated = obs_emb[None, :].to(self.device)

                for i in range(self.max_instruction_length):
                    outputs = self.transformer(inputs_embeds=generated, decoder_input_ids=tokens)
                    logits = outputs.logits  # [batch size, seq length, vocab size]
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # [batch size, vocab size]. last logit from sequence
                    next_token = torch.argmax(logits, -1).unsqueeze(0).to(self.device)
                    next_token_embed = self.transformer.encoder.embed_tokens(next_token)
                    tokens = torch.cat((tokens, next_token), dim=1)
                    # generated = torch.cat((generated, next_token_embed), dim=1)
                    probs = F.softmax(logits, -1)
                    m = torch.distributions.Categorical(probs)
                    log_prob = m.log_prob(next_token)
                    log_probs.append(log_prob.detach())
                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = self.tokenizer.decode(output_list, skip_special_tokens=True)
                # batch_log_probs.append(log_probs)
                batch_log_probs.append([0.0])

                #split_text = output_text.split()
                #clean_text = ' '.join([token for token in split_text if token not in remove_tokens]).encode('utf-8')
                #print(str(output_text))
                if output_text:
                    generated_list.append(output_text)
                else:
                    generated_list.append('')

        return generated_list, batch_log_probs

    def decode_greedy_word_prob(self, action_embeds, feature, already_dropfeat=False, temperature=1.0):
        with torch.no_grad():
            x = self.prepare_inputs(action_embeds, feature, already_dropfeat)

            #self.model.eval()
            stop_token_index = self.tokenizer.eos_token_id
            remove_tokens = list(self.special_tokens.values())
            generated_list = []
            batch_log_probs = []

            for obs_emb in x:
                log_probs = []
                tokens = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)
                generated = obs_emb[None, :].to(self.device)
                word_prob_list = []
                marginal_word_prob_list = []

                for i in range(self.max_instruction_length):
                    outputs = self.transformer(inputs_embeds=generated, decoder_input_ids=tokens)
                    logits = outputs.logits  # [batch size, seq length, vocab size]
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # [batch size, vocab size]. last logit from sequence
                    next_token = torch.argmax(logits, -1).unsqueeze(0).to(self.device)
                    next_token_embed = self.transformer.encoder.embed_tokens(next_token)
                    tokens = torch.cat((tokens, next_token), dim=1)
                    # generated = torch.cat((generated, next_token_embed), dim=1)
                    probs = F.softmax(logits, -1)
                    m = torch.distributions.Categorical(probs)
                    log_prob = m.log_prob(next_token)
                    log_probs.append(log_prob.detach())

                    token_string = self.tokenizer.decode(next_token[0][0])
                    list_probs = probs[0].tolist()
                    p = list_probs[next_token]
                    word_prob = f'{token_string} ({round(p, 3)})'
                    word_prob_list.append(word_prob)

                    sorted_prob = sorted(list_probs, reverse=True)
                    print(word_prob)
                    print(sorted_prob[:10])
                    # max_idx = list_probs.index(max(list_probs))
                    # max_token = self.tokenizer.decode(max_idx)
                    # print("max token: ", max_token)
                    next_id = sorted_prob.index(p) + 1
                    next_prob = sorted_prob[next_id]
                    print("next prob: ", next_prob)
                    marginal_prob = p - next_prob
                    marginal_word_prob_list.append(f'{token_string} ({round(marginal_prob, 3)})')

                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())
                # output_text = self.tokenizer.decode(output_list, skip_special_tokens=True)
                output_text = " ".join(word_prob_list)
                # print("output_text_1: ", output_text_1)
                # print("output_text: ", output_text)
                # batch_log_probs.append(log_probs)
                batch_log_probs.append([0.0])

                #split_text = output_text.split()
                #clean_text = ' '.join([token for token in split_text if token not in remove_tokens]).encode('utf-8')
                #print(str(output_text))
                if output_text:
                    generated_list.append(output_text)
                else:
                    generated_list.append('')

        return generated_list, batch_log_probs

    def decode_batch_greedy(self, action_embeds, feature, already_dropfeat=False, temperature=1.0):
        with torch.no_grad():
            x = self.prepare_inputs(action_embeds, feature, already_dropfeat)

            stop_token_index = self.tokenizer.eos_token_id
            pad_token_index = self.tokenizer.pad_token_id
            generated_list = []

            batch_tokens = torch.tensor([[self.tokenizer.pad_token_id] for j in range(len(x))]).to(self.device)
            ended = np.zeros(len(x), np.bool)
            log_probs = []
            # print("initial batch_tokens size: ", batch_tokens.size())

            for i in range(self.max_instruction_length):
                outputs = self.transformer(inputs_embeds=x, decoder_input_ids=batch_tokens)
                logits = outputs.logits  # [batch size, seq length, vocab size]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # [batch size, vocab size]. last logit from sequence
                next_token = torch.argmax(logits, -1).unsqueeze(0).to(self.device)
                transposed_next_token = torch.transpose(next_token, 0, 1)
                # print("transposed next_token size: ", transposed_next_token.size())
                batch_tokens = torch.cat((batch_tokens, transposed_next_token), dim=1)
                # print("batch_tokens size: ", batch_tokens.size())
                # generated = torch.cat((generated, next_token_embed), dim=1)
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                log_prob = m.log_prob(next_token)
                #log_probs.append(log_prob.detach().squeeze().tolist())
                log_probs.append(log_prob.detach().squeeze())
                #if stop_token_index == next_token.item():
                #    break
                cpu_word = next_token.squeeze().cpu().numpy()
                cpu_word[ended] = pad_token_index
                ended = np.logical_or(ended, cpu_word == stop_token_index)
                if ended.all():
                    break

            for tokens in batch_tokens:
                output_list = list(tokens.squeeze().cpu().numpy())
                if stop_token_index in output_list:
                    eos_idx = output_list.index(stop_token_index)
                    output_list = output_list[:eos_idx]
                output_text = self.tokenizer.decode(output_list, skip_special_tokens=True)

                if output_text:
                    generated_list.append(output_text)
                else:
                    generated_list.append('')

        return generated_list, torch.stack(log_probs, 1)

    def decode_batch_sampling(self, action_embeds, feature, already_dropfeat=False, temperature=1.0, top_p=1.0):
        # higher temperature => more likely to sample low probability tokens
        with torch.no_grad():
            x = self.prepare_inputs(action_embeds, feature, already_dropfeat)

            stop_token_index = self.tokenizer.eos_token_id
            pad_token_index = self.tokenizer.pad_token_id
            generated_list = []

            batch_tokens = torch.tensor([[self.tokenizer.pad_token_id] for j in range(len(x))]).to(self.device)
            ended = np.zeros(len(x), np.bool)
            log_probs = []
            # print("initial batch_tokens size: ", batch_tokens.size())

            for i in range(self.max_instruction_length):
                outputs = self.transformer(inputs_embeds=x, decoder_input_ids=batch_tokens)
                logits = outputs.logits  # [batch size, seq length, vocab size]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # [batch size, vocab size]. last logit from sequence
                if top_p < 1.0:
                    logits = top_k_top_p_filtering(logits, top_p=top_p)

                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                #probs = logits.softmax(-1)
                #sampler = torch.distributions.categorical.Categorical(probs.flatten())
                next_token = m.sample()
                # print("next_token size: ", next_token.size())
                next_token = next_token.unsqueeze(0).to(self.device)
                # print("next_token size 2: ", next_token.size())

                # next_token = torch.argmax(logits, -1).unsqueeze(0).to(self.device)
                transposed_next_token = torch.transpose(next_token, 0, 1)
                # print("transposed next_token size: ", transposed_next_token.size())
                # print("batch_tokens size: ", batch_tokens.size())
                batch_tokens = torch.cat((batch_tokens, transposed_next_token), dim=1)
                # print("batch_tokens size: ", batch_tokens.size())
                # generated = torch.cat((generated, next_token_embed), dim=1)
                log_prob = m.log_prob(next_token)
                #log_probs.append(log_prob.detach().squeeze().tolist())
                log_probs.append(log_prob.detach().squeeze())
                #if stop_token_index == next_token.item():
                #    break
                cpu_word = next_token.squeeze().cpu().numpy()
                cpu_word[ended] = pad_token_index
                ended = np.logical_or(ended, cpu_word == stop_token_index)
                if ended.all():
                    break

            for tokens in batch_tokens:
                output_list = list(tokens.squeeze().cpu().numpy())
                if stop_token_index in output_list:
                    eos_idx = output_list.index(stop_token_index)
                    output_list = output_list[:eos_idx]
                output_text = self.tokenizer.decode(output_list, skip_special_tokens=True)

                if output_text:
                    generated_list.append(output_text)
                else:
                    generated_list.append('')

        return generated_list, torch.stack(log_probs, 1)


class SpeakerT5Attention(SpeakerT5):
    def __init__(self, feature_size, hidden_size, dropout_ratio, transformer_dropout_ratio, num_transformer_layers, bidirectional):
        super().__init__(feature_size, hidden_size, dropout_ratio, transformer_dropout_ratio, num_transformer_layers, bidirectional)
        self.action_transform = MLP([self.feature_size, self.hidden_size], device=self.device)

    def prepare_inputs(self, action_embeds, feature, already_dropfeat):
        x = action_embeds.to(self.device)
        feature = feature.to(self.device)
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        transformed_action = self.action_transform(x)
        encoder_outputs = self.encoder(inputs_embeds=transformed_action)
        ctx = encoder_outputs.last_hidden_state

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        # print("ctx size: ", ctx.size())
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature

        ctx = ctx.contiguous().view(-1, self.hidden_size)  # (batch * length, emb_size)
        entire_view_feature_seqs = feature.view(batch_size * max_length, -1, self.feature_size)  # (batch x length, # of images, feature_size)

        final_obs_feat, _ = self.attention_layer(ctx, entire_view_feature_seqs)
        final_obs_feat = final_obs_feat.view(batch_size, max_length, -1)
        final_obs_feat = self.drop(final_obs_feat)

        return final_obs_feat


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
