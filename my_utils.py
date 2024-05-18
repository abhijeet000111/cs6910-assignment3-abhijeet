

# !pip install wandb

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import wandb
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.switch_backend('agg')

API_KEY = 'e1e534a4439795735bad9f7d91519924c73589c7'
wandb.login(key=API_KEY)

DECODER_ATTENTION_MODE = True
HYPERPARAMETER_TUNING_MODE_USING_WANDB = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# pip install gdown

# import gdown
# # Google Drive file URL
# # 16yBIEtMjkb-8pmiHOk84YaLXpHrX25cW
# url = 'https://drive.google.com/uc?export=download&id=16yBIEtMjkb-8pmiHOk84YaLXpHrX25cW'
# # Path to save the downloaded file
# output_file = 'my_aksharantar_dataset_sampled.zip'
# # Download the file
# gdown.download(url, output_file, quiet=False)

# !unzip my_aksharantar_dataset_sampled.zip

TRAIN_PATH = 'aksharantar_sampled/hin/hin_train.csv'
VAL_PATH = 'aksharantar_sampled/hin/hin_valid.csv'
TEST_PATH = 'aksharantar_sampled/hin/hin_test.csv'

MAX_LENGTH = 30

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in list(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Function to read language data from CSV
def readLangs(lang1, lang2, data_path, reverse=False):
    '''
    function to read language data from CSV 
    '''
    print(f"Reading lines at {data_path}...")
    # Read the file and split into lines
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [l.split(',') for l in lines]
    # Reverse pairs if needed
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# Filter out pairs longer than MAX_LENGTH

def filterPair(p):
    '''
    function to filter out pairs longer than MAX_LENGTH
    '''
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    '''
    function to filter out pairs longer than MAX_LENGTH
    '''
    return [pair for pair in pairs if filterPair(pair)]

# Function to prepare data
def prepareData(lang1, lang2, data_path=TRAIN_PATH, reverse=False):
    '''
    function to prepare data
    '''
    input_lang, output_lang, pairs = readLangs(lang1, lang2, data_path, reverse)
    print(f"Read {len(pairs)} word pairs")
    print("Counting characters...")
    pairs = filterPairs(pairs)
    for pair in pairs:

        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print(f"Counted characters  --  {input_lang.name}: {input_lang.n_words} -- {output_lang.name}: {output_lang.n_words}")

    return input_lang, output_lang, pairs

# ENCODER RNN
class EncoderRNN(nn.Module):
    '''
    class to define the encoder RNN
    '''
    def __init__(self, input_size, hidden_size,  rnn_type='gru', num_layers=1, dropout_p=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_size, hidden_size)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("rnn_type not compatible")

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        return output, hidden

# DECODER
class DecoderRNN(nn.Module):
    '''
    class to define the decoder RNN
    '''
    def __init__(self, hidden_size, output_size, rnn_type='gru', num_layers = 1, dropout_p=0.1, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        D = 2 if bidirectional else 1
        self.out = nn.Linear(D * hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden #ADD DROUPOUT

class BahdanauAttention(nn.Module):
    '''
    class to define the Bahdanau Attention
    '''
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size) # B, d -> B, d
        self.Ua = nn.Linear(hidden_size, hidden_size) # B, d -> B, d
        self.Va = nn.Linear(hidden_size, 1)# B, d -> B, 1

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))) # 32, 1
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    '''
    class to define the Attention Decoder RNN
    '''
    def __init__(self, hidden_size, output_size, rnn_type = 'gru',  num_layers=1, dropout_p=0.1, bidirectional=False):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.rnn_type = rnn_type

        if rnn_type == 'gru':
            self.rnn = nn.GRU(2*hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(2*hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(2*hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        # D = 2 if bidirectional else 1
        D = 1
        self.out = nn.Linear(D * hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):

        # print(f'Attn_forward: encoder_hidden.shape: {encoder_hidden.shape}')
        # print(f'Attn_forward: encoder_outputs.shape: {encoder_outputs.shape}')
        # print(f'Attn_forward: target_tensor.shape: {target_tensor.shape}')

        batch_size = encoder_outputs.size(0)
        # print(f'Attn_forward: batch_size.shape : (batch_size.shape)')

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)

        # print(f'Attn_forward: decoder_input.shape : {decoder_input.shape}')

        decoder_hidden = encoder_hidden
        # print(f'Attn_forward: decoder_hidden.shape : {decoder_hidden.shape}')

        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):


        # print(f'Attn_forward_step: input.shape: {input.shape}')
        # print(f'Attn_forward_step: hidden.shape: {hidden.shape}')
        # print(f'Attn_forward_step: encoder_outputs.shape: {encoder_outputs.shape}')

        embedded =  self.dropout(self.embedding(input))

        # print(f'Attn_forward_step: embedded.shape: {embedded.shape}')
        if self.rnn_type == 'lstm':
            hidden = hidden[0]

        query = hidden.permute(1, 0, 2) # a, b, c -> b, a, c
        # print(f'Attn_forward_step: query.shape {query.shape}')

        context, attn_weights = self.attention(query, encoder_outputs)

        # print(f'Attn_forward_step: context.shape {context.shape}')
        # print(f'Attn_forward_step: attn_weights.shape {attn_weights.shape}')

        input_gru = torch.cat((embedded, context), dim=2)
        # print(f'Attn_forward_step: input_gru.shape {input_gru.shape}')

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

# class Seq2Seq(nn.Module):
#     def __init__(self, input_size, hidden_size,  rnn_type='gru', num_layers=1, dropout_p=0.1, bidirectional=False):
#         super(Seq2Seq, self).__init__()
#         self.encoer_gru = gru()
#         # self.encoder = EncoderRNN(input_size, hidden_size,  rnn_type='gru', num_layers=1, dropout_p=0.1, bidirectional=False)
#         # self.decoder = DecoderRNN(input_size, hidden_size,  rnn_type='gru', num_layers=1, dropout_p=0.1, bidirectional=False)

#     def encoder():
#         # Encoder forward
#         pass

#     def decoer():
#         # decoder forward
#         pass

#     def forward(self, input):
#         encoder_outputs, encoder_hidden = self.encoder(input)
#         return self.decoder(encoder_outputs, encoder_hidden, target_tensor)

def indexesFromSentence(lang, sentence):
    
    
    return [lang.word2index.get(word,2)  for word in list(sentence)]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(data_path, mode='train', batch_size=32, reverse=False, input_lang=None, output_lang=None, MAX_LENGTH=MAX_LENGTH):
    if mode == 'train':
        input_lang, output_lang, pairs = prepareData('eng', 'hin', data_path, reverse=reverse)
    else:
        _, _, pairs = prepareData('eng', 'hin', data_path, reverse=reverse)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)

        # Append EOS token
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)

        # Truncate if longer than MAX_LENGTH
        if len(inp_ids) > MAX_LENGTH:
            inp_ids = inp_ids[:MAX_LENGTH]
        if len(tgt_ids) > MAX_LENGTH:
            tgt_ids = tgt_ids[:MAX_LENGTH]

        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                         torch.LongTensor(target_ids).to(device))

    sampler = RandomSampler(data) if mode == 'train' else None
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return input_lang, output_lang, dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):    

    encoder.train()
    decoder.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    total_correct_words = 0
    total_words = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        if wandb.run:
            wandb.log({"train_batch_loss": loss.item()})

    #     # Calculate accuracy
    #     _, predicted = decoder_outputs.max(2)
    #     correct = predicted.eq(target_tensor)
    #     total_correct += correct.sum().item()
    #     total_samples += correct.numel()

    # accuracy = total_correct / total_samples
    # return total_loss / len(dataloader), accuracy

        # Calculate accuracy
        _, predicted = decoder_outputs.max(2)
        mask = target_tensor != PAD_token
        total_correct += (target_tensor == predicted)[mask].sum().item()
        total_tokens += mask.sum().item()
        total_correct_words += ((predicted != target_tensor).sum(1) == 0).sum().item()
        total_words += len(predicted)
        # _, predicted = decoder_outputs.max(2)
        # non_pad_mask_pred = predicted.ne(PAD_token)  # Mask for non-padded tokens in predicted sequences
        # non_pad_mask_tgt = target_tensor.ne(PAD_token)  # Mask for non-padded tokens in target sequences
        # correct_mask = predicted.eq(target_tensor) & non_pad_mask_pred & non_pad_mask_tgt
        # total_correct += correct_mask.sum().item()
        # total_tokens += non_pad_mask_tgt.sum().item()

    accuracy = total_correct / total_tokens
    total_loss = total_loss / len(dataloader)
    word_accuracy = total_correct_words / total_words
    if wandb.run:
        wandb.log({"train_token_accuracy": accuracy})
        wandb.log({"train_word_accuracy": word_accuracy})
        wandb.log({"train_loss": total_loss})
    return total_loss, word_accuracy

def val_epoch(dataloader, encoder, decoder, criterion, mode="val"):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    total_correct_words = 0
    total_words = 0

    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )


        total_loss += loss.item()



    #     # Calculate accuracy
    #     _, predicted = decoder_outputs.max(2)
    #     correct = predicted.eq(target_tensor)
    #     total_correct += correct.sum().item()
    #     total_samples += correct.numel()

    # accuracy = total_correct / total_samples
    # return total_loss / len(dataloader), accuracy
        # Calculate accuracy
        # Calculate accuracy
        _, predicted = decoder_outputs.max(2)
        mask = target_tensor != PAD_token
        total_correct += (target_tensor == predicted)[mask].sum().item()
        total_tokens += mask.sum().item()
        total_correct_words += ((predicted != target_tensor).sum(1) == 0).sum().item()
        total_words += len(predicted)

        # non_pad_mask_pred = predicted.ne(PAD_token)  # Mask for non-padded tokens in predicted sequences
        # non_pad_mask_tgt = target_tensor.ne(PAD_token)  # Mask for non-padded tokens in target sequences
        # correct_mask = predicted.eq(target_tensor) & non_pad_mask_pred & non_pad_mask_tgt
        # total_correct += correct_mask.sum().item()
        # total_tokens += non_pad_mask_tgt.sum().item()

    accuracy = total_correct / total_tokens
    total_loss = total_loss / len(dataloader)
    word_accuracy = total_correct_words / total_words
    if wandb.run:
        wandb.log({f"{mode}_token_accuracy": accuracy})
        wandb.log({f"{mode}_word_accuracy": word_accuracy})
        wandb.log({f"{mode}_loss": total_loss})
    return total_loss, word_accuracy



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, val_dataloader, test_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
# def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
#                print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss, accuracy = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        val_loss, val_accuracy = val_epoch(val_dataloader, encoder, decoder, criterion, mode='val')

        print(f'Epoch: {epoch}; train_loss: {loss}; train_accuracy: {accuracy}; val_loss: {val_loss}; val_accuracy: {val_accuracy}')

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    test_loss, test_accuracy = val_epoch(test_dataloader, encoder, decoder, criterion, mode='test')

    print(f'test_loss: {test_loss}; test_accuracy: {test_accuracy}')

    showPlot(plot_losses)

# Plotting results



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# Evaluation

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

# We can evaluate random sentences from the training set and print out the input, target, and output to make some subjective quality judgements:

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(pair)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')

# ########## HYPERPARAMETER TUNING USING WANDB ##########
# if HYPERPARAMETER_TUNING_MODE_USING_WANDB:
#     print('===========================================================================================================')
#     print("------------------------           HYPERPARAMETER_TUNING_MODE_USING_WANDB          ------------------------")
#     print('===========================================================================================================')

#     # Define a training function that takes hyperparameters as arguments
#     def train_with_wandb():
#         # Initialize WandB inside the function
#         wandb.init(project="CS6910_FODL_ASSIGNMENT_03_Van_Attn", entity="abhijeet001")

#         # Fetch hyperparameters from wandb.config
#         config = wandb.config
#         HIDDEN_SIZE = config.HIDDEN_SIZE
#         BATCH_SIZE = config.BATCH_SIZE
#         NUM_LAYERS_ENCODER = config.NUM_LAYERS_ENCODER
#         NUM_LAYERS_DECODER = config.NUM_LAYERS_DECODER
#         N_EPOCHS = config.N_EPOCHS
#         RNN_TYPE_ENCODER = config.RNN_TYPE_ENCODER
#         RNN_TYPE_DECODER = config.RNN_TYPE_DECODER
#         DROPOUT = config.DROPOUT
#         BIDIRECTIONAL = config.BIDIRECTIONAL
#         LEARNING_RATE = config.LEARNING_RATE
#         BEAM_SEARCH_SIZE = config.BEAM_SEARCH_SIZE
#         # DECODER_ATTENTION_MODE = config.DECODER_ATTENTION_MODE

#         # Construct run name using config parameters
#         run_name = f"batch_size={BATCH_SIZE}_hidden_size={HIDDEN_SIZE}_num_layers_encoder={NUM_LAYERS_ENCODER}_num_layers_decoder={NUM_LAYERS_DECODER}_n_epochs={N_EPOCHS}_rnn_type_encoder={RNN_TYPE_ENCODER}_dropout={DROPOUT}_bidirectional={BIDIRECTIONAL}_learning_rate={LEARNING_RATE}"
#         wandb.run.name = run_name

#         # Get dataloaders
#         input_lang, output_lang, train_dataloader = get_dataloader(data_path=TRAIN_PATH, mode='train', batch_size=BATCH_SIZE, reverse=False)
#         _, _, val_dataloader = get_dataloader(data_path=VAL_PATH, mode='val', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)
#         _, _, test_dataloader = get_dataloader(data_path=TEST_PATH, mode='test', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)

#         if DECODER_ATTENTION_MODE:

#             encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size = HIDDEN_SIZE, num_layers = 1, dropout_p=DROPOUT, bidirectional=False).to(device)

#             decoder = AttnDecoderRNN(hidden_size = HIDDEN_SIZE, output_size = output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers = 1, dropout_p=DROPOUT, bidirectional=False).to(device)
#         else:

#             encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)

#             decoder = DecoderRNN(hidden_size = HIDDEN_SIZE, output_size = output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers = NUM_LAYERS_ENCODER, dropout_p = DROPOUT, bidirectional=BIDIRECTIONAL).to(device)


#         if DECODER_ATTENTION_MODE:
#             print(f'-------------------------TRAINING : ATTENTION MODE------------------------------')
#         else:
#             print(f'-----------------------TRAINING : NON ATTENTION MODE----------------------------')
#         train(
#             train_dataloader,
#             val_dataloader,
#             test_dataloader,
#             encoder,
#             decoder,
#             n_epochs=N_EPOCHS,
#             learning_rate=LEARNING_RATE,
#             print_every=5,
#             plot_every=5
#         )

#     # Define your configuration space for hyperparameters
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'val_word_accuracy', 'goal': 'maximize'},
#         'parameters': {
#             'HIDDEN_SIZE': {'values': [64, 128, 256]},
#             'BATCH_SIZE': {'values': [128, 256, 512, 1024]},
#             'NUM_LAYERS_ENCODER': {'values': [1]},
#             'NUM_LAYERS_DECODER': {'values': [1]},
#             # 'N_EPOCHS': {'values': [15, 20, 25, 30, 35, 40, 45, 50]},
#             'N_EPOCHS': {'values': [1]},
#             'RNN_TYPE_ENCODER': {'values': ['gru', 'rnn', 'lstm']},
#             'RNN_TYPE_DECODER': {'values': ['gru', 'rnn', 'lstm']},
#             'DROPOUT': {'values': [0, 0.1, 0.2]},
#             'BIDIRECTIONAL': {'values': [False]},
#             'LEARNING_RATE': {'values': [0.01, 0.005, 0.002]},
#             'BEAM_SEARCH_SIZE': {'values': [1, 2, 3]},
#             # 'DECODER_ATTENTION_MODE': {'values': [True, False]}
#         }
#     }

#     # Start a sweep with the defined configuration
#     sweep_id = wandb.sweep(sweep_config)

#     # Run the sweep agent to execute the hyperparameter search
#     wandb.agent(sweep_id, function=train_with_wandb, count=100)

# DECODER_ATTENTION_MODE = False

# if DECODER_ATTENTION_MODE:
#     # Training with best hyperparameter for attention mode config:
#     HIDDEN_SIZE = 128
#     BATCH_SIZE = 1024
#     NUM_LAYERS_ENCODER = 1
#     NUM_LAYERS_DECODER = 1
#     N_EPOCHS = 35
#     RNN_TYPE_ENCODER = 'gru'
#     RNN_TYPE_DECODER = 'gru'
#     DROUPOUT = 0.1
#     BIDIRECTIONAL = False
#     BEAM_SEARCH_SIZE = 1
#     LEARNING_RATE = 0.005
# else:
#     # Training with best hyperparameter for non attention mode config:
#     HIDDEN_SIZE = 64
#     BATCH_SIZE = 128
#     NUM_LAYERS_ENCODER = 3
#     NUM_LAYERS_DECODER = 3
#     N_EPOCHS = 25
#     RNN_TYPE_ENCODER = 'lstm'
#     RNN_TYPE_DECODER = 'lstm'
#     DROUPOUT = 0.1
#     BIDIRECTIONAL = False
#     BEAM_SEARCH_SIZE = 1
#     LEARNING_RATE = 0.001


# # Get dataloaders
# input_lang, output_lang, train_dataloader = get_dataloader(data_path=TRAIN_PATH, mode='train', batch_size=BATCH_SIZE, reverse=False)
# _, _, val_dataloader = get_dataloader(data_path=VAL_PATH, mode='val', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)
# _, _, test_dataloader = get_dataloader(data_path=TEST_PATH, mode='test', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)


# if DECODER_ATTENTION_MODE:
#     NUM_LAYERS_ENCODER =1
#     NUM_LAYERS_DECODER = 1
#     encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS_ENCODER, dropout_p=DROUPOUT, bidirectional=BIDIRECTIONAL).to(device)
#     decoder = AttnDecoderRNN(hidden_size = HIDDEN_SIZE, output_size = output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers = NUM_LAYERS_ENCODER, dropout_p=DROUPOUT, bidirectional=BIDIRECTIONAL).to(device)
# else:
#     encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS_ENCODER, dropout_p=DROUPOUT, bidirectional=BIDIRECTIONAL).to(device)
#     decoder = DecoderRNN(hidden_size = HIDDEN_SIZE, output_size = output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers = NUM_LAYERS_ENCODER, dropout_p = DROUPOUT, bidirectional=BIDIRECTIONAL).to(device)

# with wandb.init(project="FODL_Assignment3", entity="abhijeet001"):
#     if DECODER_ATTENTION_MODE:
#         print(f'-------------------------TRAINING : ATTENTION MODE------------------------------')
#     else:
#         print(f'-----------------------TRAINING : NON ATTENTION MODE----------------------------')

#     train(
#         train_dataloader,
#         val_dataloader,
#         test_dataloader,
#         encoder,
#         decoder,
#         n_epochs=N_EPOCHS,
#         learning_rate=LEARNING_RATE,
#         print_every=5, plot_every=5
#     )

# import csv

def predict_and_save(input_lang, output_lang, test_dataloader, encoder, decoder, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['English Word', 'Hindi Word', 'Predicted Word'])

        for data in test_dataloader:
            input_tensor, target_tensor = data
            for i in range(len(input_tensor)):
                # Convert input_tensor to input_sentence without EOS
                input_sentence = ''.join([input_lang.index2word[idx.item()] for idx in input_tensor[i] if idx.item() != 0 and input_lang.index2word[idx.item()] != 'EOS'])
                # Convert target_tensor to target_sentence without EOS
                target_sentence = ''.join([output_lang.index2word[idx.item()] for idx in target_tensor[i] if idx.item() != 0 and output_lang.index2word[idx.item()] != 'EOS'])
                # Get the predicted words
                output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
                # Exclude <EOS> and <PAD> tokens from predicted words
                predicted_word = ''.join([word for word in output_words if word not in ('EOS', 'PAD', '<EOS>')])
                # Write the results to CSV file
                writer.writerow([input_sentence, target_sentence, predicted_word])

    print(f"Predictions saved to {output_file}")

# # Replace 'output_file_path.csv' with the path where you want to save the predictions
# if DECODER_ATTENTION_MODE:
#     output_file_path = f'vanilla_s2s_predictions.csv'
# else:
#     output_file_path = f'attention_s2s_predictions.csv'

# predict_and_save(input_lang, output_lang, test_dataloader, encoder, decoder, output_file_path)

# # Commented out IPython magic to ensure Python compatibility.
# if DECODER_ATTENTION_MODE:
#     # Visualizing Attention



def get_example_from_csv(csv_file, index):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        examples = list(reader)

    if index < len(examples):
        input_sentence, target_sentence = examples[index]
        return input_sentence, target_sentence
    else:
        print("Index out of range.")
        return None, None

# if DECODER_ATTENTION_MODE:
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker
#     from matplotlib.font_manager import FontProperties
#     import time

#     TTF_PATH = '/mnt/media/guest1/abhijeet/assignments/deep_learning/assignment_03/attention_seq_to_seq/aksharantar_sampled/hin/Nirmala.ttf'
#     font_prop = FontProperties(fname=TTF_PATH, size=12)

#     def showAttention(input_sentence, target_sentence, output_words, attentions):
#         fig = plt.figure(figsize=(3, 3))
#         ax = fig.add_subplot(111)
#         cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
#         fig.colorbar(cax)

#         # Set up axes
#         ax.set_xticklabels([''] + list(target_sentence) + ['<EOS>'], rotation=90, fontproperties=font_prop)
#         ax.set_yticklabels([''] + output_words, fontproperties=font_prop)

#         # Show label at every tick
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#         plt.show()

#     def evaluateAndShowAttention(input_sentence, target_sentence, encoder, decoder, input_lang, output_lang):
#         output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
#         print('Input:', input_sentence)
#         print('Target:', target_sentence)
#         print('Predicted Output:', ''.join(output_words))
#         showAttention(input_sentence, target_sentence, output_words, attentions[0, :len(output_words), :])

#     # Get 10 samples from the training set and visualize attention for each
#     for i in range(10):
#         print(f"Processing example {i+1}")
#         input_sentence, target_sentence = get_example_from_csv(TRAIN_PATH, i)  # Implement this function to get examples from your dataset
#         evaluateAndShowAttention(input_sentence, target_sentence, encoder, decoder, input_lang, output_lang)
#         time.sleep(3)

# if DECODER_ATTENTION_MODE:
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker
#     from matplotlib.font_manager import FontProperties
#     import csv

#     TTF_PATH = '/mnt/media/guest1/abhijeet/assignments/deep_learning/assignment_03/attention_seq_to_seq/aksharantar_sampled/hin/Nirmala.ttf'
#     font_prop = FontProperties(fname=TTF_PATH, size=12)

#     def showAttention(input_sentence, target_sentence, output_words, attentions, ax):
#         cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
#         fig.colorbar(cax, ax=ax)

#         # Set up axes
#         ax.set_xticklabels([''] + list(target_sentence) + ['<EOS>'], rotation=90, fontproperties=font_prop)
#         ax.set_yticklabels([''] + output_words, fontproperties=font_prop)

#         # Show label at every tick
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     def evaluateAndShowAttention(input_sentence, target_sentence, encoder, decoder, input_lang, output_lang, ax):
#         output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
#         print('Input:', input_sentence)
#         print('Target:', target_sentence)
#         print('Predicted Output:', ''.join(output_words))
#         showAttention(input_sentence, target_sentence, output_words, attentions[0, :len(output_words), :], ax)

#     # Load examples from CSV
#     examples = []
#     with open(TEST_PATH, 'r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         examples = list(reader)

#     # Create subplots
#     fig, axs = plt.subplots(4, 3, figsize=(15, 15))

#     # Get 10 samples and visualize attention for each
#     for i, ax in enumerate(axs.flat):
#         if i < len(examples):
#             input_sentence, target_sentence = examples[i]
#             evaluateAndShowAttention(input_sentence, target_sentence, encoder, decoder, input_lang, output_lang, ax)
#         else:
#             # If there are fewer than 10 examples, leave the subplot empty
#             ax.axis('off')

#     plt.tight_layout()
#     plt.show()

# if DECODER_ATTENTION_MODE:
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker
#     from matplotlib.font_manager import FontProperties
#     import csv

#     TTF_PATH = '/mnt/media/guest1/abhijeet/assignments/deep_learning/assignment_03/attention_seq_to_seq/aksharantar_sampled/hin/Nirmala.ttf'
#     font_prop = FontProperties(fname=TTF_PATH, size=12)

#     def showAttention(input_sentence, target_sentence, output_words, attentions, ax):
#         cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
#         fig.colorbar(cax, ax=ax)

#         # Set up axes
#         ax.set_xticklabels([''] + list(target_sentence) + ['<EOS>'], rotation=90, fontproperties=font_prop)
#         ax.set_yticklabels([''] + output_words, fontproperties=font_prop)

#         # Show label at every tick
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     def evaluateAndShowAttention(input_sentence, target_sentence, encoder, decoder, input_lang, output_lang, ax):
#         output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
#         print('Input:', input_sentence)
#         print('Target:', target_sentence)
#         print('Predicted Output:', ''.join(output_words))
#         showAttention(input_sentence, target_sentence, output_words, attentions[0, :len(output_words), :], ax)

#     # Load examples from CSV
#     examples = []
#     with open(TEST_PATH, 'r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         examples = list(reader)

#     # Create subplots
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))

#     # Get 10 samples and visualize attention for each
#     for i, ax in enumerate(axs.flat):
#         if i < len(examples):
#             input_sentence, target_sentence = examples[i]
#             ax.set_title(f"input: {input_sentence}\n output: {target_sentence}", fontproperties=font_prop)
#             evaluateAndShowAttention(input_sentence, target_sentence, encoder, decoder, input_lang, output_lang, ax)
#         else:
#             # If there are fewer than 10 examples, leave the subplot empty
#             ax.axis('off')

#     plt.tight_layout()
#     plt.show()









