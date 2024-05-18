
print("+++++++++++++++++++++++++ RUNNING train.py +++++++++++++++++++++++++++++++++")

from my_utils import *
import argparse
import torch
import wandb 
import csv

TRAIN_PATH = 'aksharantar_sampled/hin/hin_train.csv'
VAL_PATH = 'aksharantar_sampled/hin/hin_valid.csv'
TEST_PATH = 'aksharantar_sampled/hin/hin_test.csv'

# Define the argparse function to parse command-line arguments
def parse_args():
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser(description="Sequence-to-Sequence Model for English to Hindi Translation")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer in the RNN")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_layers_encoder", type=int, default=1, help="Number of layers in the encoder RNN")
    parser.add_argument("--num_layers_decoder", type=int, default=1, help="Number of layers in the decoder RNN")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs for training")
    parser.add_argument("--rnn_type_encoder", type=str, default="gru", choices=["gru", "rnn", "lstm"], help="Type of RNN for the encoder")
    parser.add_argument("--rnn_type_decoder", type=str, default="gru", choices=["gru", "rnn", "lstm"], help="Type of RNN for the decoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional RNN layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--beam_search_size", type=int, default=1, help="Size of beam search")
    parser.add_argument("--decoder_attention_mode", action="store_true", default=False, help="Use attention mechanism in the decoder")
    parser.add_argument("--hyperparameter_tuning", action="store_true", default=False, help="Enable hyperparameter tuning mode using W&B")
    return parser.parse_args()
    

args = parse_args()
print('++++++++++++++++++++++++ args parsed ++++++++++++++++++++++++++++++++++++++')

DECODER_ATTENTION_MODE = args.decoder_attention_mode
# Define a training function that takes hyperparameters as arguments
def train_with_wandb():
    
    wandb.init(project="CS6910_FODL_ASSIGNMENT_03_Van_Attn", entity="abhijeet001")
    config = wandb.config

    HIDDEN_SIZE = config.HIDDEN_SIZE
    BATCH_SIZE = config.BATCH_SIZE
    NUM_LAYERS_ENCODER = config.NUM_LAYERS_ENCODER
    NUM_LAYERS_DECODER = config.NUM_LAYERS_DECODER
    N_EPOCHS = config.N_EPOCHS
    RNN_TYPE_ENCODER = config.RNN_TYPE_ENCODER
    RNN_TYPE_DECODER = config.RNN_TYPE_DECODER
    DROPOUT = config.DROPOUT
    BIDIRECTIONAL = config.BIDIRECTIONAL
    LEARNING_RATE = config.LEARNING_RATE
    BEAM_SEARCH_SIZE = config.BEAM_SEARCH_SIZE

    run_name = f"batch_size={BATCH_SIZE}_hidden_size={HIDDEN_SIZE}_num_layers_encoder={NUM_LAYERS_ENCODER}_num_layers_decoder={NUM_LAYERS_DECODER}_n_epochs={N_EPOCHS}_rnn_type_encoder={RNN_TYPE_ENCODER}_dropout={DROPOUT}_bidirectional={BIDIRECTIONAL}_learning_rate={LEARNING_RATE}"
    wandb.run.name = run_name

    input_lang, output_lang, train_dataloader = get_dataloader(data_path=TRAIN_PATH, mode='train', batch_size=BATCH_SIZE, reverse=False)
    _, _, val_dataloader = get_dataloader(data_path=VAL_PATH, mode='val', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)
    _, _, test_dataloader = get_dataloader(data_path=TEST_PATH, mode='test', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)

    if config.DECODER_ATTENTION_MODE:
        encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size=HIDDEN_SIZE, num_layers=1, dropout_p=DROPOUT, bidirectional=False).to(device)
        decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers=1, dropout_p=DROPOUT, bidirectional=False).to(device)
    else:
        encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
        decoder = DecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers=NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)

    train(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        encoder,
        decoder,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        print_every=5,
        plot_every=5
    )

def predict_and_save(input_lang, output_lang, test_dataloader, encoder, decoder, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['English Word', 'Hindi Word', 'Predicted Word'])

        for data in test_dataloader:
            input_tensor, target_tensor = data
            for i in range(len(input_tensor)):
                input_sentence = ''.join([input_lang.index2word[idx.item()] for idx in input_tensor[i] if idx.item() != 0 and input_lang.index2word[idx.item()] != 'EOS'])
                target_sentence = ''.join([output_lang.index2word[idx.item()] for idx in target_tensor[i] if idx.item() != 0 and output_lang.index2word[idx.item()] != 'EOS'])
                output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
                predicted_word = ''.join([word for word in output_words if word not in ('EOS', 'PAD', '<EOS>')])
                writer.writerow([input_sentence, target_sentence, predicted_word])
    
    print(f"Predictions saved to {output_file}")



if args.hyperparameter_tuning:
    print('===========================================================================================================')
    print("------------------------           HYPERPARAMETER_TUNING_MODE_USING_WANDB          ------------------------")
    print('===========================================================================================================')

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_word_accuracy', 'goal': 'maximize'},
        'parameters': {
            'HIDDEN_SIZE': {'values': [64, 128, 256]},
            'BATCH_SIZE': {'values': [128, 256, 512, 1024]},
            'NUM_LAYERS_ENCODER': {'values': [1]},
            'NUM_LAYERS_DECODER': {'values': [1]},
            'N_EPOCHS': {'values': [1]},
            'RNN_TYPE_ENCODER': {'values': ['gru', 'rnn', 'lstm']},
            'RNN_TYPE_DECODER': {'values': ['gru', 'rnn', 'lstm']},
            'DROPOUT': {'values': [0, 0.1, 0.2]},
            'BIDIRECTIONAL': {'values': [False]},
            'LEARNING_RATE': {'values': [0.01, 0.005, 0.002]},
            'BEAM_SEARCH_SIZE': {'values': [1, 2, 3]},
            'DECODER_ATTENTION_MODE': {'values': [True, False]}
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train_with_wandb, count=100)
else:
    HIDDEN_SIZE = args.hidden_size
    BATCH_SIZE = args.batch_size
    NUM_LAYERS_ENCODER = args.num_layers_encoder
    NUM_LAYERS_DECODER = args.num_layers_decoder
    N_EPOCHS = args.n_epochs
    RNN_TYPE_ENCODER = args.rnn_type_encoder
    RNN_TYPE_DECODER = args.rnn_type_decoder
    DROPOUT = args.dropout
    BIDIRECTIONAL = args.bidirectional
    LEARNING_RATE = args.learning_rate
    BEAM_SEARCH_SIZE = args.beam_search_size
    DECODER_ATTENTION_MODE = args.decoder_attention_mode

    input_lang, output_lang, train_dataloader = get_dataloader(data_path=TRAIN_PATH, mode='train', batch_size=BATCH_SIZE, reverse=False)
    _, _, val_dataloader = get_dataloader(data_path=VAL_PATH, mode='val', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)
    _, _, test_dataloader = get_dataloader(data_path=TEST_PATH, mode='test', batch_size=BATCH_SIZE, reverse=False, input_lang=input_lang, output_lang=output_lang)

    if DECODER_ATTENTION_MODE:
        NUM_LAYERS_ENCODER = 1
        NUM_LAYERS_DECODER = 1
        encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
        decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers=NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
    else:
        encoder = EncoderRNN(input_lang.n_words, rnn_type=RNN_TYPE_ENCODER, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
        decoder = DecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_lang.n_words, rnn_type=RNN_TYPE_ENCODER, num_layers=NUM_LAYERS_ENCODER, dropout_p=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)

    with wandb.init(project="FODL_Assignment3", entity="abhijeet001"):
        if DECODER_ATTENTION_MODE:
            print(f'-------------------------TRAINING : ATTENTION MODE------------------------------')
        else:
            print(f'-----------------------TRAINING : NON ATTENTION MODE----------------------------')
        
        train(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            encoder,
            decoder,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            print_every=5,
            plot_every=5
        )

# # Evaluation and prediction saving
# output_file_path = 'attention_s2s_predictions.csv' if DECODER_ATTENTION_MODE else 'vanilla_s2s_predictions.csv'
# predict_and_save(input_lang, output_lang, test_dataloader, encoder, decoder, output_file_path)
