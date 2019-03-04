#Part of the code is adapted from OpenNMT (https://github.com/OpenNMT/OpenNMT-py) and pytorch language model example (https://github.com/pytorch/examples/tree/master/word_language_model)
import argparse
import math
import torch.nn as nn
import Constants
import Data
import Model
import Optim
import json
from utils import *
import torch.nn.functional as F
import sys, os
import time
from collections import defaultdict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default="../../data/train.pt",
                    help='location of the data file')
parser.add_argument('--RNN_type', type=str, default='GRU', help='')
parser.add_argument('--tied', type=int, default=1)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save', type=str,  default='model',
                    help='path to save the final model')
parser.add_argument('--learning_rate_decay', type=float, default=1.0,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('--start_decay_at', default=300,
                    help="Start decay after this epoch")
parser.add_argument('--embedding_file', default="../../embed/glove.6B.300d.txt",
                    help="Start decay after this epoch")
parser.add_argument('--pre_trained', default=None, type=str)
parser.add_argument('--test_only', default=0, type=int)
parser.add_argument('--other_word_cost', type=float, default=1)
parser.add_argument('--ori_cost', type=float, default=1)
parser.add_argument('--infor_softmax_temp', type=float, default=-1)
parser.add_argument('--infor_weighting', type=int, default=0)

args = parser.parse_args()
if args.other_word_cost > 0:
    # TODO employing multi-layer bidirectional RNN would have problem in the current approach of masking. See paper Appendix for the masking structure
    assert args.nlayers == 1
opt_values = vars(args)
param_print = ""
for item in vars(parser.parse_args()):
    if item != "gpu" and item != "random_seed" and opt_values[item] != parser.get_default(item) and item != "embedding_file" and item != "pre_trained":
        if item == "data":
            c_value = opt_values[item].replace("/", "#")
        else:
            c_value = str(opt_values[item])
        param_print += item + "_" + c_value + "-"
exp_path = os.path.dirname(os.path.realpath(__file__)) + "/../../obj/exp-%stime-%s/" % (param_print, time.strftime("%y.%m.%d_%H.%M.%S"))
if not args.test_only:
    os.mkdir(exp_path)
    args.save = exp_path + "model.pt"
    logger = Logger(exp_path + "log.txt")
    sys.stdout = logger
# print(args)
# args.cuda = args.gpu is not None
args.cuda = None
# if args.cuda:
#     torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

dataset = torch.load(args.data)
vocab = dataset["vocab"]
data = dataset["data"]
ntokens = vocab.size()
if not args.test_only:
    train_data = Data.BucketIterator(data['train'], args.batch_size, args.cuda, args, shuffle=True, infor_weighting=args.infor_weighting)
    valid_data = Data.BucketIterator(data['valid'], args.batch_size, args.cuda, args,  shuffle=False)
test_data = Data.BucketIterator(data['test']["whole"], args.batch_size, args.cuda, args, shuffle=False)
if args.pre_trained is not None or args.embedding_file == "None":
    embeddings = gen_embeddings(vocab, ntokens, args.emsize)
else:
    embeddings = gen_embeddings(vocab, ntokens, args.emsize, in_file=args.embedding_file)

###############################################################################
# Build the model
###############################################################################

model = Model.Model(args, ntokens, embeddings)
print(model)

if args.pre_trained:
    model.load_state_dict(torch.load(args.pre_trained))

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
print('* total number of parameters: %d' % nParams)
trainable_parameters = sum([p.nelement() for p in model.parameters()])
print('* total number of trainable parameters: %d' % trainable_parameters)


crit_weight = torch.ones(ntokens)
crit_weight[Constants.PAD] = 0
crit_weight[Constants.EOS] = 0
crit_weight[Constants.BOS] = 0
crit_weight[vocab.lookup("_")] = 0
inc_scale = 1
criterion = Model.InforWeightedCE(None, args.infor_softmax_temp)
aug_criterion = Model.InforWeightedCE(crit_weight, args.infor_softmax_temp)

if args.cuda:
    criterion = criterion.cuda()
    if aug_criterion is not None:
        aug_criterion = aug_criterion.cuda()


###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    model.eval()
    total_loss = 0
    n_samples = 0
    acc = 0
    total_correctness = []
    for i, (articles, options, answers, place_holder_pos, article_idx, predict_blank) in enumerate(data_source):
        output, _ = model(articles, options, place_holder_pos, article_idx)
        #total_loss += criterion(torch.max(output, Variable(torch.ones(output.size()).cuda()) * 1e-9).log(), answers, None).data * answers.size()[0]
        total_loss += criterion(torch.max(output, Variable(torch.ones(output.size())) * 1e-9).log(), answers,
                                None).data * answers.size()[0]
        n_samples += answers.size()[0]
        _, max_indexes = torch.kthvalue(output.cpu(), 4)
        correctness = torch.eq(max_indexes, answers.cpu()).long().data
        acc += correctness.sum()
        total_correctness += correctness.tolist()
    return total_loss[0] / n_samples, acc * 1.0 / n_samples, total_correctness

def test_all():
    # Run on test data.
    for test_subset in data['test']:
        test_data = Data.BucketIterator(data['test'][test_subset], args.batch_size, args.cuda, args, shuffle=False)
        print('=' * 89)
        test_loss, test_acc, total_correctness = evaluate(test_data)
        print('| subset %s | test loss %5.3f | test acc %8.3f' % (test_subset, test_loss, test_acc))
    print('=' * 89)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_samples = 0
    acc = 0
    ori_loss = 0

    for i, (articles, options, answers, place_holder_pos, article_idx, infor) in enumerate(train_data):
        model.zero_grad()
        output, aug_score = model(articles, options, place_holder_pos, article_idx)
        #loss = criterion(torch.max(output, Variable(torch.ones(output.size()).cuda()) * 1e-9).log(), answers, None)
        loss = criterion(torch.max(output, Variable(torch.ones(output.size())) * 1e-9).log(), answers, None)
        ori_loss += loss.data
        loss *= args.ori_cost
        if args.other_word_cost:
            if infor is not None:
                informativeness = infor[1:-1].t().contiguous()
            else:
                informativeness = None
            #aug_loss = aug_criterion(torch.max(aug_score, Variable(torch.ones(aug_score.size()).cuda()) * 1e-9).log(), articles[1:-1].t().contiguous().view(-1), informativeness, vocab)
            aug_loss = aug_criterion(torch.max(aug_score, Variable(torch.ones(aug_score.size())) * 1e-9).log(), articles[1:-1].t().contiguous().view(-1), informativeness, vocab)
            loss += args.other_word_cost * aug_loss
        loss.backward()
        optim.step()
        num_sample = answers.size()[0]
        total_loss += loss.data #* num_sample
        total_samples += num_sample
        output = F.softmax(output)
        max_score, max_indexes = torch.kthvalue(output.cpu(), 4)
        acc += torch.eq(max_indexes.cpu(), answers.cpu()).long().sum().item()

    cur_loss = total_loss[0] / total_samples
    cur_ori_loss = ori_loss[0] / total_samples
    print('|training epoch {:3d} | lr {:02.2f} | loss {:5.5f} | ori loss {:5.5f} | acc {:5.2f}'.format(
              epoch, optim.lr, cur_loss, cur_ori_loss,  acc * 1.0 / total_samples))

# Loop over epochs.
best_val_acc = None
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
    lr_decay=args.learning_rate_decay,
    start_decay_at=args.start_decay_at
)

if args.test_only:
    test_all()
    exit(0)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training')
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, valid_acc, total_correctness = evaluate(valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | '
                'valid acc {:5.3f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, valid_acc))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.

        if not best_val_acc or valid_acc > best_val_acc:
            print("best valid")
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_acc = valid_acc
        optim.updateLearningRate(-valid_acc, epoch)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
print("loading best model")
with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load(f))
test_all()
print(exp_path)
