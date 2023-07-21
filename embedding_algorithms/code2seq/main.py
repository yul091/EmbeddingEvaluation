import argparse

import pickle
import torch.optim as optim
import torch
import tqdm

from torch.utils.data import DataLoader

from embedding_algorithms.code2seq.dataset import C2SDataSet
from embedding_algorithms.code2seq.model import Code2Seq
#from se_tasks.code_summary.scripts.main import my_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", "-e", type=int, default=1,
                        help="Number of examples in epoch")
    parser.add_argument("--batchsize", "-b", type=int, default=128,
                        help="Number of examples in each mini-batch")
    parser.add_argument("--gpu", "-g", type=int, default=3,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--out", "-o", default="result",
                        help="Directory to output the result")
    parser.add_argument("--resume", "-r", default="",
                        help="Resume the training from snapshot")

    parser.add_argument("--datapath",
                        default='../../dataset/java-small-code2seq/tk.pkl',
                        help="path of input dataset")
    parser.add_argument("--trainpath",
                        default='../../dataset/java-small-code2seq/train.pkl',
                        help="path of train dataset")
    parser.add_argument("--validpath",
                        default='../../dataset/java-small-code2seq/test.pkl',
                        help="path of valid dataset")
    parser.add_argument("--savename", default="",
                        help="name of saved model")
    parser.add_argument("--max_size", type=int, default=None,
                        help="size of train dataset")
    parser.add_argument("--context_length", type=int, default=200,
                        help="length of context")
    parser.add_argument("--terminal_length", type=int, default=5,
                        help="length of terminal")
    parser.add_argument("--path_length", type=int, default=9,
                        help="length of path")
    parser.add_argument("--target_length", type=int, default=8,
                        help="length of target")
    parser.add_argument("--eval", action="store_true",
                        help="is eval")
    parser.add_argument("--path_rnn_drop", type=float, default=0.5,
                        help="drop rate of path rnn")
    parser.add_argument("--embed_drop", type=float, default=0.25,
                        help="drop rate of embbeding")
    parser.add_argument("--num_worker", type=int, default=0,
                        help="the number of worker")
    parser.add_argument("--decode_size", type=int, default=320,
                        help="decode size")
    parser.add_argument("--embed_dim", type=int, default=100)

    args = parser.parse_args()

    device = torch.device(
        args.gpu if args.gpu != -1 and torch.cuda.is_available() else "cpu")
    print(device)

    with open(args.datapath, "rb") as file:
        terminal_dict, path_dict, target_dict = pickle.load(file)
        print("Dictionaries loaded.")

    print("terminal_vocab:", len(terminal_dict))
    print("target_vocab:", len(target_dict))

    c2s = Code2Seq(args, terminal_vocab_size=len(terminal_dict),
                   path_element_vocab_size=len(path_dict),
                   target_dict=target_dict, device=device,
                   path_embed_size=args.embed_dim,
                   terminal_embed_size=args.embed_dim,
                   path_rnn_size=args.embed_dim * 2,
                   target_embed_size=args.embed_dim,
                   decode_size=args.decode_size)\
        .to(device)

    if args.resume != "":
        c2s.load_state_dict(torch.load(args.resume))

    train_path, test_path = args.trainpath, args.validpath
    val_dataset = \
        C2SDataSet(args, train_path, terminal_dict, path_dict,
                   target_dict, max_size=args.max_size, device=device)
    train_dataset = \
        C2SDataSet(args, test_path, terminal_dict, path_dict,
                   target_dict, max_size=args.max_size, device=device)
    batch_size = args.batchsize
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = optim.SGD(c2s.parameters(), lr=0.01, momentum=0.95)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.95, last_epoch=-1)

    for epoch in range(1, args.epoch+1):
        if not args.eval:
            sum_loss = 0
            train_count = 0
            c2s.train()
            scheduler.step()  # epochごとなのでここ
            for data in tqdm.tqdm(train_loader):
                loss = c2s(*data, is_eval=False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                if train_count % 250 == 0 and train_count != 0:
                    print(sum_loss / 250)
                    sum_loss = 0
                train_count += 1
        true_positive, false_positive, false_negative = 0, 0, 0
        for data in tqdm.tqdm(val_loader):
            c2s.eval()
            with torch.no_grad():
                true_positive_, false_positive_, false_negative_ = c2s(
                    *data, is_eval=True)
            true_positive += true_positive_
            false_positive += false_positive_
            false_negative += false_negative_

        pre_score, rec_score, f1_score = calculate_results(
            true_positive, false_positive, false_negative)
        print("f1:", f1_score, "prec:", pre_score,
              "rec:", rec_score)
        if args.eval:
            break
        if args.savename != "":
            torch.save(c2s.state_dict(), args.savename + str(epoch) + ".model")

        if (epoch % 2) == 0:
            w = c2s.terminal_element_embedding.weight.detach().cpu().numpy()
            save_dir = '../../embedding_vec/' + str(args.embed_dim) + '_' + str(epoch) + '/'
            save_file = save_dir + 'ori_code2seq.vec'
            torch.save([terminal_dict, w], save_file)


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


if __name__ == "__main__":
    main()
