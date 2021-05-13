import sys, os, shutil, math, random, time, subprocess, json
import time, datetime
import argparse, itertools

import vocabulary
from network import Network
from utils import out, format_elapsed
from const import START, STOP
from dataset import load_train_data, load_test_data

from collections import defaultdict

import paddle
import paddle.fluid as fluid
import numpy as np


def process_vocabulary(args, data, quiet=False):
    """
    Creates and returns vocabulary objects.
    Only iterates through the first 100 sequences, to save computation.
    """
    if not quiet:
        out(args.logfile, "initializing vacabularies... ", end="")
    seq_vocab = vocabulary.Vocabulary()
    bracket_vocab = vocabulary.Vocabulary()
    # loop_type_vocab = vocabulary.Vocabulary()

    for vocab in [seq_vocab, bracket_vocab]:  # , loop_type_vocab]:
        vocab.index(START)
        vocab.index(STOP)
    for x in data[:100]:
        seq = x["sequence"]
        dot = x["structure"]
        # loop = x["predicted_loop_type"]
        for character in seq:
            seq_vocab.index(character)
        for character in dot:
            bracket_vocab.index(character)
        # for character in loop:
        #    loop_type_vocab.index(character)
    for vocab in [seq_vocab, bracket_vocab]:  # , loop_type_vocab]:
        # vocab.index(UNK)
        vocab.freeze()
    if not quiet:
        out(args.logfile, "done.")

    def print_vocabulary(name, vocab):
        # special = {START, STOP, UNK}
        special = {START, STOP}
        out(args.logfile, "{}({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if not quiet:
        print_vocabulary("Sequence", seq_vocab)
        print_vocabulary("Brackets", bracket_vocab)
    return seq_vocab, bracket_vocab

def pad(x):
    if len(x)<500:
        return np.concatenate([x, np.zeros(500-len(x)).astype(np.int64)])
    else:
        return x
# def reader_creator(args, data,
#                    sequence_vocabulary, bracket_vocabulary,
#                    test=False):
#     def reader():
#         for i, x in enumerate(data):
#             seq = x["sequence"]
#             dot = x["structure"]
#             sequence = np.array([sequence_vocabulary.index(x) for x in list(seq)])
#             structure = np.array([bracket_vocabulary.index(x) for x in list(dot)])
#
#             # sequence = pad(sequence)
#             # structure = pad(structure)
#
#             if not test:
#                 LP_v_unpaired_prob = x["p_unpaired"]
#                 LP_v_unpaired_prob = np.array([x for x in LP_v_unpaired_prob])
#                 # LP_v_unpaired_prob = pad(LP_v_unpaired_prob -0.5) + 0.5
#
#                 yield sequence, structure, LP_v_unpaired_prob
#             else:
#                 yield sequence, structure
#
#     return reader

def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return positional_encoding

def reader_creator(args, data,
                   sequence_vocabulary, bracket_vocabulary,
                   test=False, reversed=False):
    def reader():
        for i, x in enumerate(data):
            seq = x["sequence"]
            dot = x["structure"]
            linear_rna = x['linear_rna']

            if not test:
                sequence = pad(np.array([sequence_vocabulary.index(x) for x in list(seq)]))
                structure = pad(np.array([bracket_vocabulary.index(x) for x in list(dot)]))
                linear_rna = pad(np.array(linear_rna)+2)-2
                LP_v_unpaired_prob = x["p_unpaired"]
                LP_v_unpaired_prob = pad(np.array([x for x in LP_v_unpaired_prob])+2)-2
                yield sequence, structure, linear_rna, LP_v_unpaired_prob
            else:
                sequence = np.array([sequence_vocabulary.index(x) for x in list(seq)])
                structure = np.array([bracket_vocabulary.index(x) for x in list(dot)])
                linear_rna = np.array(linear_rna)
                yield sequence, structure, linear_rna

    return reader
# def reader_creator(args, data,
#                    sequence_vocabulary, bracket_vocabulary,
#                    test=False, val=False):
#     def reader():
#         for i, x in enumerate(data):
#             seq = x["sequence"]
#             dot = x["structure"]
#             sequence = np.array([sequence_vocabulary.index(x) for x in list(seq)])
#             structure = np.array([bracket_vocabulary.index(x) for x in list(dot)])
#             if not test:
#                 LP_v_unpaired_prob = x["p_unpaired"]
#                 LP_v_unpaired_prob = np.array([x for x in LP_v_unpaired_prob])
#                 if not val:
#                     sequence = pad(sequence)
#                     structure = pad(structure)
#                     LP_v_unpaired_prob = pad(LP_v_unpaired_prob -0.5) + 0.5
#
#                 yield sequence, structure, LP_v_unpaired_prob
#             else:
#                 yield sequence, structure
#
#     return reader
'''
 sequence, structure, LP_v_unpaired_prob都是一维数组，长度为每个RNA序列的长度
 e.g.
     sequence: array(3,4,2,4,2,4,5,5)
     structure: array(2,3,4,4,2,3,4,2)
     LP_v_unpaired_prob: array(0.923,  ....)
'''


def Mse(y_true, y_pred):
    mask = paddle.fluid.layers.cast((y_true != -2), dtype=np.float32)
    mse = paddle.fluid.layers.square(y_true - y_pred)

    mse = paddle.fluid.layers.reduce_sum(mse * mask, dim=-1) / paddle.fluid.layers.reduce_sum(mask, dim=-1)
    mse = paddle.sqrt(mse)
    mse = paddle.fluid.layers.reduce_mean(mse)

    return mse

def run_train(args):
    if not os.path.exists(args.model_path_base):
        os.mkdir(args.model_path_base)
    # 打印训练开始时间，和命令行的参数
    # 2021-01-28 00:09:50.950989
    # # python3 src/main.py train --model-path-base model-0 --epochs 1 --checks-per-epoch 1

    out(args.logfile, datetime.datetime.now())
    out(args.logfile, "# python3 " + " ".join(sys.argv))

    # 在命令行 --logfile [数据路径]
    # 用load_train_data() 参数读取数据
    # 打印训练集和验证集的数据量
    log = args.logfile
    train_data, val_data = load_train_data()
    train_data_all = train_data + val_data

    out(log, "# Training set contains {} Sequences.".format(len(train_data)))
    out(log, "# Validation set contains {} Sequences.".format(len(val_data)))

    # 动态计算GPU卡号。从环境变量获取设备的ID，并指定给CUDAPlace
    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) if trainer_count > 1 else fluid.CUDAPlace(0)
    # 创建调试器
    exe = fluid.Executor(place)

    # 允许使用静态图，打印使用的计算设备
    # Paddle: Using device: CPUPlace
    # Initializing model...
    paddle.enable_static()
    out(log, "# Paddle: Using device: {}".format(place))
    out(log, "# Initializing model...")

    # 用process_vocabulary方法读取两个vocabulary对象
    seq_vocab, bracket_vocab = process_vocabulary(args, train_data)
    # 实例化Network对象， 传参
    network = Network(
        N_neuron=args.N_neuron,
        N_neuron2=args.N_neuron2,
        N_trans=args.N_trans,
        N_lstm=args.N_lstm,
        N_head=args.N_head
    )
    # 初始化程序用于初始化，一般只运行一次来初始化参数，主程序将会包含用来训练的网络结构和变量。在每个mini batch中运行并更新权重。
    # main_program = fluid.default_main_program()
    # startup_program = fluid.default_startup_program()

    # ?
    current_processed, total_processed = 0, 0
    check_every = math.floor((len(train_data) / args.checks_per_epoch))
    best_dev_loss, best_dev_model_path = [np.inf] * 20, [None] * 20

    start_time = time.time()
    out(log, "# Checking validation {} times an epoch (every {} batches)".format(args.checks_per_epoch, check_every))
    patience = check_every * args.checks_per_epoch * 2
    batches_since_dev_update = 0


    # seq: 中心词 dot: 目标词 y:
    seq = fluid.data(name="seq", shape=[None, None], dtype="int64")#, lod_level=1)
    dot = fluid.data(name="dot", shape=[None, None], dtype="int64")#, lod_level=1)
    linear_rna = fluid.data(name='linear_rna', shape=[None, None], dtype="float32")#, lod_level=1)
    y = fluid.data(name="label", shape=[None, None], dtype="float32")
    predictions = network.forward(seq, dot, linear_rna)

    avg_loss = Mse(y, predictions)
    # loss = fluid.layers.mse_loss(input=predictions, label=y)
    # loss = paddle.sqrt(loss)
    # avg_loss = fluid.layers.mean(loss)

    # test_program = main_program.clone(for_test=True)
    feeder = paddle.fluid.DataFeeder(place=place, feed_list=[seq, dot, linear_rna, y])

    # scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1e-3, factor=0.5, patience=2, verbose=True)
    # # learning_rate = 1e-3
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-08
    # optimizer = fluid.optimizer.Adam(
    #     learning_rate=scheduler,
    #     beta1=beta1,
    #     beta2=beta2,
    #     epsilon=epsilon,
    # )
    # optimizer.minimize(avg_loss)
    '''
    Executor 实际上将 Program 转化为C++后端可以执行的代码，以提高运行效率。
    '''

    # exe.run(startup_program)
    # exe_test = fluid.Executor(place)

    for i in range(20):
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1e-3, factor=0.5, patience=2, verbose=True)
        # learning_rate = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        optimizer = fluid.optimizer.Adam(
            learning_rate=scheduler,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        optimizer.minimize(avg_loss)
        main_program = fluid.default_main_program()
        startup_program = fluid.default_startup_program()
        exe.run(startup_program)
        exe_test = fluid.Executor(place)
        test_program = main_program.clone(for_test=True)

        train_reader = fluid.io.batch(
            fluid.io.shuffle(
                reader_creator(args, train_data_all[:i * 250] + train_data_all[(i+1)*250:], seq_vocab, bracket_vocab), buf_size=500),
            batch_size=args.batch_size)
        val_reader = fluid.io.batch(
            fluid.io.shuffle(
                reader_creator(args, train_data_all[i * 250: (i+1)*250], seq_vocab, bracket_vocab), buf_size=500),
            batch_size=1)

        start_epoch_index = 1
        for epoch in itertools.count(start=start_epoch_index):
            if epoch >= args.epochs + 1:
                break
            # train_reader 返回列表
            train_reader = fluid.io.batch(
                fluid.io.shuffle(
                    reader_creator(args, train_data, seq_vocab, bracket_vocab), buf_size=500),
                batch_size=args.batch_size)

            out(log, "# Epoch {} starting.".format(epoch))
            epoch_start_time = time.time()
            for batch_index, batch in enumerate(train_reader()):
                # for item in batch[0]:

                #     print(len(item))
                batch_loss, pred_values = exe.run(main_program, feed=feeder.feed(batch),
                                                fetch_list=[avg_loss.name, predictions.name],
                                                return_numpy=False)
                batch_loss = np.array(batch_loss)
                pred_values = np.array(pred_values)

                # scheduler.step(batch_loss)

                total_processed += len(batch)
                current_processed += len(batch)
                batches_since_dev_update += 1
                out(log,
                    "epoch {:,} "
                    "batch {:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {} "
                    "".format(
                        epoch,
                        batch_index + 1,
                        total_processed,
                        float(batch_loss),
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                    )
                    )
                if math.isnan(float(batch_loss[0])):
                    sys.exit("got NaN loss, training failed.")
                if current_processed >= check_every:
                    current_processed -= (check_every)

                    val_results = []
                    for data in val_reader():
                        loss, pred = exe.run(test_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_loss.name, predictions.name],
                                            return_numpy=False
                                            )
                        loss = np.array(loss)
                        val_results.append(loss[0])

                    val_loss = sum(val_results) / len(val_results)
                    out(log, "# Dev Average Loss: {:5.4f} (MSE) -> {:5.4f} (RMSD)".format(float(val_loss),
                                                                                        math.sqrt(float(val_loss))))
                    if val_loss < best_dev_loss[i]:
                        batches_since_dev_update = 0
                        if best_dev_model_path[i] is not None:
                            path = "{}/{}_dev={:.4f}_cv{}".format(args.model_path_base, args.model_path_base, best_dev_loss[i], i)

                            print("\t\t", best_dev_model_path[i], os.path.exists(path))
                            if os.path.exists(path):
                                out(log, "* Removing previous model file {}...".format(path))
                                shutil.rmtree(path)
                        best_dev_loss[i] = val_loss
                        best_dev_model_path[i] = "{}_dev={:.4f}_cv{}".format(args.model_path_base, val_loss, i)
                        out(log, "* Saving new best model to {}...".format(best_dev_model_path[i]))
                        if not os.path.exists(args.model_path_base):
                            os.mkdir(args.model_path_base)
                        fluid.io.save_inference_model(args.model_path_base + "/" + best_dev_model_path[i], ['seq', 'dot','linear_rna'],
                                                    [predictions], exe)
            scheduler.step(val_loss)


def run_test(args):
    log = args.logfile
    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id()) if trainer_count > 1 else fluid.CUDAPlace(0)
    print("Loading data...")
    train_data, val_data = load_train_data()
    test_data = load_test_data()

    print("Loading model...")
    seq_vocab, bracket_vocab = process_vocabulary(args, train_data, quiet=True)
    network = Network(
        # seq_vocab,
        # bracket_vocab,
        N_neuron=args.N_neuron,
        N_neuron2=args.N_neuron2,
        N_trans=args.N_trans,
        N_lstm=args.N_lstm,
        N_head=args.N_head


        # dropout=0,
    )

    exe = fluid.Executor(place)
    paddle.enable_static()
    fluid.io.load_inference_model(args.model_path_base, exe)
    test_reader = fluid.io.batch(
        reader_creator(args, test_data, seq_vocab, bracket_vocab, test=True),
        batch_size=1)

    seq = fluid.data(name="seq", shape=[None, None], dtype="int64")#, lod_level=1)
    dot = fluid.data(name="dot", shape=[None, None], dtype="int64")#, lod_level=1)
    linear_rna = fluid.data(name='linear_rna', shape=[None, None], dtype="float32")#, lod_level=1)
    predictions = network(seq, dot, linear_rna)

    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)
    test_feeder = fluid.DataFeeder(place=place, feed_list=[seq, dot, linear_rna])
    test_results = []
    for data in test_reader():
        pred, = exe.run(test_program,
                        feed=test_feeder.feed(data),
                        fetch_list=[predictions.name],
                        return_numpy=False
                        )
        # print('pred:', np.array(pred))
        pred = list(np.array(pred)[0])
        test_results.append(pred)

        out(log, " ".join([str(x) for x in pred]))


def main():
    if not os.path.exists('train_log'):
        os.mkdir('train_log')
    if not os.path.exists('test_log'):
        os.mkdir('test_log')
    ctime = time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime())
    train_log_path = os.path.join('train_log', "train_log {}.txt".format(ctime))
    test_log_path = os.path.join('test_log', "test_log {}.txt".format(ctime))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--logfile", default=train_log_path)
    subparser.add_argument("--batch-size", default=64)
    subparser.add_argument("--epochs", type=int, default=50)
    subparser.add_argument("--checks-per-epoch", type=float, default=1)
    subparser.add_argument("--N_neuron", type=int, default=128)
    subparser.add_argument("--N_neuron2", type=int, default=128)
    subparser.add_argument("--N_trans", type=int, default=1)
    subparser.add_argument("--N_lstm", type=int, default=2)
    subparser.add_argument("--N_head", type=int, default=2)

    subparser.add_argument("--dropout", type=float, default=0.15)

    # subparser = subparsers.add_parser("test_withlabel")
    # subparser.set_defaults(callback=run_test_withlabel)
    # subparser.add_argument("--model-path-base", required=False)
    # subparser.add_argument("--logfile", default="test_log.txt")
    # subparser.add_argument("--batch-size", default=1)
    # subparser.add_argument("--dmodel", type=int, default=128)
    # subparser.add_argument("--layers", type=int, default=8)
    # subparser.add_argument("--dropout", type=float, default=0.15)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=False)
    subparser.add_argument("--logfile", default=test_log_path)
    subparser.add_argument("--batch-size", default=64)
    subparser.add_argument("--N_neuron", type=int, default=128)
    subparser.add_argument("--N_neuron2", type=int, default=128)
    subparser.add_argument("--N_trans", type=int, default=1)
    subparser.add_argument("--N_lstm", type=int, default=2)
    subparser.add_argument("--N_head", type=int, default=2)
    subparser.add_argument("--dropout", type=float, default=0.15)

    args = parser.parse_args()
    args.logfile = open(args.logfile, "w")
    args.callback(args)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()