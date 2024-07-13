import sys
sys.path.append("../common")
from training_utils import get_test_trainer, get_trainer, get_test_model, get_model, init_wandb
from transformers import set_seed
from ScriptUtils import parse_training_args, log_test_results
from data import FinetuningData, PretrainedVectorsData
import scipy
import numpy as np


def get_data(args):
    if args.no_finetuning:
        data_path = f"../data/adult_age/vectors_extracted_from_trained_models/{args.model}/pretrained/vectors_{args.model}_128.pt"
        data_train = PretrainedVectorsData(data_path, args.seed, split="train",
                                           balanced=args.balanced)
        data_valid = PretrainedVectorsData(data_path, args.seed, split="valid",
                                           balanced=args.balanced)
    else:
        data_path = f"../data/adult_age/tokens_{args.model}_128.pt"

        data_train = FinetuningData(data_path, args.seed, "train", args.balanced)
        data_valid = FinetuningData(data_path, args.seed, "valid", args.balanced)

    return data_train, data_valid

def __main__():
    args = parse_training_args()
    init_wandb(args, "Adult", "Adult_age train")
    set_seed(args.seed)
    data_train, data_valid = get_data(args)

    checkpoint_folder = "../checkpoints/adult_age"

    range_grads = [-0.09, -0.07, -0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09]
    # range_grads = [0.09]
    grads = np.array(range_grads)
    acc = np.zeros(len(range_grads))
    independence = np.zeros(len(range_grads))
    separation = np.zeros(len(range_grads))
    sufficiency = np.zeros(len(range_grads))


    for g in range(len(range_grads)):
        grad = range_grads[g]
        model, checkpoint_folder = get_test_model(args.no_finetuning, data_train.n_labels, checkpoint_folder, args.model,
                                             args.seed, args.data, args.balanced, args.dfl_gamma, args.temp,
                                             args.no_group_labels, args.control, grad, groups=['<25', '>60', '25-60'])

        trainer = get_test_trainer(args, model, ['<25', '>60', '25-60'], metric='acc')
        trainer.fit(data_train, data_valid, 4, checkpoint_folder, checkpoint_every=args.checkpointevery,
                    print_every=args.printevery)

        # validation
        trainer.load_checkpoint(f"{checkpoint_folder}/best_model.pt")
        res = trainer.evaluate(data_valid, "valid")
        # log_test_results(res)

        acc[g] = res['acc']
        independence[g] = res['independence_sum']
        separation[g] = res['separation_gap-abs_sum']
        sufficiency[g] = res['sufficiency_gap-abs_sum']

        checkpoint_folder = "../checkpoints/adult_age"

    for k in range(len(range_grads)):
        print(f"acc:{acc[k]}, independence:{independence[k]}, separation:{separation[k]}, sufficiency:{sufficiency[k]}")
    filt = acc >= 0.0
    corr1, p1 = scipy.stats.spearmanr(grads[filt], independence[filt])
    corr2, p2 = scipy.stats.spearmanr(grads[filt], separation[filt])
    corr3, p3 = scipy.stats.spearmanr(grads[filt], sufficiency[filt])
    print(f"independence:corr={corr1}, p={p1}")
    print(f"separation:corr={corr2}, p={p2}")
    print(f"sufficiency:corr={corr3}, p={p3}")
    w = abs(corr2)
    # w = 0.5
    print(f"w={w}")


    model, checkpoint_folder = get_model(args.no_finetuning, data_train.n_labels, checkpoint_folder, args.model,
                                         args.seed, args.data, args.balanced, args.dfl_gamma, args.temp,
                                         args.no_group_labels, args.control)

    trainer = get_trainer(args, model, ['<25', '>60', '25-60'], metric='acc', w=w)
    trainer.fit(data_train, data_valid, args.epochs, checkpoint_folder, checkpoint_every=args.checkpointevery,
                print_every=args.printevery)

    # validation
    trainer.load_checkpoint(f"{checkpoint_folder}/best_model.pt")
    res = trainer.evaluate(data_valid, "valid")
    log_test_results(res)

if __name__ == "__main__":
    __main__()
