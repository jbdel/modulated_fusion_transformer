import torch
import numpy as np

def compute_args(args):

    # DataLoader
    if args.dataset == "MOSEI": args.dataloader = 'Mosei_Dataset'
    if args.dataset == "MELD": args.dataloader = 'Meld_Dataset'
    if args.dataset == "MOSI": args.dataloader = 'Mosi_Dataset'
    if args.dataset == "IEMOCAP": args.dataloader = 'Iemocap_Dataset'
    if args.dataset == "VGAF": args.dataloader = 'Vgaf_Dataset'

    # Loss function to use
    if args.dataset == 'MOSEI' and args.task == 'sentiment': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").cuda()
    if args.dataset == 'MOSEI' and args.task == 'emotion': args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum").cuda()
    if args.dataset == 'MELD' and args.task == 'sentiment':  args.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.6]), reduction="sum").cuda()
    if args.dataset == 'MELD' and args.task == 'emotion': args.loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 4709/1205, 4709/268, 4709/683, 4709/1743, 4709/271, 4709/1109]),
        reduction="sum").cuda()

    if args.dataset == 'MOSI': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").cuda()
    if args.dataset == "IEMOCAP": args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").cuda()
    if args.dataset == "VGAF": args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").cuda()

    # Answer size
    if args.dataset == 'MOSEI' and args.task == "sentiment": args.ans_size = 7
    if args.dataset == 'MOSEI' and args.task == "sentiment" and args.task_binary: args.ans_size = 2
    if args.dataset == 'MOSEI' and args.task == "emotion": args.ans_size = 6
    if args.dataset == 'MELD' and args.task == "emotion": args.ans_size = 7
    if args.dataset == 'MELD' and args.task == "sentiment": args.ans_size = 3
    if args.dataset == 'MOSI': args.ans_size = 2
    if args.dataset == "IEMOCAP": args.ans_size = 4
    if args.dataset == "VGAF": args.ans_size = 3

    # Pred function
    if args.dataset == 'MOSEI': args.pred_func = "amax"
    if args.dataset == 'MOSEI' and args.task == "emotion": args.pred_func = "multi_label"
    if args.dataset == 'MELD': args.pred_func = "amax"
    if args.dataset == 'MOSI': args.pred_func = "amax"
    if args.dataset == 'IEMOCAP': args.pred_func = "amax"
    if args.dataset == "VGAF": args.pred_func = "amax"

    return args