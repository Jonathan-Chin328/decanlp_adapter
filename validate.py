import torch
import json
import os
from pathlib import Path

from util import pad
from metrics import compute_metrics

heatmap = {
    'squad': 0,
    'iwslt': 1,
    'cnn_dailymail': 2,
    'multinli': 3,
    'sst': 4,
    'srl': 5,
    'zre': 6,
    'woz': 7,
    'wikisql': 8,
    'schema': 9
}

def compute_validation_outputs(model, val_iter, field, task, optional_names=[], args=None):
    print('start validate')
    loss, predictions, answers = [], [], []
    outputs = [[] for _ in range(len(optional_names))]
    num_batch = 0
    task_id_distribution = 0
    for batch_idx, batch in enumerate(val_iter):
        l, p, task_id = model(batch, task)
        # save distribution
        if args.adapter_classification is not None:
            task_id_distribution += task_id
            num_batch += 1
        #############################
        loss.append(l)
        predictions.append(pad(p, 150, dim=-1, val=field.vocab.stoi['<pad>']))
        a = None
        if hasattr(batch, 'wikisql_id'):
            a = batch.wikisql_id.data.cpu()
        elif hasattr(batch, 'squad_id'):
            a = batch.squad_id.data.cpu()
        elif hasattr(batch, 'woz_id'):
            a = batch.woz_id.data.cpu()
        else:
            a =  pad(batch.answer.data.cpu(), 150, dim=-1, val=field.vocab.stoi['<pad>'])
        answers.append(a)
        for opt_idx, optional_name in enumerate(optional_names):
            outputs[opt_idx].append(getattr(batch, optional_name).data.cpu()) 
    if args.adapter_classification is not None:
        save_adapter_id_distribution(task, task_id_distribution / num_batch, args)
    loss = torch.cat(loss, 0) if loss[0] is not None else None
    predictions = torch.cat(predictions, 0)
    answers = torch.cat(answers, 0)
    return loss, predictions, answers, [torch.cat([pad(x, 150, dim=-1, val=field.vocab.stoi['<pad>']) for x in output], 0) for output in outputs]


def get_clip(val_iter):
    return -val_iter.extra if val_iter.extra > 0 else None


def all_reverse(tensor, world_size, field, clip, dim=0):
    if world_size > 1:
        tensor = tensor.float() # tensors must be on cpu and float for all_gather
        all_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.barrier() # all_gather is experimental for gloo, found that these barriers were necessary
        torch.distributed.all_gather(all_tensors, tensor)
        torch.distributed.barrier()
        tensor = torch.cat(all_tensors, 0).long() # tensors must be long for reverse
    # for distributed training, dev sets are padded with extra examples so that the
    # tensors are all of a predictable size for all_gather. This line removes those extra examples
    return field.reverse(tensor)[:clip] 

def save_adapter_id_distribution(task, task_id_distribution, args):
    # print(task, task_id_distribution)
    json_path = os.path.join(args.log_dir, 'heatmap.json')
    Path(json_path).touch()
    with open(json_path, 'r') as file:
        text = file.read()
        try:
            heatmap = json.loads(text)
        except:
            heatmap = ""

    with open(json_path, 'w') as file:
        if type(heatmap) is dict:
            heatmap[task.split('.')[0]] = task_id_distribution.tolist()
        else:
            heatmap = {
                'squad': 0,
                'iwslt': 1,
                'cnn_dailymail': 2,
                'multinli': 3,
                'sst': 4,
                'srl': 5,
                'zre': 6,
                'woz': 7,
                'wikisql': 8,
                'schema': 9
            }
        file.write(json.dumps(heatmap))



def gather_results(model, val_iter, field, world_size, task, optional_names=[], args=None):
    loss, predictions, answers, outputs = compute_validation_outputs(model, val_iter, field, task, optional_names=optional_names, args=args)
    clip = get_clip(val_iter)
    if not hasattr(val_iter.dataset.examples[0], 'squad_id') and not hasattr(val_iter.dataset.examples[0], 'wikisql_id') and not hasattr(val_iter.dataset.examples[0], 'woz_id'):
        answers = all_reverse(answers, world_size, field, clip)
    return loss, all_reverse(predictions, world_size, field, clip), answers, [all_reverse(x, world_size, field, clip) for x in outputs],


def print_results(keys, values, rank=None, num_print=1):
    start = rank * num_print if rank is not None else 0
    end = start + num_print
    values = [val[start:end] for val in values]
    for ex_idx in range(len(values[0])):
        for key_idx, key in enumerate(keys):
            value = values[key_idx][ex_idx]
            v = value[0] if isinstance(value, list) else value
            print(f'{key}: {repr(v)}')


def validate(task, val_iter, model, logger, field, world_size, rank, num_print=10, args=None):
    with torch.no_grad():
        model.eval()
        required_names = ['greedy', 'answer']
        optional_names = ['context', 'question']
        loss, predictions, answers, results = gather_results(model, val_iter, field, world_size, task, optional_names=optional_names, args=args)
        predictions = [p.replace('UNK', 'OOV') for p in predictions]
        names = required_names + optional_names 
        if hasattr(val_iter.dataset.examples[0], 'wikisql_id') or hasattr(val_iter.dataset.examples[0], 'squad_id') or hasattr(val_iter.dataset.examples[0], 'woz_id'):
            answers = [val_iter.dataset.all_answers[sid] for sid in answers.tolist()]
        metrics, answers = compute_metrics(predictions, answers, bleu='iwslt' in task or 'multi30k' in task, dialogue='woz' in task,
            rouge='cnn' in task, logical_form='sql' in task, corpus_f1='zre' in task, args=args)
        results = [predictions, answers] + results
        print_results(names, results, rank=rank, num_print=num_print)

        return loss, metrics
