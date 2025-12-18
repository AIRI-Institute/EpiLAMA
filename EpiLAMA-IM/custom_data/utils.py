from itertools import groupby
from collections import defaultdict
import numpy as np
from typing import List, Tuple
import torch
from tqdm.auto import tqdm
from model import LSTMCNNCRF, CRF
from transformers import AutoTokenizer, AutoModel


def parse_fasta(fastafile: str):
    '''
    Parses fasta file into lists of identifiers and sequences.
    Can handle multi-line sequences and empty lines.
    Appends numeration if there are duplicate identifiers.
    '''
    ids = []
    seqs = []
    with open(fastafile, 'r') as f:

        id_seq_groups = (group for group in groupby(f, lambda line: line.startswith(">")))

        for is_id, id_iter in id_seq_groups:
            if is_id: # Only needed to find first id line, always True thereafter
                ids.append(next(id_iter).strip())
                seqs.append("".join(seq.strip() for seq in next(id_seq_groups)[1]))

    #truncate and un-duplicate identifiers
    #ids = [x[:80] for x in ids] #no more truncation.
    already_seen = defaultdict(int)
    outnames = []
    for i in ids:
        #replace special characters
        #i = i.replace(' ','_')
        already_seen[i] += 1
        if already_seen[i]>1:
            outnames.append(i + '_' + str(already_seen[i]))
        else:
            outnames.append(i)
    ids = outnames
    #remove whitespace in fasta
    seqs = [x.replace(' ', '') for x in seqs]

    return ids, seqs

class ESMEmbedder:
    """
    Дроп-ин замена для fair-esm.
    Использует модели:
      - esm='esm2'  -> 'facebook/esm2_t33_650M_UR50D' (repr layer = 33)
      - esm='esm1b' -> 'facebook/esm1b_t33_650M_UR50S' (repr layer = 32)
    Чанкование идентично исходному коду на fair-esm.
    """
    def __init__(self, esm: str = 'esm2', local_esm_path: str = None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if local_esm_path is not None:
            model_id = local_esm_path
        else:
            if esm == 'esm2':
                model_id = 'facebook/esm2_t33_650M_UR50D'
            elif esm == 'esm1b':
                model_id = 'facebook/esm1b_t33_650M_UR50S'
            else:
                raise NotImplementedError(esm)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        # соответствие слоям как в исходнике:
        self.return_layer = 33 if esm == 'esm2' else 32

        # лимит длинны у ESM ~1024 токена с учётом спец-токенов; в старом коде резали по 1022
        self._CHUNK = 1022
        self._STEP_BACK = 300  # укорочение всех неконечных окон на 300

    def __call__(self, sequences, repr_layers: bool = False, progress_bar: bool = False):
        """
        Возвращает список тензоров [seq_len, hidden_dim], как и раньше.
        """
        from tqdm.auto import tqdm as _tqdm
        iterator = _tqdm(sequences, desc='Embedding...', leave=False) if progress_bar else sequences

        embeddings = []
        with torch.no_grad():
            for sequence in iterator:
                # токенизируем всю последовательность сразу (с CLS/EOS)
                toks = self.tokenizer(sequence, return_tensors='pt', add_special_tokens=True, padding=False)
                input_ids = toks['input_ids'].to(self.device)    # [1, T]
                attn_mask = toks.get('attention_mask', torch.ones_like(input_ids)).to(self.device)

                T = input_ids.size(1)

                # чанкуем ПОВЕРХ уже токенизированной последовательности (как в fair-esm коде)
                tokens_list = []
                end = 0
                while end <= T:
                    start = end
                    end = start + self._CHUNK
                    if end <= T:
                        end = end - self._STEP_BACK  # все, кроме последнего окна, короче

                    ids_chunk = input_ids[:, start:end]
                    mask_chunk = attn_mask[:, start:end]

                    # важно: просим скрытые состояния, чтобы взять конкретный слой
                    out = self.model(input_ids=ids_chunk,
                                     attention_mask=mask_chunk,
                                     output_hidden_states=True)

                    # hidden_states: tuple(len = num_layers+1), [0] — эмбеддинг, [L] — верхний слой
                    # выбираем ровно тот слой, что использовался ранее
                    layer_h = out.hidden_states[self.return_layer]  # [1, chunk_len, H]
                    tokens_list.append(layer_h)

                out_all = torch.cat(tokens_list, dim=1).detach()  # [1, T, H]
                out_all[out_all != out_all] = 0.0  # NaN -> 0, как в исходнике

                # убрать спец-токены (CLS/EOS), получить [seq_len, H]
                res = out_all.transpose(0, 1)[1:-1]   # -> [T-2, 1, H]
                seq_embedding = res[:, 0].cpu()       # -> [T-2, H]
                embeddings.append(seq_embedding)

        return embeddings


def infer_sizes(state_dict):
    '''Retrieve the weight shapes for a LSTMCNNCRF checkpoint.'''
    n_filters = state_dict['feature_extractor.conv1.weight'].shape[0]
    filter_size = state_dict['feature_extractor.conv1.weight'].shape[2]
    hidden_size = state_dict['feature_extractor.conv2.weight'].shape[1] //2
    return n_filters, filter_size, hidden_size


def load_models(model_list):
    '''Load all the models from a list of checkpoints.'''
    models = []
    for path in model_list:
        state_dict = torch.load(path, map_location='cpu')
        n_filters, filter_size, hidden_size = infer_sizes(state_dict)
        model = LSTMCNNCRF(n_filters=n_filters, filter_size=filter_size, hidden_size=hidden_size, num_labels=3, num_states=101)
        model.eval()
        model.load_state_dict(state_dict)
        models.append(model)

    return models


def combine_crf(models):
    '''Make an ensemble CRF.'''

    transitions = []
    starts = []
    ends = []

    for m in models:
        transitions.append(m.crf.transitions)
        starts.append(m.crf.start_transitions)
        ends.append(m.crf.end_transitions)

    with torch.no_grad():
        crf = CRF(101, batch_first=True, include_start_end_transitions=True)
        crf.transitions.data = torch.stack(transitions, dim=0).mean(dim=0)
        crf.start_transitions.data = torch.stack(starts, dim=0).mean(dim=0)
        crf.end_transitions.data = torch.stack(ends, dim=0).mean(dim=0)

    return crf

def batchify_embeddings(embeddings, batch_size: int = 100):
    '''Make padded batches from list of embedding tensors.'''
    b_start = 0
    b_end = batch_size
    batches = []
    while b_start<len(embeddings):
        batch_embeddings = embeddings[b_start:b_end]
        batch_masks = [torch.ones(x.shape[0]) for x in batch_embeddings]
        batch_embeddings = torch.nn.utils.rnn.pad_sequence(batch_embeddings, batch_first=True)
        batch_masks = torch.nn.utils.rnn.pad_sequence(batch_masks, batch_first=True)

        batches.append((batch_embeddings, batch_masks))
        b_start = b_start + batch_size
        b_end = b_end + batch_size

    return batches

def batchify_sequences(sequences, batch_size: int = 100):
    '''Make batches from list of sequences.'''
    b_start = 0
    b_end = batch_size
    batches = []
    while b_start<len(sequences):
        batch_sequences = sequences[b_start:b_end]
        batch_masks = [torch.ones(len(x)) for x in batch_sequences]

        batches.append((batch_sequences, batch_masks))
        b_start = b_start + batch_size
        b_end = b_end + batch_size

    return batches


def convert_path_to_peptide_borders(pred: List[int], start_state, stop_state, offset: int=0) -> List[Tuple[int,int]]:
    '''Given a sequence of states, find the borders of contiguous peptide segments.
       Offset adds a constant to all coordinates (1-based indexing in uniprot)
    '''

    seq_peptides = []
    is_peptide = False

    for pos, p in enumerate(pred):

        if p == start_state and not is_peptide: # open a new peptide
            is_peptide = True
            peptide_start = pos

        # Close the peptide at the position that has the stop state. (can restart peptide immediately without NO-peptide gap.)
        elif p == stop_state and is_peptide: #close the peptide
            is_peptide = False
            seq_peptides.append((peptide_start +offset, pos +offset))
        else:
            pass # for positions that are not start_state or stop_state, do nothing.

    # close the last peptide if same as sequence end.
    if is_peptide:
        seq_peptides.append((peptide_start +offset,pos +offset))

    return seq_peptides


def simplify_probs(probs):
    out = []
    for p in probs:
        probs_simple = p[:,:3].copy()
        probs_simple[:,1] =  p[:,1:51].sum(axis=1)
        probs_simple[:,2] =  p[:,51:].sum(axis=1)
        out.append(probs_simple)

    return out

def simplify_preds(preds):

    def simplify_fn(x):
        if x>0:
            if x>50:
                return 2
            else:
                return 1
        else:
            return 0

    out = []
    for pred in preds:
        pred_simple = [simplify_fn(x) for x in pred]
        out.append(pred_simple)
    return out

import re
import unicodedata
def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = unicodedata.normalize('NFKD', value)
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value



import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
def plot_predictions(probs: np.ndarray, preds:List[int], save_path: str):

    cmap = matplotlib.colors.ListedColormap(['#DFDBDB',  '#048BA8', '#E8AE68'], name='from_list', N=None)
    fig = plt.figure(figsize=(12,4))
    axs = matplotlib.gridspec.GridSpec(
                    nrows=2,
                    ncols=2,
                    width_ratios=[1,0.01],
                    wspace=0.1 / 6,
                    # hspace=0.13 / height,
                    height_ratios=[3,0.5],
                )


    ax = fig.add_subplot(axs[0,0])
    ax.plot(probs[:,0], fillstyle='full', label='None', linestyle='--', linewidth=0.5,  c=cmap.colors[0])
    ax.plot(probs[:,1], fillstyle='full', label='Peptide', c=cmap.colors[1])
    ax.plot(probs[:,2], fillstyle='full', label='Propeptide', c=cmap.colors[2])



    ax.set_ylim(-0.01,1.05)
    ax.axhline(0.5, linestyle='--', c='red', xmin=0, xmax=1, linewidth=1)
    ax.set_ylabel('Probability')
    ax.yaxis.grid(False)
    ax.xaxis.grid()
    ax.legend(loc='upper left')
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    sns.despine(ax=ax, bottom=False)

    ax = fig.add_subplot(axs[1,0], sharex=ax)

    norm = matplotlib.colors.BoundaryNorm([0,1,2],2)
    preds = np.array([preds]) # make a 2D array so imshow works
    ax.imshow(preds, cmap=cmap, aspect='auto', norm=norm)
    ax.grid(False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    sns.despine(ax=ax,left=True)
    ax.set_ylabel('Prediction', rotation='horizontal', ha='right', va='center')
    ax.set_xlabel('Sequence position')


    plt.savefig(save_path)
    plt.close()