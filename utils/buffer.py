# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch


def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])


class DomainBuffer:
    """
    Buffer for single (class) of the data.
    """
    SAMPLE_METHODS = ('random')
    MAX_SIZE = 4096
    def __init__(self, max_size, device, input_size, num_classes, domain_id, examples=None, preds=None, labels=None):
        self.buffer_size = max_size # maximum number of data to be stored.
        self.mem_size = 0 # current number of data stored.
        self.ptr = 0 # pointer for returning the next batch.
        self.indices = None # indices used to iterate through the memory bank.
        self.device = device
        self.input_size = input_size # input shape of the data.
        self.num_classes = num_classes # number of classes.
        self.domain_id = domain_id

        self.examples = torch.zeros((max_size, *input_size), dtype=torch.float32) # data
        self.preds = torch.zeros((max_size, num_classes), dtype=torch.float32) # preds of the prev model.
        self.labels = torch.zeros((max_size, ), dtype=torch.int64) # true labels 

        # if initial data is provided.
        if all([x is not None for x in [examples, preds, labels]]):
            self.add_data(examples, preds, labels)
        
        # automatically send to device.
        self.to(device=self.device)

    def add_data(self, examples, preds, labels):
        """Add data to the buffer, overwriting the original data."""
        assert examples.shape[0] == preds.shape[0] == labels.shape[0], "The number of the examples passed to the memory doesn't match."
        assert tuple(examples.shape[1:]) == tuple(self.input_size), "The shape of the data doesn't match the designated input size."

        if examples.shape[0] <= self.buffer_size: 
            self.mem_size = mem_size = examples.shape[0]
            self.examples[:mem_size], self.preds[:mem_size], self.labels[:mem_size] = examples, preds, labels
        else:
            self.mem_size = self.buffer_size
            self.examples, self.preds, self.labels = self.resample(mem_size, examples, preds, labels)
        
        # set the indices.
        self.set_indices()

    def resample(self, sample_num, examples=None, preds=None, labels=None, mode='random'):
        """Resample the examples in the buffer, or given the examples do resampling."""
        # if it's inplace. 
        inplace = all([x is None for x in [examples, preds, labels]])
        
        buffer_size = self.mem_size if inplace else examples.shape[0] 
        sample_num = min(buffer_size, sample_num)

        # simple random wi
        assert mode in self.SAMPLE_METHODS, f"Sample method '{mode}' not supported, current supported methods: {self.SAMPLE_METHODS}"
        inds = self.indices if inplace else np.arange(buffer_size)
        if mode == 'random':
            if not inplace: # if **NOT** inplacely resampling the buffer.
                np.random.shuffle(inds)
                new_inds = inds[:sample_num]
                return examples[inds], preds[inds], labels[inds]
            # guarantee each class has one sample.
            num_per_cls = sample_num // self.num_classes
            new_inds = []
            for cls in range(self.num_classes):
                cls_inds = inds[(self.labels[inds]==cls).cpu().numpy()]
                np.random.shuffle(cls_inds)
                new_inds.append(cls_inds[:num_per_cls])
            self.indices = np.hstack(new_inds)
        
        self.mem_size = len(self.indices)
    
    def set_indices(self):
        self.indices = np.arange(self.mem_size)

    def shuffle_indices(self):
        np.random.shuffle(self.indices)

    def collect_data(self):
        return self.examples[self.indices], self.preds[self.indices], self.labels[self.indices]

    def __len__(self):
        return self.mem_size

    def to(self, device):
        self.device = device
        self.examples = self.examples.to(device)
        self.preds = self.preds.to(device)
        self.labels = self.labels.to(device)
        return self

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        return self.mem_size == 0


class Buffer:
    """
    The memory buffer.
    """

    def __init__(self, buffer_size, device, input_size, num_classes, batch_size=0, domain_buffers=None):
        self.buffer_size = buffer_size # maximum number of data to be stored.
        self.indices = None # indices used to iterate through the memory bank.
        self.device = device
        self.input_size = input_size # input shape of the data.
        self.num_classes = num_classes # number of classes.
        self.batch_size = batch_size

        self.domain_buffers = domain_buffers if domain_buffers is not None else []
        self.n_domains = len(self.domain_buffers)

        self.reset()

    def __len__(self):
        if not self.domain_buffers: return 0
        return sum([x.mem_size for x in self.domain_buffers])

    def __iter__(self):
        self.collect_all_data()
        self.shuffle()
        self.ptr = 0
        return self
        
    def __next__(self):
        assert self.indices is not None, "Not Iterable, please call iter() to get the loader."
        assert self.ptr < self.examples.shape[0], "Out of range of the buffer, need to reinitialize."

        if self.batch_size == 0: return self.examples, self.labels, self.preds, self.domain_ids # batch_size = 0: return all the samples.

        inds = self.indices[self.ptr: self.ptr + self.batch_size] 
        res = self.examples[inds], self.labels[inds], self.preds[inds], self.domain_ids[inds]
        self.ptr += self.batch_size # increase the ptr.
        
        return res

    def to(self, device):
        if not self.domain_buffers: # if empty domain_buffers. 
            return self

        for buf in self.domain_buffers:
            buf.to(device)
        return self

    def update(self, examples, labels, preds):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param domain_ids: tensor containing the task labels
        :return:
        """
        self.n_domains += 1

        # create and add new domain buffer.
        self.domain_buffers.append(DomainBuffer(
            DomainBuffer.MAX_SIZE, 
            self.device, input_size=self.input_size, num_classes=self.num_classes, 
            domain_id=self.n_domains, 
            examples=examples, labels=labels, preds=preds
        ))
        
        if len(self) > self.buffer_size: 
            # if exceeding the buffer capacity.
            # calculate num of examples for each domain
            # currently evenly preserve the number of each domains.
            num_samples = self.buffer_size // self.n_domains
            for buf in self.domain_buffers:
                buf.resample(num_samples)

        # reset the indices and data. 
        self.reset()

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        return len(self) == 0

    def reset(self):
        """reset the data and pointer."""
        self.examples, self.preds, self.labels, self.domain_ids = None, None, None, None
        self.examples_list, self.preds_list, self.labels_list, self.domain_ids_list = None, None, None, None
        self.ptr = 0

    def collect_all_data(self, combine=True):
        if len(self) == 0: return
        if not combine:
            if self.examples_list is not None:
                return self.examples_list, self.preds_list, self.labels_list, self.domain_ids_list
        elif self.examples is not None:
            return self.examples, self.preds, self.labels, self.domain_ids
        
        # after reset, need to recollect.
        examples, preds, labels, domain_ids = [], [], [], []
        for buf in self.domain_buffers:
            ee, pp, ll = buf.collect_data()
            examples.append(ee)
            preds.append(pp)
            labels.append(ll)
            domain_ids.append(buf.domain_id)
        
        # separately return each domain's memory bank.
        if not combine: 
            self.examples_list, self.preds_list, self.labels_list, self.domain_ids_list = examples, preds, labels, domain_ids
            return self.examples_list, self.preds_list, self.labels_list, self.domain_ids_list

        self.examples = torch.vstack(examples)
        self.preds = torch.vstack(preds)
        self.labels = torch.hstack(labels)
        self.domain_ids = torch.hstack([buf.domain_id * torch.ones((buf.mem_size,)) for buf in self.domain_buffers])
        return self.examples, self.preds, self.labels, self.domain_ids

    def collect_all_sizes(self):
        return torch.tensor([buf.mem_size for buf in self.domain_buffers])
    
    def shuffle(self):
        assert self.examples is not None, "Data from the domain buffers not set up. call 'iter()' before shuffling the data."
        
        self.indices = np.arange(self.examples.shape[0])
        np.random.shuffle(self.indices)