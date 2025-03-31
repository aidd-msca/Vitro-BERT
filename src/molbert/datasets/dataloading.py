import logging
from typing import Callable

import torch
from torch.utils.data import DataLoader
from molbert.datasets.smiles import BertSmilesDataset
import random
import numpy as np


class MolbertDataLoader(DataLoader):
    """
    A custom data loader that does some molbert specific things.
    1) it skips invalid batches and replaces them with oversampled valid batches such that always n_batches are
       created.
    2) it does the valid filtering and trimming in the workers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # See dataloader.pyi for examplanation of type: ignore
        self.collate_fn = self.wrapped_collate_fn(self.collate_fn)  # type: ignore

    @staticmethod
    def trim_batch(batch_inputs, batch_labels):
        _, cols = torch.where(batch_inputs['attention_mask'] == 1)
        max_idx: int = int(cols.max().item() + 1)

        for k in ['input_ids', 'token_type_ids', 'attention_mask']:
            batch_inputs[k] = batch_inputs[k][:, :max_idx].contiguous()

        for k in ['lm_label_ids', 'unmasked_lm_label_ids']:
            batch_labels[k] = batch_labels[k][:, :max_idx].contiguous()
        

        return batch_inputs, batch_labels

    def wrapped_collate_fn(self, collate_fn) -> Callable:
        def collate(*args, **kwargs):
            batch = collate_fn(*args, **kwargs)

            # valids here is a sequence of valid flags with the same length as the batch
            (batch_inputs, batch_labels), valids = batch
            if not valids.all():
                # filter invalid
                batch_inputs = {k: v[valids] for k, v in batch_inputs.items()}
                batch_labels = {k: v[valids] for k, v in batch_labels.items()}
                # keep trues only to make sure the format is the same
                valids = valids[valids]

            # whole batch is invalid?
            if len(valids) == 0:
                return (None, None), valids

            # trim out excessive padding
            batch_inputs, batch_labels = self.trim_batch(batch_inputs, batch_labels)

            return (batch_inputs, batch_labels), valids

        return collate

    def __iter__(self):
        num_batches_so_far = 0
        num_total_batches = len(self)
        num_accessed_batches = 0

        while num_batches_so_far < num_total_batches:
            for (batch_inputs, batch_labels), valids in super().__iter__():
                num_accessed_batches += 1
                if len(valids) == 0:
                    logging.info('EMPTY BATCH ENCOUNTERED. Skipping...')
                    continue
                num_batches_so_far += 1
                yield (batch_inputs, batch_labels), valids

        logging.info(
            f'Epoch finished. Accessed {num_accessed_batches} batches in order to train on '
            f'{num_total_batches} batches.'
        )

#####################################################################################
# trim both corrupted and clean tokens
#####################################################################################
class MolbertDataLoader_corrupted_clean_seq(DataLoader):
    """
    A custom data loader that does some molbert specific things.
    1) it skips invalid batches and replaces them with oversampled valid batches such that always n_batches are
       created.
    2) it does the valid filtering and trimming in the workers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # See dataloader.pyi for examplanation of type: ignore
        self.collate_fn = self.wrapped_collate_fn(self.collate_fn)  # type: ignore

    @staticmethod
    def trim_batch(batch_inputs, batch_labels,trim_batch = False):
        
        if trim_batch == True:
            _, cols = torch.where(batch_inputs['attention_mask'] == 1)
            max_idx: int = int(cols.max().item() + 1)

            for k in ['input_ids', 'token_type_ids', 'attention_mask']:
                batch_inputs[k] = batch_inputs[k][:, :max_idx].contiguous()

            for k in ['lm_label_ids', 'unmasked_lm_label_ids']:
                batch_labels[k] = batch_labels[k][:, :max_idx].contiguous()
        

        return batch_inputs, batch_labels

    def wrapped_collate_fn(self, collate_fn) -> Callable:
        def collate(*args, **kwargs):
            batch = collate_fn(*args, **kwargs)

            # valids here is a sequence of valid flags with the same length as the batch
            (batch_inputs, batch_labels), valids = batch
            if not valids.all():
                # filter invalid
                batch_inputs = {k: v[valids] for k, v in batch_inputs.items()}
                batch_labels = {k: v[valids] for k, v in batch_labels.items()}
                # keep trues only to make sure the format is the same
                valids = valids[valids]

            # whole batch is invalid?
            if len(valids) == 0:
                return (None, None), valids

            # trim out excessive padding
            corrupted_batch_inputs, corrupted_batch_labels = self.trim_batch(batch_inputs[0], batch_labels)
            clean_batch_inputs, clean_batch_labels = self.trim_batch(batch_inputs[1], batch_labels)
            
            for key in corrupted_batch_labels.keys():
                assert torch.equal(corrupted_batch_labels[key], clean_batch_labels[key]), f"Tensors do not match for key {key}!"

            return ((corrupted_batch_inputs,clean_batch_inputs),  corrupted_batch_labels), valids

        return collate

    def __iter__(self):
        num_batches_so_far = 0
        num_total_batches = len(self)
        num_accessed_batches = 0

        while num_batches_so_far < num_total_batches:
            for (batch_inputs, batch_labels), valids in super().__iter__():
                num_accessed_batches += 1
                if len(valids) == 0:
                    logging.info('EMPTY BATCH ENCOUNTERED. Skipping...')
                    continue
                num_batches_so_far += 1
                yield (batch_inputs, batch_labels), valids

        logging.info(
            f'Epoch finished. Accessed {num_accessed_batches} batches in order to train on '
            f'{num_total_batches} batches.'
        )
#####################################################################################
# Combine two dataloaders
#####################################################################################

class MolbertDataLoader_for_Multiple_datasets(DataLoader):
    """
    A custom data loader that does some MolBERT specific things.
    1) It skips invalid batches and replaces them with oversampled valid batches such that always n_batches are
       created.
    2) It does the valid filtering and trimming in the workers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = self.wrapped_collate_fn(self.collate_fn)  # type: ignore

    @staticmethod
    def trim_batch(self,batch_inputs, batch_labels, max_idx):

        # Trim keys based on the maximum sequence length
        for k in ['input_ids', 'token_type_ids', 'attention_mask']:
            batch_inputs[k] = batch_inputs[k][:, :max_idx].contiguous()

        for k in ['lm_label_ids', 'unmasked_lm_label_ids']:
            batch_labels[k] = batch_labels[k][:, :max_idx].contiguous()

        return batch_inputs, batch_labels

    def wrapped_collate_fn(self, collate_fn) -> Callable:
        def collate(*args, **kwargs):
            batch = collate_fn(*args, **kwargs)

            (batch_inputs, batch_labels), valids = batch

            if not valids.all():
                batch_inputs = {k: v[valids] for k, v in batch_inputs.items()}
                batch_labels = {k: v[valids] for k, v in batch_labels.items()}
                valids = valids[valids]

            if len(valids) == 0:
                return (None, None), valids

            #batch_inputs, batch_labels = self.trim_batch(batch_inputs, batch_labels)

            return (batch_inputs, batch_labels), valids

        return collate

    def __iter__(self):
        return super().__iter__()
    
class Combined_invitro_invivo_dataloader:
    """
    A custom data loader that combines two MolBERT data loaders.
    Iterates through the first data loader and cycles through the second data loader.
    """

    def __init__(self, 
                 dataloader1: MolbertDataLoader_for_Multiple_datasets, 
                 dataloader2: MolbertDataLoader_for_Multiple_datasets,
                 dataloader3: MolbertDataLoader_for_Multiple_datasets,
                 dataloader4: MolbertDataLoader_for_Multiple_datasets):
        
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.dataloader3 = dataloader3
        self.dataloader4 = dataloader4

    def __iter__(self):
        iter1 = iter(self.dataloader1)
        iter2 = cycle(self.dataloader2)
        iter3 = iter(self.dataloader3)
        iter4 = cycle(self.dataloader4)

        while True:
            try:
                batch1 = next(iter1)
                batch2 = next(iter2)
                batch3 = next(iter3)
                batch4 = next(iter4)
            except StopIteration:
                break

           

            (batch_inputs1, batch_labels1), valids1 = batch1
            (batch_inputs2, batch_labels2), valids2 = batch2
            (batch_inputs3, batch_labels3), valids3 = batch3
            (batch_inputs4, batch_labels4), valids4 = batch4
            #print("invitro", batch_inputs1["attention_mask"].shape, batch_inputs3["attention_mask"].shape)
            #print("invivo", batch_inputs2["attention_mask"].shape, batch_inputs4["attention_mask"].shape)

            if len(valids1) == 0 and len(valids2) == 0 and len(valids3) == 0 and len(valids4) == 0:
                logging.info('EMPTY BATCH ENCOUNTERED. Skipping...')
                continue
            '''
            # trim batches based on maximum sequence length
            _, cols = torch.where(batch_inputs1['attention_mask'] == 1)
            max_idx_1: int = int(cols.max().item() + 1)

            _, cols = torch.where(batch_inputs2['attention_mask'] == 1)
            max_idx_2: int = int(cols.max().item() + 1)

            _, cols = torch.where(batch_inputs3['attention_mask'] == 1)
            max_idx_3: int = int(cols.max().item() + 1)

            _, cols = torch.where(batch_inputs4['attention_mask'] == 1)
            max_idx_4: int = int(cols.max().item() + 1)


            max_idx = max(max_idx_1, max_idx_2,max_idx_3, max_idx_4)
            print("all max", max_idx_1, max_idx_2,max_idx_3, max_idx_4)
            print("max_idx", max_idx)

            batch_inputs1, batch_labels1 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs1, batch_labels1, max_idx)
            batch_inputs2, batch_labels2 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs2, batch_labels2, max_idx)
            batch_inputs3, batch_labels3 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs3, batch_labels3, max_idx)
            batch_inputs4, batch_labels4 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs4, batch_labels4, max_idx)

            print("invitro_trim", batch_inputs1["attention_mask"].shape, batch_inputs3["attention_mask"].shape)
            print("invivo_trim ", batch_inputs2["attention_mask"].shape, batch_inputs4["attention_mask"].shape)
            '''
            yield [((batch_inputs1, batch_labels1), valids1), 
                   ((batch_inputs2, batch_labels2), valids2),
                   ((batch_inputs3, batch_labels3), valids3),
                   ((batch_inputs4, batch_labels4), valids4)]

        logging.info('Epoch finished.')

    def __len__(self):
        return len(self.dataloader1)
    
class CombinedMolbertDataLoader_max:
    """
    A custom data loader that combines two MolBERT data loaders.
    Iterates through the first data loader and cycles through the second data loader.
    """

    def __init__(self, 
                 dataloader1: MolbertDataLoader_for_Multiple_datasets, 
                 dataloader2: MolbertDataLoader_for_Multiple_datasets,
                 dataloader3: MolbertDataLoader_for_Multiple_datasets,
                 dataloader4: MolbertDataLoader_for_Multiple_datasets):
        
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.dataloader3 = dataloader3
        self.dataloader4 = dataloader4

    def __iter__(self):
        iter1 = iter(self.dataloader1)
        iter2 = iter(self.dataloader2)
        iter3 = iter(self.dataloader3)
        iter4 = iter(self.dataloader4)

        while True:
            try:
                batch1 = next(iter1)
            except StopIteration:
                break

            batch2 = next(iter2, None)
            batch3 = next(iter3, None)
            batch4 = next(iter4, None)
           

            (batch_inputs1, batch_labels1), valids1 = ((None, None), None) if batch1 is None else batch1
            (batch_inputs2, batch_labels2), valids2 = ((None, None), None) if batch2 is None else batch2
            (batch_inputs3, batch_labels3), valids3 = ((None, None), None) if batch3 is None else batch3
            (batch_inputs4, batch_labels4), valids4 = ((None, None), None) if batch4 is None else batch4

            if all(v is None for v in [valids1, valids2, valids3, valids4]):
                logging.info('EMPTY BATCH ENCOUNTERED. Skipping...')
                continue
            '''
            # trim batches based on maximum sequence length
            max_idx_1 = (int(torch.where(batch_inputs1['attention_mask'] == 1)[1].max().item() + 1) 
                         if batch_inputs1 is not None else 1)

            max_idx_2 = (int(torch.where(batch_inputs2['attention_mask'] == 1)[1].max().item() + 1) 
                         if batch_inputs2 is not None else 1)

            max_idx_3 = (int(torch.where(batch_inputs3['attention_mask'] == 1)[1].max().item() + 1) 
                         if batch_inputs3 is not None else 1)

            max_idx_4 = (int(torch.where(batch_inputs4['attention_mask'] == 1)[1].max().item() + 1) 
                         if batch_inputs4 is not None else 1)


            max_idx = max(max_idx_1, max_idx_2,max_idx_3, max_idx_4)
            if batch_inputs1 is not None:
                batch_inputs1, batch_labels1 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs1, batch_labels1, max_idx)
            if batch_inputs2 is not None:
                batch_inputs2, batch_labels2 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs2, batch_labels2, max_idx)
            if batch_inputs3 is not None:
                batch_inputs3, batch_labels3 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs3, batch_labels3, max_idx)
            if batch_inputs4 is not None:
                batch_inputs4, batch_labels4 = MolbertDataLoader_for_Multiple_datasets.trim_batch(batch_inputs4, batch_labels4, max_idx)
            '''
            yield [((batch_inputs1, batch_labels1), valids1), 
                   ((batch_inputs2, batch_labels2), valids2),
                   ((batch_inputs3, batch_labels3), valids3),
                   ((batch_inputs4, batch_labels4), valids4)]

        logging.info('Epoch finished.')

    def __len__(self):
        return len(self.dataloader1)
    

def get_dataloaders(featurizer, targets, num_workers,config_dict, train_shuffle = True):

    if targets == "invitro":
        train_input_path = config_dict['invitro_train']
        val_input_path = config_dict['invitro_val']
        label_column = config_dict["invitro_columns"]
        num_tasks = config_dict["num_invitro_tasks"]
        batch_size = config_dict["invitro_batch_size"]

    if targets == "invivo":
        train_input_path = config_dict['invivo_train']
        val_input_path = config_dict['invivo_test']
        label_column = config_dict["invivo_columns"]
        num_tasks = config_dict["num_invivo_tasks"]
        batch_size = config_dict["invivo_batch_size"]


    train_dataset = BertSmilesDataset(
            input_path= train_input_path,
            featurizer= featurizer,
            single_seq_len= config_dict["max_seq_length"],
            total_seq_len= config_dict["max_seq_length"],
            label_column= label_column,
            is_same= config_dict["is_same_smiles"],
            num_invitro_tasks = num_tasks,
            num_physchem= config_dict["num_physchem_properties"],
            permute= config_dict["permute"],
            named_descriptor_set=config_dict["named_descriptor_set"],
            inference_mode = False
        )

    validation_dataset = BertSmilesDataset(
                input_path= val_input_path,
                featurizer= featurizer,
                single_seq_len= config_dict["max_seq_length"],
                total_seq_len= config_dict["max_seq_length"],
                label_column= label_column,
                is_same= config_dict["is_same_smiles"],
                num_invitro_tasks = num_tasks,
                num_physchem= config_dict["num_physchem_properties"],
                permute= config_dict["permute"],
                named_descriptor_set=config_dict["named_descriptor_set"],
                inference_mode = False
            )

    train_dataloader = MolbertDataLoader_corrupted_clean_seq(train_dataset, 
                                        batch_size=batch_size,
                                        pin_memory=False,
                                        num_workers=num_workers, 
                                        shuffle = train_shuffle)

    val_dataloader = MolbertDataLoader_corrupted_clean_seq(validation_dataset, 
                                        batch_size=batch_size,
                                        pin_memory=False,
                                        num_workers=num_workers, 
                                        shuffle = False)
    
    return  train_dataloader, val_dataloader