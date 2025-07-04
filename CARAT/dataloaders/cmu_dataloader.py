# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import json
import random
import time
import pickle
import math


"""
CMU-MOSEI info
Train 16326 samples
Val 1871 samples
Test 4659 samples
CMU-MOSEI feature shapes
visual: (60, 35)
audio: (60, 74)
text: GLOVE->(60, 300)
label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
    averaged from 3 annotators
unaligned:
text: (50, 300)
visual: (500, 35)
audio: (500, 74)    
"""
FIXED_LENGTH = 500
emotion_dict = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}
class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, args):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        data = torch.load(self.data_path)
        data = data[data_type]
        visual = data['src-visual']
        audio = data['src-audio']
        text = data['src-text']
        labels = data['tgt']      
        return visual, audio, text, labels

    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]
        text_mask = np.array(text_mask)
        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)
        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]
        audio_mask =  np.array(audio_mask)
        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] = 1
        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)
        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label


class M3EDDataset(Dataset):
    def __init__(self, data_path, data_type, args):
        self.args = args
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
        self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        data = torch.load(self.data_path)
        data = data[data_type]
        visual = data['src-visual']
        audio = data['src-audio']
        text = data['src-text']
        labels = data['tgt']
        return visual, audio, text, labels

    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]
        text_mask = np.array(text_mask)
        return text, text_mask

    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)
        return visual, visual_mask

    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]
        audio_mask = np.array(audio_mask)
        return audio, audio_mask

    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(7, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emo-4] = 1
        return label

    def _get_label_input(self):
        labels_embedding = np.arange(self.args.num_classes)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
               audio, audio_mask, label


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, args):
        self.data_path = data_path
        self.data_type = data_type
        self.args = args
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)


    def _get_data(self, data_type):
        label_data = torch.load(self.data_path)
        label_data = label_data[data_type]
        with open('./data/mosei_senti_data_noalign.pkl', 'rb') as f:
            data = pickle.load(f)
        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        audio = np.array(audio)
        labels = label_data['tgt']      
        return visual, audio, text, labels
    
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]
        text_mask = np.array(text_mask)
        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        if self.args.unaligned_mask_same_length:
            visual_mask = [1] * 50
        else:
            visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)
        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        if self.args.unaligned_mask_same_length:
            audio_mask = [1] * 50
        else:
            audio_mask = [1] * audio.shape[0]
        audio_mask = np.array(audio_mask)
        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] =  1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label
    

class CustomNonAlignedMOSEI(Dataset):
    def __init__(self, data_path, data_type, args):
        self.data_path = data_path
        self.data_type = data_type
        self.args = args
        self.visual, self.audio, self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        """Load custom unaligned data directly from .pt file"""
        data = torch.load(self.data_path)
        data = data[data_type]
        
        # Extract modalities with original temporal intervals preserved
        visual = data['src-visual']
        audio = data['src-audio'] 
        text = data['src-text']
        labels = data['tgt']
        
        return visual, audio, text, labels

    def _get_text(self, index):
        """Get text features with variable length preserved"""
        text = self.text[index]

        if self.args.unaligned_mask_same_length:
            if text.shape[0] > FIXED_LENGTH:
                text = text[:FIXED_LENGTH]
            text_mask = [1] * text.shape[0] + [0] * (FIXED_LENGTH - text.shape[0])
            if text.shape[0] < FIXED_LENGTH:
                padding = np.zeros((FIXED_LENGTH - text.shape[0], text.shape[1]))
                text = np.concatenate([text, padding], axis=0)
        else:
            text_mask = [1] * text.shape[0]

        #text_mask = [1] * text.shape[0]
        text_mask = np.array(text_mask)
        return text, text_mask

    def _get_visual(self, index):
        """Get visual features with original temporal intervals"""
        visual = self.visual[index]
        
        # Use actual sequence length for mask (no fixed length)
        if self.args.unaligned_mask_same_length:
            # Option to use fixed mask length if specified
            
            # Greater than fixed length so truncate:
            if visual.shape[0] > FIXED_LENGTH:
                visual = visual[:FIXED_LENGTH]
            visual_mask = [1] * visual.shape[0] + [0] * (FIXED_LENGTH - visual.shape[0])

            # Lesser than fixed length so pad:
            if visual.shape[0] < FIXED_LENGTH:
                padding = np.zeros((FIXED_LENGTH - visual.shape[0], visual.shape[1]))
                visual = np.concatenate([visual, padding], axis=0)

            #visual_mask = [1] * min(visual.shape[0], 500)  
        else: # using original sequence length if unaligned_mask_same_length is not me3ntioned
            visual_mask = [1] * visual.shape[0]

        visual_mask = np.array(visual_mask)
        return visual, visual_mask

    def _get_audio(self, index):
        """Get audio features with original temporal intervals"""
        audio = self.audio[index]        
    
        if self.args.unaligned_mask_same_length:
            FIXED_LENGTH = 500
            if audio.shape[0] > FIXED_LENGTH:
                audio = audio[:FIXED_LENGTH]
            audio_mask = [1] * audio.shape[0] + [0] * (FIXED_LENGTH - audio.shape[0])
            if audio.shape[0] < FIXED_LENGTH:
                padding = np.zeros((FIXED_LENGTH - audio.shape[0], audio.shape[1]))
                audio = np.concatenate([audio, padding], axis=0)
        else:
            audio_mask = [1] * audio.shape[0]

        audio_mask = np.array(audio_mask)
        return audio, audio_mask

    def _get_labels(self, index):
        """Get emotion labels - already processed to 6-emotion format"""
        label = self.labels[index]   
        return label.astype(np.float32)

    def _get_label_input(self):
        """Get label embedding for model input"""
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        return labels_embedding, labels_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
               audio, audio_mask, label
