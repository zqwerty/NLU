# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
from pprint import pprint

from allennlp.common.checks import check_for_gpu
from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from zipfile import ZipFile

from tasktk.modules.nlu.nlu import NLU 
from tasktk.modules.nlu.mlst import model, dataset_reader 

DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "models/model.tar.gz")
class MlstNLU(NLU):
    """Multilabel sequence tagging model."""

    def __init__(self, 
                archive_file=DEFAULT_ARCHIVE_FILE,
                cuda_device=DEFAULT_CUDA_DEVICE,
                model_file=None):
        """ Constructor for NLU class. """
        check_for_gpu(cuda_device)

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MlstNLU is specified!")
            file_path = cached_path(model_file)
            zip_ref = ZipFile(file_path, 'r')
            zip_ref.extractall(DEFAULT_DIRECTORY)
            zip_ref.close()

        archive = load_archive(archive_file,
                            cuda_device=cuda_device)
        self.tokenizer = SpacyWordSplitter(language="en_core_web_sm") 
        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.model = archive.model
        self.model.eval()

    def parse(self, utterance):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        # print("nlu input:")
        # pprint(utterance)

        if len(utterance) == 0:
            return {} 

        tokens = self.tokenizer.split_words(utterance)
        instance = self.dataset_reader.text_to_instance(tokens)
        outputs = self.model.forward_on_instance(instance)
        # spans = bio_tags_to_spans(outputs["tags"])
        # dialacts = {}
        # for span in spans:
        #     domain_act = span[0].split("+")[0]
        #     slot = span[0].split("+")[1]
        #     value = " ".join(outputs["words"][span[1][0]:span[1][1]+1])
        #     if domain_act not in dialacts:
        #         dialacts[domain_act] = [[slot, value]]
        #     else:
        #         dialacts[domain_act].append([slot, value])
        # for intent in outputs["intents"]:
        #     if "+" in intent: 
        #         # Request
        #         domain_act = intent.split("+")[0] 
        #         if domain_act not in dialacts:
        #             dialacts[domain_act] = [[intent.split("+")[1], "?"]]
        #         else:
        #             dialacts[domain_act].append([intent.split("+")[1], "?"])
        #     else:
        #         dialacts[intent] = [["none", "none"]]

        # pprint(outputs["dialog_act"])
        return outputs["dialog_act"]


if __name__ == "__main__":
    nlu = MlstNLU()
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?"
        "you're welcome! enjoy your visit! goodbye.",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "What is the Name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for utt in test_utterances:
        print(utt)
        pprint(nlu.parse(utt))
