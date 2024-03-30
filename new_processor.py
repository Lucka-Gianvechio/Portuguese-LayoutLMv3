"""special2bbox = {
    tokenizer.cls_token_id : lmv3_tok.cls_token_box,
    tokenizer.sep_token_id : lmv3_tok.sep_token_box,
    tokenizer.pad_token_id : lmv3_tok.pad_token_box
}


pad_obj = {
    "input_ids": tokenizer.pad_token_id,
    "bbox": special2bbox[tokenizer.pad_token_id],
    "label": -100
}"""
from typing import Any
import torch

SPECIAL2BBOX = {101: [0, 0, 0, 0], 102: [0, 0, 0, 0], 0: [0, 0, 0, 0]}
PAD_OBJ = {'input_ids': 0, 'bbox': [0, 0, 0, 0], 'labels': -100, 'attention_mask': 0}



class BertimbauLayoutLMv3Processor:

    """
    Classe que integra o processador do LayoutLMv3 com o tokenizador
    do Bertimbau.
    """

    def __init__(
        self,
        layoutlmv3_processor,
        bertimbau_tokenizer
    ) -> None:
        self.processor = layoutlmv3_processor
        self.tokenizer = bertimbau_tokenizer
        self.PAD_OBJ = PAD_OBJ
        self.SPECIAL2BBOX = SPECIAL2BBOX


    def _pad_tokenized(
        self,
        tokenized_dict,
        max_length
        ) -> dict:
        """
        Completa os items do dicionario tokenizado com seus items de tokenização até
        atingir o tamanho máximo.
        """

        padded_dict = tokenized_dict.copy()

        for key, val in tokenized_dict.items():
            if len(val) >= max_length:
                return tokenized_dict

            padded_dict[key] = val + [self.PAD_OBJ[key] for k in range(max_length - len(val))]

        return padded_dict

    def tokenize(
        self,
        words,
        boxes,
        word_labels = [],
        max_length = 512
        ) -> dict:
        """
        Recebe uma lista de palavras e bboxs extraídas do ocr e uma lista de labels
        e retorna uma lista de tokens, cujas bboxs e labels herdam a da palavra original.
        """

        # Verifica se há uma correspondencia 1 para 1
        assert len(words) == len(boxes)

        input_ids = []
        guide = []
        tokenized_boxes = []
        tokenized_labels = []


        for wd, box in zip(words, boxes):
            if wd == "":
                raise ValueError("words must not be \"\" (void string)")
            tokenized_word_with_cls_sep = self.tokenizer.encode(wd)

            # Remove o cls e sep
            tokenized_word = tokenized_word_with_cls_sep[1:-1]

            # Herda as bboxs da palavra para cada token
            tokenized_box = [box for k in tokenized_word]

            # Guia usado para controlar quantos tokens foram gerados por palavra
            guide.append(len(tokenized_word))
            input_ids += tokenized_word
            tokenized_boxes += tokenized_box

        # Truncamento
        if len(input_ids) >= max_length - 2:
            restrict = slice(0, max_length - 2)
        else:
            restrict = slice(0, len(input_ids))

        input_ids = [self.tokenizer.cls_token_id] + input_ids[restrict] + [self.tokenizer.sep_token_id]
        tokenized_boxes = [self.SPECIAL2BBOX[self.tokenizer.cls_token_id]] + \
                tokenized_boxes[restrict] + \
                [self.SPECIAL2BBOX[self.tokenizer.sep_token_id]]

        # Adiciona a attention mask
        attention_mask = [1 for i in input_ids]

        return_dict =  {"input_ids": input_ids, "bbox": tokenized_boxes, "attention_mask": attention_mask}

        # Adiciona as labels a partir do guia
        if len(word_labels) != 0:
            assert len(word_labels) == len(words)

            for wd_len, label in zip(guide, word_labels):

                tokenized_label = [-100 for k in range(wd_len)]
                tokenized_label[0] = label

                tokenized_labels += tokenized_label

            tokenized_labels = [-100] + tokenized_labels[restrict] + [-100]

            return_dict["labels"] = tokenized_labels

        # Efetua o padding
        padded = self._pad_tokenized(return_dict, max_length)

        final_output = {}
        for k, v in padded.items():
            final_output[k] = [v]

        # O guia é útil para mascarar os tokens
        final_output["guide"] = [guide]

        return final_output


    def tokenize_batches(
        self,
        words,
        boxes,
        word_labels = [],
        max_length = 512
        ) -> dict:
        """
        Tokeniza um batch
        """

        tokenized_batches = {
            "input_ids": [],
            "bbox": [],
            "attention_mask": [],
            "guide": []
        }
        if len(word_labels) != 0:
            tokenized_batches["labels"] = []

        for batch_idx in range(len(words)):

            batch = {
                "words": words[batch_idx],
                "boxes": boxes[batch_idx],
                "max_length": max_length,
            }
            if len(word_labels) != 0:
                batch["word_labels"] = word_labels[batch_idx]

            tokenized_batch = self.tokenize(
                    **batch
                )

            for k, v in tokenized_batch.items():
                tokenized_batches[k].append(v)

        return tokenized_batch


    def __call__(
        self,
        images: Any or list[Any],
        words: list[str] or list[list[str]],
        boxes: list[list[int]] or list[list[list[int]]],
        word_labels: list[int] or list[list[int]] = [],
        max_length: int = 512
        ) -> dict:
        """
        Processa a entrada da rede para um batch, retorna um dicionario com tensores
        """

        assert len(words) == len(boxes)

        tok_entrie = {
            "words": words,
            "boxes": boxes,
            "max_length": max_length
        }

        processor_entrie = {
            "text": words,
            "boxes": boxes,
            "max_length": max_length,
            "images": images,
            "return_tensors": "pt",
            "truncation": True,
            "padding": "max_length"
        }

        if not word_labels == []:
            assert len(words) == len(word_labels)
            tok_entrie["word_labels"] = word_labels
            processor_entrie["word_labels"] = word_labels


        pre_processed = self.processor(
            **processor_entrie
        )

        if isinstance(words[0], list):
            assert len(images) == len(words)
            tokenized = self.tokenize_batches(
                **tok_entrie
            )
        else:
            tokenized = self.tokenize(
                **tok_entrie
            )

        processed = {"pixel_values": pre_processed["pixel_values"]}
        for k, v in tokenized.items():
            processed[k] = torch.tensor(v, dtype = torch.int32)

        return processed
