import pytesseract
from pytesseract import Output
from pathlib import Path
import numpy.random as rnd
import torch
from numpy import linspace

from copy import deepcopy
from new_processor import BertimbauLayoutLMv3Processor
import ocr_tools




class PreProcessor:

    def __init__(self, model_processor : BertimbauLayoutLMv3Processor):
        self.model_processor = model_processor
        self.rng : rnd.Generator = rnd.default_rng()
        self.MASK_TOKEN_ID = self.model_processor.tokenizer.mask_token_id
        self.special_tokens = self.model_processor.SPECIAL2BBOX.keys()
        self.possible_subs = [k for k in range(0, self.model_processor.tokenizer.vocab_size) if k not in self.special_tokens]

        self.num_patches = 16
        self.x_marks = linspace(0, 1000, self.num_patches + 1)
        self.y_marks = linspace(0, 1000, self.num_patches + 1)

        self.patches = self._get_patches()

    def _get_patches(
        self
    ) -> list[int]:

        patches = []

        for y_idx in range(len(self.y_marks[:-1])):
            patch_line = []
            for x_idx in range(len(self.x_marks[:-1])):
                top_left_y = int(self.y_marks[y_idx])
                top_left_x = int(self.x_marks[x_idx])
                bottom_right_y = int(self.y_marks[y_idx+1])
                bottom_right_x = int(self.x_marks[x_idx+1])
                patch_line.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
            patches.append(patch_line)

        return patches

    def _get_bbox_patch(
        self,
        bbox: list[int]
    ) -> list[int]:

        for y_idx in range(len(self.y_marks[:-1])):
            if  self.y_marks[y_idx] <= bbox[1] and bbox[1] <= self.y_marks[y_idx+1]:
                break

        for x_idx in range(len(self.x_marks[:-1])):
            if  self.x_marks[x_idx] <= bbox[0] and bbox[0] <= self.x_marks[x_idx+1]:
                break

        box_patch = self.patches[y_idx][x_idx]

        return box_patch

    def get_image_ocr(self, path_to_image : Path):

        cv2_image = ocr_tools.read_image(
            image_path = path_to_image
        )

        resized_image, shape = ocr_tools.resize_image(
            cv2_image = cv2_image
        )

        preprocessed_image = ocr_tools.preprocess_image(
            cv2_image = resized_image
        )

        # configuring parameters for tesseract
        custom_config = r'--oem 3 --psm 6'

        # Get all OCR output information from pytesseract
        ocr_output_details = pytesseract.image_to_data(
            preprocessed_image,
            output_type = Output.DICT,
            config=custom_config,
            lang='eng'
        )

        text_bbox = ocr_tools.get_text_bbox(
            ocr_details = ocr_output_details
        )

        return text_bbox, shape, cv2_image

    def get_processed_data(self, text_bbox, shape, image):

        words = [k["text"] for k in text_bbox]
        boxes = [ocr_tools.normalize_bbox(k["bbox"], shape[0], shape[1]) for k in text_bbox]

        processed_data = self.model_processor(
                            images=image,
                            words=words,
                            boxes=boxes,
                            max_length=512
                        )

        return processed_data

    def _get_mask_index(self):
        w, h = 16, 16
        masking_ratio = 0.4*w*h
        M = set()
        while len(M) < masking_ratio:
            diff = max(masking_ratio - len(M), 1)
            s = max(16, self.rng.integers(0, diff))

            r = self.rng.uniform(0.3, 1/0.3)
            a, b = (s*r)**.5, (s/r)**.5
            t = self.rng.integers(0, max(int(h - a), 1)).astype(int)
            l = self.rng.integers(0, max(int(w - b), 1)).astype(int)
            a, b = int(a), int(b)
            M = M.union({(i, j) for i in range(t, t+a) for j in range(l, l+b)})

        return M

    def mask_image(self, image):
        pass

    def mask_tokens(self, processed_data) -> list[list[int]]:
        """
        Span masking ver artigo do spanbert
        """

        toks_batch = processed_data["input_ids"]
        guide_batch = processed_data["guide"]

        masked_tokens_batch = []


        for batch, guide_list in enumerate(guide_batch):

            # Word indexes
            masked_indexes = []

            # Number of tokens to mask per entrie
            max_budget = sum(guide_list) * 0.15
            masked_cost = 0
            ctr = 0

            # Runs until masked 15% of sentence or 100 iters max
            while masked_cost < max_budget and ctr < 100:
                ctr += 1

                span = min(self.rng.poisson(3), 9) + 1
                word_start = self.rng.integers(0, len(guide_list))

                if span + word_start >= len(guide_list):
                    continue

                word_span = list(range(word_start, word_start + span))

                if set(word_span).intersection({m[0] for m in masked_indexes}):
                    continue



                masking_type = self.rng.choice(["MASK", "SUB", "DONT"], p = [0.8, 0.1, 0.1])
                masked_indexes += [(w_s, masking_type) for w_s in word_span]

                masked_cost += sum(guide_list[word_span[0]: word_span[-1]])

            masked_indexes.sort()
            tokens_to_mask = []

            for masked_index in masked_indexes:
                # Converts word index to token index
                # Sums 1 to compensate for CLS start token
                offset = sum(guide_list[:masked_index[0]]) + 1
                mask_type = masked_index[1]
                try:
                    tokens_to_mask += [(offset + k, mask_type) for k in range(guide_list[masked_index[0]].item())]
                except:
                    breakpoint()

            masked_tokens = deepcopy(toks_batch[batch])


            for (idx, mask_type) in tokens_to_mask:

                if mask_type == "MASK":
                    token_to_append = self.MASK_TOKEN_ID
                elif mask_type == "SUB":
                    token_to_append = self.rng.choice(self.possible_subs, 1)[0]
                elif mask_type == "DONT":
                    token_to_append = masked_tokens[idx]
                else:
                    raise Exception("No valid masking_type")

                masked_tokens[idx] = token_to_append

            #masked_tokens = [
            #    self.MASK_TOKEN_ID if idx in tokens_to_mask else token_id for idx, token_id in enumerate(toks_batch[batch])
            #]

            if not self.model_processor.tokenizer.sep_token_id in masked_tokens:
                raise Exception("No sep token detected in sequence!")

            masked_tokens_batch.append(masked_tokens)

        return torch.stack(masked_tokens_batch)


    def get_alignment_labels(self, processed_data, masked_tokens_batch, masked_patches_list):
        pass

    def preprocess():
        pass


if __name__ == '__main__':
    from new_processor import BertimbauLayoutLMv3Processor
    from transformers import BertTokenizer, LayoutLMv3Processor
    from tqdm import tqdm

    lmv3_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    lmv3_processor.feature_extractor.apply_ocr = False

    model_processor = BertimbauLayoutLMv3Processor(
        layoutlmv3_processor=lmv3_processor,
        bertimbau_tokenizer=BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
    )

    main_path = Path("/home/luckagianvechio/Documents/Material Estudo TCC/IIT CDIP/images.a.a/imagesa/a/a")
    a_path = Path("a/")
    path_to_image = main_path / a_path / "aaa0a000/92464841_4842.tif"

    processor = PreProcessor(model_processor=model_processor)

    text_bbox, shape, image = processor.get_image_ocr(path_to_image)
    processed_data = processor.get_processed_data(text_bbox, shape, image)
    #print(processed_data)
    #print("\n\n")
    for k in range(1):
        try:
            masks = processor.mask_tokens(processed_data)
        except Exception as e:
            print(e)
            #print(traceback.format_exc(e))

    #print("\n\n")
    #print(masks)
    #print(processor.model_processor.tokenizer.decode(processed_data["input_ids"][0]))
    #print("\n\n")
    #print(processor.model_processor.tokenizer.decode(masks[0]))
    mask = processor._get_mask_index()
    for i in range(0, 16):
        for j in range(0, 16):
            if (i, j) in mask:
                print(".", end="")
            else:
                print("%", end="")
            print(" ", end = "")
        print()

    breakpoint()
