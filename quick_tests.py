from transformers import LayoutLMv3Processor
from pathlib import Path
from os import listdir
from ocr_tools import get_ocr_word_box_list, preprocess_image, resize_image, read_image, normalize_bbox
from transformers import LayoutLMv3Config, LayoutLMv3Model
import json

lmv3_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
lmv3_processor.feature_extractor.apply_ocr = False

main_path = Path("/home/luckagianvechio/Documents/Material Estudo TCC/IIT CDIP/images.a.a/imagesa/a/a")
a_path = Path("a/")
a_img_folder_path = [main_path / a_path / Path(pt) for pt in listdir(main_path / a_path)]
a_img_path = []
for pt in a_img_folder_path:
    files = listdir(pt)
    for file in files:
        if not file.split(".")[1] == "xml":
            a_img_path.append(pt / file)



image = read_image(a_img_path[0])
text_boxes, shape = get_ocr_word_box_list(a_img_path[0])
words = [k["text"] for k in text_boxes]
boxes = [normalize_bbox(k["bbox"], shape[0], shape[1]) for k in text_boxes]

processed = lmv3_processor(
    image,
    words,
    boxes=boxes,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

with open("/home/luckagianvechio/Documents/Material Estudo TCC/code/layoutlmv3/config.json", "r") as jeiso:
    configuration = LayoutLMv3Config(**json.load(jeiso))

# Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
model = LayoutLMv3Model(configuration)

# Accessing the model configuration
configuration = model.config

model(**processed)