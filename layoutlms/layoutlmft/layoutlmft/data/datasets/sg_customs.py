# coding=utf-8

import json
import os
import datasets
import shutil

from layoutlmft.data.utils import load_image, normalize_bbox_sg_customs


logger = datasets.logging.get_logger(__name__)


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="sg_customs", version=datasets.Version("1.0.0"), description="SG-Customs dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                    "O",
                                    "S-INVCOMMODITY.PARTNUMBER",
                                    "S-INVCOMMODITY.ITEMNO",
                                    "S-INVCOMMODITY.COO",
                                    "S-INVCOMMODITY.DESC",
                                    "S-INVCOMMODITY.UNIT",
                                    "S-INVCOMMODITY.HSCODE",
                                    "S-INVCOMMODITY.GW",
                                    "S-INVCOMMODITY.PRICE",
                                    "S-INVCOMMODITY.QTY",
                                    "S-INVCOMMODITY.BOXNUMBER",
                                    "S-INVCOMMODITY.TOTAL",
                                    "S-INVSHIPPER",
                                    "S-INVCONSIGNEE",
                                    "S-INVNO",
                                    "S-INVPAGE",
                                    "S-INVDATE",
                                    "S-INVTERMTYPE",
                                    "S-INVCURRENCY",
                                    "S-INVQTYUOM",
                                    "S-INVTOTALGW",
                                    "S-INVTOTALNW",
                                    "S-INVTOTALQTY",
                                    "S-INVTOTAL",
                                    "S-INVWTUNIT"
                                ]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        
        datasets_path = f'{downloaded_file}/dataset/'

        proj_data_path = os.path.join(datasets_path.split(".cache")[0], '../content/thesis/layoutlms/layoutlmft/data')
        # proj_data_path = os.path.join(f'{downloaded_file}'.split(".cache")[0], 'thesis/thesis/layoutlms/layoutlmft/data')
        dir_list = os.listdir(datasets_path)
        dir_list.remove('.DS_Store')
        for file in dir_list:
            shutil.rmtree(os.path.join(datasets_path, file))

        shutil.copytree(os.path.join(proj_data_path,'test'), os.path.join(datasets_path, 'test'))
        shutil.copytree(os.path.join(proj_data_path,'train'), os.path.join(datasets_path, 'train'))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/train/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{downloaded_file}/dataset/test/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "json")
        img_dir = os.path.join(filepath, "image")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path[:-4] + "png"
            image, size = load_image(image_path)
            for item in data["items"]:
                label = list(item.keys())[0]
                words = [item[label]["value"]]
                box = item[label]["locations"]
                if len(words) == 0:
                    continue
                if label == "ignore":
                    for w in words:
                        tokens.append(w)
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox_sg_customs(box, size))
                else:
                    tokens.append(words[0])
                    ner_tags.append("S-" + label.upper())
                    bboxes.append(normalize_bbox_sg_customs(box, size))
                    for w in words[1:]:
                        print("ERROR")
                        tokens.append(w)
                        ner_tags.append("S-" + label.upper())
                        bboxes.append(normalize_bbox_sg_customs(box, size))

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}
