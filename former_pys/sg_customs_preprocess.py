import argparse
import json
import os

from PIL import Image
from transformers import AutoTokenizer
import numpy as np

def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


def convert(args):
    dataset = ['ACTMAX', 'ADAMP', 'AKRIBIS', 'ALPHA', 'AMPHENOLE', 'RUCKSWIRE',
           'CISCO', 'INFINEON', 'MICRON', 'SIIX', 'TI', 'TOSHIBA', 'NETAPP', 'KOMAGUSAM']

    for set in dataset:
        if not os.path.exists(os.path.join(args.output_dir, set)):
            os.mkdir(os.path.join(args.output_dir, set))

        with open(
            os.path.join(args.output_dir, set, args.data_split + ".txt.tmp"),
            "w",
            encoding="utf8",
        ) as fw, open(
            os.path.join(args.output_dir, set, args.data_split + "_box.txt.tmp"),
            "w",
            encoding="utf8",
        ) as fbw, open(
            os.path.join(args.output_dir, set, args.data_split + "_image.txt.tmp"),
            "w",
            encoding="utf8",
        ) as fiw:

            args.data_dir = os.path.join("Singapore Customs Dataset", set)

            for file in os.listdir(os.path.join(args.data_dir, 'train/json')):
                file_path = os.path.join(os.path.join(args.data_dir, 'train/json'), file)
                with open(file_path, "r", encoding="utf8") as f:
                    data = json.load(f)
                image_path = file_path.replace("json", "image")
                image_path = image_path[:-5] + "png"
                file_name = os.path.basename(image_path)
                image = Image.open(image_path)
                width, length = image.size

                for item in data['single'][0]['res']['items']:
                    label = str(item.keys())[12:-3]
                    words = item[label]['value']
                    bbox = item[label]['locations']
                    bbox = [bbox[0][0][0], bbox[0][0][1], bbox[0][2][0], bbox[0][2][1]]

                    if len(words) == 0 or words.strip() == '':
                        continue

                    if label == "ignore":
                        fw.write(words + "\tO\n")
                        fbw.write(
                            words
                            + "\t"
                            + bbox_string(bbox, width, length)
                            + "\n"
                        )
                        fiw.write(
                            words
                            + "\t"
                            + actual_bbox_string(bbox, width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                    else:
                        # if len(words) == 1:
                        fw.write(words + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words
                            + "\t"
                            + bbox_string(bbox, width, length)
                            + "\n"
                        )
                        fiw.write(
                            words
                            + "\t"
                            + actual_bbox_string(bbox, width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                        # else:
                        #     fw.write(words + "\tB-" + label.upper() + "\n")
                        #     fbw.write(
                        #         words
                        #         + "\t"
                        #         + bbox_string(bbox, width, length)
                        #         + "\n"
                        #     )
                        #     fiw.write(
                        #         words
                        #         + "\t"
                        #         + actual_bbox_string(bbox, width, length)
                        #         + "\t"
                        #         + file_name
                        #         + "\n"
                        #     )
                        #     for w in words[1:-1]:
                        #         fw.write(words + "\tI-" + label.upper() + "\n")
                        #         fbw.write(
                        #             words
                        #             + "\t"
                        #             + bbox_string(bbox, width, length)
                        #             + "\n"
                        #         )
                        #         fiw.write(
                        #             words
                        #             + "\t"
                        #             + actual_bbox_string(bbox, width, length)
                        #             + "\t"
                        #             + file_name
                        #             + "\n"
                        #         )
                        #     fw.write(words + "\tE-" + label.upper() + "\n")
                        #     fbw.write(
                        #         words
                        #         + "\t"
                        #         + bbox_string(bbox, width, length)
                        #         + "\n"
                        #     )
                        #     fiw.write(
                        #         words
                        #         + "\t"
                        #         + actual_bbox_string(bbox, width, length)
                        #         + "\t"
                        #         + file_name
                        #         + "\n"
                        #     )
                fw.write("\n")
                fbw.write("\n")
                fiw.write("\n")


def seg_file(file_path, tokenizer, max_len):
    subword_len_counter = 0
    output_path = file_path[:-4]

    with open(file_path, "r", encoding="utf8") as f_p, \
            open(output_path, "w", encoding="utf8") as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue

            token = line.split("\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


def seg(args):
    dataset = ['ACTMAX', 'ADAMP', 'AKRIBIS', 'ALPHA', 'AMPHENOLE', 'RUCKSWIRE',
           'CISCO', 'INFINEON', 'MICRON', 'SIIX', 'TI', 'TOSHIBA', 'NETAPP', 'KOMAGUSAM']

    for set in dataset:
        args.data_dir = os.path.join("output", set)

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, do_lower_case=True
        )
        seg_file(
            os.path.join('SG', args.data_split + ".txt.tmp"),
            tokenizer,
            args.max_len,
        )
        seg_file(
            os.path.join(args.data_dir, args.data_split + "_box.txt.tmp"),
            tokenizer,
            args.max_len,
        )
        seg_file(
            os.path.join(args.data_dir, args.data_split + "_image.txt.tmp"),
            tokenizer,
            args.max_len,
        )



# def clean_wrong_coordinate:
    # with open('SG/train_box.txt') as fb:
    #     blines = fb.readlines()
    # with open('SG/train.txt') as f:
    #     lines = f.readlines()
    # with open('SG/train_image.txt') as fi:
    #     ilines = fi.readlines()
    #
    # error = []
    # for i, line in enumerate(blines):
    #     if not line == '\n':
    #         bbox = line.split('\t')[1]
    #         ordinate = bbox.split(' ')
    #         x0 = int(ordinate[0])
    #         y0 = int(ordinate[1])
    #         x1 = int(ordinate[2])
    #         y1 = int(ordinate[3])
    #
    #         if x0 > x1:
    #             error.append(i)
    #             print(i, line)
    #         if y0 > y1:
    #             error.append(i)
    #             print(i, line)
    # print(i)
    #
    # for i in range(len(error)-1, -1, -1):
    #     id = error[i]
    #     del lines[id]
    #     del blines[id]
    #     del ilines[id]

    # with open('SG/test_box.txt', 'w') as fb:
    #     for bline in blines:
    #         fb.write(bline)
    # with open('SG/test.txt', 'w') as f:
    #     for line in lines:
    #         f.write(line)
    # with open('SG/train_image.txt', 'w') as fi:
    #     for iline in ilines:
    #         fi.write(iline)

    # print(len(error))

if __name__ == "__main__":
    import torch
    from bert_serving.client import BertClient

    a = torch.load('sentence_level_cached_test_layoutlm-base-uncased_512')
    # bc = BertClient()
    for i, example in enumerate(a):
        for j, item in enumerate(example.input_ids):
            if not isinstance(item, int):
                a[i].input_ids[j] = item[0]
                print(i)
            # if item == '[CLS]':
            #     a[i].input_ids[j] = bc.encode(['102'])
            # elif item == '[SEP]':
            #     a[i].input_ids[j] = bc.encode(['103'])
            # elif item == '[PAD]':
            #     a[i].input_ids[j] = bc.encode(['1'])
            # elif item == '[UNK]':
            #     a[i].input_ids[j] = bc.encode(['101'])
            # elif item == '[MASK]':
            #     a[i].input_ids[j] = bc.encode(['104'])
            # else:
            #     if isinstance(a[i].input_ids[j], int):
            #         continue
            #     else:
            #         a[i].input_ids[j] = bc.encode([item])
            #     print(i, j)

    torch.save(a, 'SG/embedding_cached_test_layoutlm-base-uncased_512' + str(i))
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_dir", type=str, default="Singapore Customs Dataset"
    # )
    # parser.add_argument("--data_split", type=str, default="train")
    # parser.add_argument("--output_dir", type=str, default="output")
    # parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    # parser.add_argument("--max_len", type=int, default=510)
    # args = parser.parse_args()
    #
    # convert(args)
    # seg(args)
