import os
from transformers import AutoTokenizer


def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0][0] / width)))
        + " "
        + str(int(1000 * (box[0][1] / length)))
        + " "
        + str(int(1000 * (box[1][0] / width)))
        + " "
        + str(int(1000 * (box[1][1] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0][0])
        + " "
        + str(box[0][1])
        + " "
        + str(box[1][0])
        + " "
        + str(box[1][1])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


def convert(imagesize, bboxes, words, image_name):
    with open(
        os.path.join('data', "test.txt.tmp"), "w", encoding="utf8",
    ) as fw, open(
        os.path.join('data', "test_box.txt.tmp"), "w", encoding="utf8",
    ) as fbw, open(
        os.path.join('data', "test_image.txt.tmp"), "w", encoding="utf8",
    ) as fiw:
        width, length = imagesize[1], imagesize[0]
        for i, bbox in enumerate(bboxes):
            word = words[i]
            if len(word) == 0:
                continue
            fw.write(word + "\tO\n")
            fbw.write(word + "\t" + bbox_string(bbox, width, length) + "\n")
            fiw.write(word + "\t" + actual_bbox_string(bbox, width, length) 
                        + "\t" + image_name + "\n")
        fw.write("\n")
        fbw.write("\n")
        fiw.write("\n")

def seg_file(file_path, tokenizer, max_len):
    subword_len_counter = 0
    output_path = file_path[:-4]
    with open(file_path, "r", encoding="utf8") as f_p, open(
        output_path, "w", encoding="utf8"
    ) as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue
            token = line.split("\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


def seg():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    seg_file(os.path.join("data", "test" + ".txt.tmp"), tokenizer, 510)
    seg_file(os.path.join("data", "test" + "_box.txt.tmp"), tokenizer, 510)
    seg_file(os.path.join("data", "test" + "_image.txt.tmp"), tokenizer, 510)


if __name__ == "__main__":
    convert()
    seg()
