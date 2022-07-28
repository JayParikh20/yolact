import json
import operator
import os.path
from collections import defaultdict
import random

import cv2
import torch
import torch.backends.cudnn as cudnn

from data import COLORS
from data import cfg, set_cfg
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
from utils.augmentations import FastBaseTransform
from utils.functions import SavePath
from yolact import Yolact

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from pycocotools.coco import COCO
import seaborn as sn
import pandas as pd
from pdflatex import PDFLaTeX

model_name = "yolact_plus_resnet50_94_130000.pth"
dataset_path = r"D:\NoSpaceFolder\Projects\EvTek\dataset\coco\scaled\2021-07-27_14-50-49_mixed_set_1\550_550"
model_dataset = "2021-10-14-mrcnn_40_classes_only_new_belt_adapt_resize"
test_dataset = "2021-07-27_14-50-49_mixed_set_1"

top_k = 25  # Max number of objects to be detected
mask_iou_threshold = 0.7  # Mask IOU threshold
center_threshold = 15  # Threshold distance between two centers of bbox to detect them as same objects

ih = 2448
iw = 2048
scaleh = 550 / ih
scalew = 550 / iw

ih_scaled = round(ih * scaleh)
iw_scaled = round(iw * scalew)

weights_path = os.path.join("weights", model_name)
json_path = os.path.join(dataset_path, "annotations", "data.json")
images_path = os.path.join(dataset_path, "images")

color_cache = defaultdict(lambda: {})


class Classification:
    def __init__(self, tp, fp, fn):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __add__(self, other):
        return Classification(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def format_boxes(bboxes):
    for box in bboxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        width = x2 - x1
        height = y2 - y1
        box[0], box[1], box[2], box[3] = x1, y1, width, height
    return bboxes


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=True, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=0.9)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < 0.9:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1

    text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

    img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    # Draw the text on the CPU
    text_pt = (4, text_h + 2)
    text_color = [255, 255, 255]

    cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    # bboxes = format_boxes([boxes[i, :] for i in reversed(range(num_dets_to_consider))])
    # features = encoder(img_numpy, bboxes)
    # class_names = [cfg.dataset.class_names[classes[cid]] for cid in reversed(range(num_dets_to_consider))]
    # detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, class_names, features)]

    # boxs = np.array([d.tlwh for d in detections])
    # scores = np.array([d.confidence for d in detections])
    # classes = np.array([d.class_name for d in detections])
    #
    # cmap = plt.get_cmap('tab20b')
    # colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # tracker.predict()
    # tracker.update(detections)
    #
    # # update tracks
    # for track in tracker.tracks:
    #     if not track.is_confirmed() or track.time_since_update > 1:
    #         continue
    #     bbox = track.to_tlbr()
    #     class_name = track.get_class()
    #
    #     # draw bbox on screen
    #     color = colors[int(track.track_id) % len(colors)]
    #     color = [i * 255 for i in color]
    #     cv2.rectangle(img_numpy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    #     cv2.rectangle(img_numpy, (int(bbox[0]), int(bbox[1] - 80)), (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 22, int(bbox[1])), color, -1)
    #     cv2.putText(img_numpy, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 1, (255, 255, 255), 2)

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]

        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        _class = cfg.dataset.class_names[classes[j]]
        text_str = '%s: %.2f' % (_class, score)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_numpy


def apply_mask(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    mask = np.stack((mask,) * 3, axis=-1)
    resultant = image * mask
    return resultant


def get_mask_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    return np.sum(intersection) / np.sum(union)


def initialize_dataset(net: Yolact):
    cudnn.benchmark = True

    model_classes = cfg.dataset.class_names

    missing_classes = []

    net = CustomDataParallel(net).cuda()

    with open(os.path.join(json_path), 'r') as f:
        data = json.load(f)
        test_dataset_names = [cat['name'] for cat in data['categories']]

        for index in range(len(test_dataset_names)):
            if test_dataset_names[index] not in model_classes:
                missing_classes.append(test_dataset_names[index])

        if missing_classes:
            print("\n\nMissing Test Classes in Model Config:", missing_classes)
            print("Missing Classes will be changed to \"Rest\"")

        total_objects = 0
        true_positive = 0
        matched_multiple = 0
        false_negative = 0
        false_positive = 0
        undetected_classes = {}
        mismatched_classes = {}
        object_classification = {}
        max_image_id = len(data['images'])
        confusion_matrix = np.zeros((len(model_classes), len(model_classes)))

        for img_id in range(max_image_id):
            file_name = data['images'][img_id]['file_name']
            image = cv2.imread(os.path.join(images_path, file_name), cv2.IMREAD_COLOR)

            annotation_bboxes = []
            annotation_classes = []
            annotation_masks = []

            for annotation in data['annotations']:
                if annotation['image_id'] == img_id + 1:
                    total_objects += 1
                    annotation_bboxes.append(annotation['bbox'])
                    _class = data['categories'][annotation['category_id'] - 1]['name']
                    if _class in missing_classes:
                        _class = "Rest"
                    annotation_classes.append(_class)

                    segmentation = annotation['segmentation'][0]
                    image_binary_mask = np.zeros((550, 550), np.uint8)
                    cv2.drawContours(image_binary_mask, [np.round(segmentation).reshape((len(segmentation) // 2, 2)).astype(np.int64)], -1, (255, 255, 255), -1)
                    annotation_masks.append(image_binary_mask)
            if not annotation_bboxes:
                continue

            torch_image = torch.from_numpy(image).cuda().float()
            batch = FastBaseTransform()(torch_image.unsqueeze(0))
            preds = net(batch)

            h, w, _ = torch_image.shape
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.9)
            idx = t[1].argsort(0, descending=True)[:top_k]
            pred_masks = t[3][idx]
            pred_classes, pred_scores, pred_bboxes = [x[idx].cpu().numpy() for x in t[:3]]
            pred_classes = [cfg.dataset.class_names[c] for c in pred_classes]
            pred_bboxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in pred_bboxes]
            pred_masks = pred_masks.cpu().detach().numpy()
            cfg.rescore_bbox = save

            for index, annotation_mask in enumerate(annotation_masks):
                tp = 0
                fp = 0

                for index2, pred_mask in enumerate(pred_masks):
                    annotation_center = np.array([annotation_bboxes[index][0] + annotation_bboxes[index][2] // 2, annotation_bboxes[index][1] + annotation_bboxes[index][3] // 2])
                    pred_center = np.array([pred_bboxes[index2][0] + pred_bboxes[index2][2] // 2, pred_bboxes[index2][1] + pred_bboxes[index2][3] // 2])
                    center_distance = np.linalg.norm(annotation_center - pred_center)
                    if get_mask_iou(annotation_mask, pred_mask) > mask_iou_threshold and center_distance < center_threshold:
                        confusion_matrix[model_classes.index(annotation_classes[index]), model_classes.index(pred_classes[index2])] += 1
                        if annotation_classes[index] == pred_classes[index2]:
                            tp += 1
                        else:
                            if center_distance < center_threshold:
                                fp += 1
                                if annotation_classes[index] not in mismatched_classes:
                                    mismatched_classes[annotation_classes[index]] = [pred_classes[index2]]
                                elif annotation_classes[index] in mismatched_classes and pred_classes[index2] not in mismatched_classes[annotation_classes[index]]:
                                    mismatched_classes[annotation_classes[index]].append(pred_classes[index2])

                _classification = Classification(0, 0, 0)

                if tp == 0 and fp == 0:
                    false_negative += 1
                    _classification.fn = 1
                    if annotation_classes[index] in undetected_classes:
                        undetected_classes[annotation_classes[index]] += 1
                    else:
                        undetected_classes[annotation_classes[index]] = 1
                elif tp == 0 and fp == 1:
                    false_positive += 1
                    _classification.fp = 1
                elif tp == 1 and fp == 0:
                    true_positive += 1
                    _classification.tp = 1
                elif tp > 1 or fp > 1:
                    matched_multiple += 1

                if annotation_classes[index] in object_classification:
                    object_classification[annotation_classes[index]] += _classification
                else:
                    object_classification[annotation_classes[index]] = _classification

        undetected_classes = dict(sorted(undetected_classes.items(), key=lambda item: item[1], reverse=True))

        print("\n\nMeta")
        print("Model Weight:", model_name)
        print("Model Dataset:", model_dataset)
        print("Test Dataset:", test_dataset)
        print("Number of Classes: ", len(model_classes))
        print("Number of Test Images", len(data['images']))

        print("\nTotal Objects:", total_objects)
        print("Matched Correctly:", true_positive, f"({np.round((true_positive / total_objects) * 100, decimals=2)}%)")
        print("Mis-Classified (FP) Objects:", false_positive, f"({np.round((false_positive / total_objects) * 100, decimals=2)}%)")
        print("Undetected (FN) Objects:", false_negative, f"({np.round((false_negative / total_objects) * 100, decimals=2)}%)")
        print("Matched Multiple Times:", matched_multiple)

        print("\nMis-Classified (FP) Classes:")
        for key, value in mismatched_classes.items():
            print(f"{key} sometimes detected as {', '.join(value)}")
        print("\nUndetected (FN) Classes:")
        for key, value in undetected_classes.items():
            print(f"{key}: x{value}")

        print("\n\nObject Classification:\n")
        # classification_dict = {'Class': ['precision', 'recall', 'f1', 'accuracy']}
        table_data = []
        head = ['Class', 'Precision', 'Recall', 'F1 score', 'Accuracy']
        for key, value in object_classification.items():
            precision = value.tp / float(value.tp + value.fp)
            recall = value.tp / float(value.tp + value.fn)
            f1 = (2 * precision * recall) / float(precision + recall)
            acc = value.tp / float(value.tp + value.fp + value.fn)
            # print(key, precision, recall, f1, acc)
            table_data.append([key, round(precision, 2), round(recall, 2), round(f1, 2), round(acc, 2)])

        print(tabulate(table_data, headers=head, tablefmt="github"))

        confusion_matrix_df = pd.DataFrame(confusion_matrix.astype(np.uint8), index=model_classes, columns=model_classes)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = sn.heatmap(confusion_matrix_df, annot=True, square=True, annot_kws={"size": 10}, fmt='d', cmap='YlGnBu', cbar=False, ax=ax)  # font size
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=11, rotation_mode='anchor', ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')

        outstr = "\\begin{tabular}{@{}lllll}\n"
        # first, write header
        outstr += "Class Name"
        for name in ['Precision', 'Recall', 'f1-score', 'Accuracy']:
            outstr += " & %s" % name
        outstr += "\\\\\n"
        for key, value in object_classification.items():
            outstr += key.replace("_", "\\_")
            precision = value.tp / float(value.tp + value.fp)
            recall = value.tp / float(value.tp + value.fn)
            f1 = (2 * precision * recall) / float(precision + recall)
            acc = value.tp / float(value.tp + value.fp + value.fn)
            outstr += " & %.3f" % round(precision, 3)
            outstr += " & %.3f" % round(recall, 3)
            outstr += " & %.3f" % round(f1, 3)
            outstr += " & %.3f" % round(acc, 3)
            outstr += "\\\\\n"
        outstr += "\\end{tabular}\n\n"

        with open('report.tex', 'w+') as outfile:
            outfile.write("\\documentclass[12pt, a4paper]{article}\n")
            outfile.write("\\usepackage[margin=2.0cm]{geometry}\n")
            outfile.write("\\usepackage{graphicx}\n")
            outfile.write("\\title{Evtek AI Evaluation Report}\n")
            outfile.write("\\date{\\today}\n")
            outfile.write("\\setlength{\\parindent}{0pt}\n")
            outfile.write("\\begin{document}\n")
            outfile.write("\\includegraphics[width=.3\\textwidth]{../assets/evtek_logo.png}\\\\\n\n")
            outfile.write("{\\Large{Evtek AI Evaluation Report \\\\\\today}}\n\n")

            outfile.write("\\vspace{1cm}\n\n")

            # outfile.write("\\begin{minipage}[t]{0.49\\textwidth}\n")
            # outfile.write("\\centering\n")
            # outfile.write("\\includegraphics[width=.15\\textwidth]{../assets/icon_search.png}\\\\\n")
            # outfile.write("\\vspace{.5cm}\n")
            # outfile.write("{\\Huge{%.2f\\%%}}\\\\\n" % (cnt_det_match_unique / cnt_det_total * 100.0))
            # outfile.write("Detection rate\\\\\n")
            # outfile.write("\\end{minipage}\n")
            #
            # outfile.write("\\begin{minipage}[t]{0.49\\textwidth}\n")
            # outfile.write("\\centering\n")
            # outfile.write("\\includegraphics[width=.15\\textwidth]{../assets/icon_analysis.png}\\\\\n")
            # outfile.write("\\vspace{.5cm}\n")
            # outfile.write("{\\Huge{%.2f\\%%}}\\\\\n" % (clfr['accuracy'] * 100.))
            # outfile.write("Classification accuracy\\\\\n")
            # outfile.write("\\end{minipage}\\\\\n")
            #
            # outfile.write("\\vspace{.5cm}\n\n")
            #
            # outfile.write("\\begin{minipage}[t]{0.49\\textwidth}\n")
            # outfile.write("\\centering\n")
            # outfile.write("\\includegraphics[width=.15\\textwidth]{../assets/icon_count.png}\\\\\n")
            # outfile.write("\\vspace{.5cm}\n")
            # outfile.write("{\\Huge{%.d}}\\\\\n" % (len(self.class_names)))
            # outfile.write("Classes\\\\\n")
            # outfile.write("\\end{minipage}\n")
            #
            # outfile.write("\\begin{minipage}[t]{0.49\\textwidth}\n")
            # outfile.write("\\centering\n")
            # outfile.write("\\includegraphics[width=.15\\textwidth]{../assets/icon_stopwatch.png}\\\\\n")
            # outfile.write("\\vspace{.5cm}\n")
            # outfile.write("{\\Huge{%.3fs}}\\\\\n" % (np.mean(times[1:])))
            # outfile.write("per image\\\\\n")
            # outfile.write("\\end{minipage}\\\\\n\n")

            outfile.write("\\vspace{.2cm}\n\n")

            outfile.write("\\begin{figure}[!h]\n")
            outfile.write("\\center")
            outfile.write("\\includegraphics[width=.8\linewidth]{./confmat.png}\n")
            outfile.write("\\end{figure}\n")

            outfile.write("\\newpage\n")

            outfile.write("\\textbf{Meta}\\\\\n")
            outfile.write("Model Weight: %s\\\\\n" % model_name.replace("_", "\\_"))
            outfile.write("Model Dataset: %s\\\\\n" % model_dataset.replace("_", "\\_"))
            outfile.write("Test Dataset: %s\\\\\n" % test_dataset.replace("_", "\\_"))
            outfile.write("Number of Classes: %d\\\\\n" % len(model_classes))  # first one is considered burn-in
            outfile.write("Number of Test Images: %d\n\n" % len(data['images']))

            outfile.write("\\vspace{.5cm}\n\n")

            outfile.write("\\textbf{Missing Test Classes in Model Config:}\\\\\n")
            outfile.write("%s\\\\\n" % "\\\\\n".join(missing_classes))
            outfile.write("Missing Classes will be changed to Rest")

            outfile.write("\\vspace{.5cm}\n\n")

            ### DETECTION REPORT

            outfile.write("\\textbf{Detection report}\\\\\n")
            outfile.write("\\begin{tabular}{@{}ll}\n")
            outfile.write("Total objects & %d\\\\\n" % total_objects)
            outfile.write("Matched Correctly & %d (%.2f\\%%)\\\\\n" % (true_positive, (true_positive / total_objects) * 100.0))
            outfile.write("Mis-Classified (FP) Objects & %d (%.2f\\%%)\\\\\n" % (false_positive, (false_positive / total_objects) * 100.0))
            outfile.write("Undetected (FN) Objects & %d (%.2f\\%%)\\\\\n" % (false_negative, (false_negative / total_objects) * 100.0))
            outfile.write("Matched Multiple Times & %d\\\\\n" % matched_multiple)
            outfile.write("\\end{tabular}\n\n")

            outfile.write("\\vspace{.5cm}\n\n")

            # Misclassified report
            outfile.write("\\textbf{Mis-Classified (FP) Classes}\\\\\n")
            mc = len(mismatched_classes.items()) - 1
            for index, (key, value) in enumerate(mismatched_classes.items()):
                if index == mc:
                    outfile.write("%s sometimes detected as %s\n" % (key, ', '.join(value)))
                else:
                    outfile.write("%s sometimes detected as %s\\\\\n" % (key, ', '.join(value)))
            outfile.write("\n")

            outfile.write("\\vspace{.5cm}\n\n")

            # Undetected report
            outfile.write("\\textbf{Undetected (FN) Classes}\\\\\n")
            udc = len(undetected_classes.items()) - 1
            for index, (key, value) in enumerate(undetected_classes.items()):
                # print(f"{key}: x{value}")
                if index == udc:
                    outfile.write("%s: %s\n" % (key, f"x{value}"))
                else:
                    outfile.write("%s: %s\\\\\n" % (key, f"x{value}"))
            outfile.write("\n")

            outfile.write("\\vspace{.5cm}\n\n")

            ### CLASSIFICATION REPORT
            outfile.write("\\newpage")
            outfile.write("\\textbf{Classification report}\\\\\n")
            outfile.write(outstr)

            outfile.write("\\newpage")
            outfile.write("\\end{document}\n")

    os.system("pdflatex report.tex")

    cv2.destroyAllWindows()


def track(net: Yolact):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False

    initialize_dataset(net)


if __name__ == '__main__':
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model_path = SavePath.from_str(weights_path)

    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    set_cfg(config)

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(weights_path)
    net.eval()
    print(' Done.')
    net = net.cuda()

    track(net)
