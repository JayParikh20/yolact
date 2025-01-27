import os.path
from collections import defaultdict

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

weights_path = os.path.join("weights", "yolact_plus_resnet50_94_130000.pth")
video_path = r"test.mp4"
top_k = 20

color_cache = defaultdict(lambda: {})


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


max_cosine_dst = 0.05
nn_budget = None
nms_max_overlap = 1.0
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_dst, nn_budget)
tracker = Tracker(metric)
encoder = gdet.create_box_encoder(r"mars-small128.pb", batch_size=1)


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

    bboxes = format_boxes([boxes[i, :] for i in reversed(range(num_dets_to_consider))])
    features = encoder(img_numpy, bboxes)
    class_names = [cfg.dataset.class_names[classes[cid]] for cid in reversed(range(num_dets_to_consider))]
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, class_names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    tracker.predict()
    tracker.update(detections)

    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()

        # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(img_numpy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img_numpy, (int(bbox[0]), int(bbox[1] - 80)), (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 22, int(bbox[1])), color, -1)
        cv2.putText(img_numpy, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 1, (255, 255, 255), 2)

    # for j in reversed(range(num_dets_to_consider)):
    #     x1, y1, x2, y2 = boxes[j, :]
    #     color = get_color(j)
    #     score = scores[j]
    #
    #     cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
    #
    #     _class = cfg.dataset.class_names[classes[j]]
    #     text_str = '%s: %.2f' % (_class, score)
    #
    #     font_face = cv2.FONT_HERSHEY_DUPLEX
    #     font_scale = 0.6
    #     font_thickness = 1
    #
    #     text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    #
    #     text_pt = (x1, y1 - 3)
    #     text_color = [255, 255, 255]
    #
    #     cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
    #     cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_numpy


def initialize_video(net: Yolact):
    cudnn.benchmark = True

    vid = cv2.VideoCapture(video_path)
    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("deep_sort.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    net = CustomDataParallel(net).cuda()

    while True:
        frame = vid.read()[1]
        if frame is None or cv2.waitKey(1) == 27:
            break
        frame = torch.from_numpy(frame).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        # print(preds)
        img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        # cv2.imshow("windows", img_numpy)
        out.write(img_numpy)

    vid.release()
    out.release()
    cv2.destroyAllWindows()


def track(net: Yolact):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False

    initialize_video(net)


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
