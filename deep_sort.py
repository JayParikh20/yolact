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
        cv2.rectangle(img_numpy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 4)
        cv2.rectangle(img_numpy, (int(bbox[0]), int(bbox[1] - 80)), (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 38, int(bbox[1])), color, -1)
        cv2.putText(img_numpy, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 2, (255, 255, 255), 4)

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
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    # frame_times = MovingAverage(100)
    # frame_time_target = 1 / target_fps
    # running = True
    # fps_str = ''
    # vid_done = False
    # frames_displayed = 0
    #
    # def cleanup_and_exit():
    #     print()
    #     pool.terminate()
    #     vid.release()
    #     cv2.destroyAllWindows()
    #     exit()
    #
    # def get_next_frame(vid):
    #     frames = []
    #     for idx in range(1):
    #         frame = vid.read()[1]
    #         if frame is None:
    #             return frames
    #         frames.append(frame)
    #     return frames
    #
    # def transform_frame(frames):
    #     with torch.no_grad():
    #         frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
    #         return frames, transform(torch.stack(frames, 0))
    #
    # def eval_network(inp):
    #     with torch.no_grad():
    #         frames, imgs = inp
    #         num_extra = 0
    #         while imgs.size(0) < 1:
    #             imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
    #             num_extra += 1
    #         out = net(imgs)
    #         if num_extra > 0:
    #             out = out[:-num_extra]
    #         return frames, out
    #
    # def prep_frame(inp, fps_str):
    #     with torch.no_grad():
    #         frame, preds = inp
    #         return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)
    #
    # frame_buffer = Queue()
    # video_fps = 0
    #
    # # All this timing code to make sure that
    # def play_video():
    #     try:
    #         nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done
    #
    #         video_frame_times = MovingAverage(100)
    #         frame_time_stabilizer = frame_time_target
    #         last_time = None
    #         stabilizer_step = 0.0005
    #         progress_bar = ProgressBar(30, num_frames)
    #
    #         while running:
    #             frame_time_start = time.time()
    #
    #             if not frame_buffer.empty():
    #                 next_time = time.time()
    #                 if last_time is not None:
    #                     video_frame_times.add(next_time - last_time)
    #                     video_fps = 1 / video_frame_times.get_avg()
    #
    #                 cv2.imshow(video_path, frame_buffer.get())
    #
    #                 frames_displayed += 1
    #                 last_time = next_time
    #
    #             # This is split because you don't want savevideo to require cv2 display functionality (see #197)
    #             if cv2.waitKey(1) == 27:
    #                 # Press Escape to close
    #                 running = False
    #             if not (frames_displayed < num_frames):
    #                 running = False
    #
    #             if not vid_done:
    #                 buffer_size = frame_buffer.qsize()
    #                 if buffer_size < 1:
    #                     frame_time_stabilizer += stabilizer_step
    #                 elif buffer_size > 1:
    #                     frame_time_stabilizer -= stabilizer_step
    #                     if frame_time_stabilizer < 0:
    #                         frame_time_stabilizer = 0
    #
    #                 new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
    #             else:
    #                 new_target = frame_time_target
    #
    #             next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
    #             target_time = frame_time_start + next_frame_target - 0.001  # Let's just subtract a millisecond to be safe
    #
    #             while time.time() < target_time:
    #                 time.sleep(0.001)
    #
    #     except:
    #         # See issue #197 for why this is necessary
    #         import traceback
    #         traceback.print_exc()
    #
    # extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])
    #
    # # Prime the network on the first frame because I do some thread unsafe things otherwise
    # print('Initializing model... ', end='')
    # first_batch = eval_network(transform_frame(get_next_frame(vid)))
    # print('Done.')
    #
    # # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    # sequence = [prep_frame, eval_network, transform_frame]
    # pool = ThreadPool(processes=len(sequence) + 1 + 2)
    # pool.apply_async(play_video)
    # active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]
    #
    # print()
    # print('Press Escape to close.')
    # try:
    #     while vid.isOpened() and running:
    #         # Hard limit on frames in buffer so we don't run out of memory >.>
    #         while frame_buffer.qsize() > 100:
    #             time.sleep(0.001)
    #
    #         start_time = time.time()
    #
    #         # Start loading the next frames from the disk
    #         if not vid_done:
    #             next_frames = pool.apply_async(get_next_frame, args=(vid,))
    #         else:
    #             next_frames = None
    #
    #         if not (vid_done and len(active_frames) == 0):
    #             # For each frame in our active processing queue, dispatch a job
    #             # for that frame using the current function in the sequence
    #             for frame in active_frames:
    #                 _args = [frame['value']]
    #                 if frame['idx'] == 0:
    #                     _args.append(fps_str)
    #                 frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
    #
    #             # For each frame whose job was the last in the sequence (i.e. for all final outputs)
    #             for frame in active_frames:
    #                 if frame['idx'] == 0:
    #                     frame_buffer.put(frame['value'].get())
    #
    #             # Remove the finished frames from the processing queue
    #             active_frames = [x for x in active_frames if x['idx'] > 0]
    #
    #             # Finish evaluating every frame in the processing queue and advanced their position in the sequence
    #             for frame in list(reversed(active_frames)):
    #                 frame['value'] = frame['value'].get()
    #                 frame['idx'] -= 1
    #
    #                 if frame['idx'] == 0:
    #                     # Split this up into individual threads for prep_frame since it doesn't support batch size
    #                     active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
    #                     frame['value'] = extract_frame(frame['value'], 0)
    #
    #             # Finish loading in the next frames and add them to the processing queue
    #             if next_frames is not None:
    #                 frames = next_frames.get()
    #                 if len(frames) == 0:
    #                     vid_done = True
    #                 else:
    #                     active_frames.append({'value': frames, 'idx': len(sequence) - 1})
    #
    #             # Compute FPS
    #             frame_times.add(time.time() - start_time)
    #             fps = 1 / frame_times.get_avg()
    #         else:
    #             fps = 0
    #
    #         fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (fps, video_fps, frame_buffer.qsize())
    #         print('\r' + fps_str + '    ', end='')
    #
    # except KeyboardInterrupt:
    #     print('\nStopping...')
    #
    # cleanup_and_exit()
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
