######################### Import for server
from pprint import pprint

import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


######################### Import for detector

# import argparse
import sys
import time
from pathlib import Path

# import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.augmentations import letterbox

import numpy as np

import multiprocessing

#########################

def tracker_process(input_queue, output_queue):
    trackers = {}
    while True:
        data = input_queue.get()
        if data is not None:
            if data['type'] == 'create_tracker':
                tracker = cv2.TrackerKCF.create()
                print(data)
                tracker.init(data['frame'], data['init_bbox'])
                trackers[data['id']] = { 'tracker': tracker }
            elif data['type'] == 'get_tracker_count':
                output_queue.put({'tracker_count': len(trackers)})
            elif data['type'] == 'update_tracker':
                output = {'type': 'update_tracker', 'data': {}}
                for tracker_id, tracker in trackers.items():
                    output['data'][tracker_id] = {}
                    track_ok, track_bbox = trackers[tracker_id]['tracker'].update(data['frame'])
                    if track_ok:
                        # Tracking success
                        output['data'][tracker_id]['track_ok'] = True
                        output['data'][tracker_id]['updated_bbox'] = [int(v) for v in track_bbox]
                    else:
                        output['data'][tracker_id]['track_ok'] = False
                output_queue.put(output)
            elif data['type'] == 'remove_tracker':
                if data['id'] in trackers:
                    del trackers[data['id']]
            else:
                raise Exception('unknown data from parent process')

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """
    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        print('=== video transform track init ===')

        # ##### init detector
        self.OBJECT_DETECT_DELAY = 1
        self.TRACKER_UPDATE_DELAY = 0
        self.TRACKER_FAIL_COUNT_THRESHOLD = 20
        self.POOL_COUNT = min(8, multiprocessing.cpu_count())
        self.COUNT_RIGHT = True
        self.INTERSECT_DELAY = 0.5

        self.total_passed_objects = 0
        self.vacant_tracker_id = 1

        # id (number): {bbox, last bbox, fail_count}
        self.tracker_data_list = {}

        self.last_detect_tick_count = None
        self.last_tracker_update_tick_count = None

        self.fps = 0

        self.input_queues = [multiprocessing.Queue() for x in range(self.POOL_COUNT) ]
        self.output_queues = [multiprocessing.Queue() for x in range(self.POOL_COUNT) ] 
        for i in range(self.POOL_COUNT):
            p = multiprocessing.Process(target=tracker_process, args=(self.input_queues[i], self.output_queues[i], ))
            p.daemon = True
            p.start()

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    async def recv(self):
        frame = await self.track.recv()
        v_frame = frame.to_ndarray(format="bgr24")

        # Letterbox
        # img0 = self.imgs.copy()
        v_frame = letterbox(v_frame, detect_args['imgsz'], stride=32)[0]

        # Stack
        v_frame = np.stack(v_frame, 0)

        # Convert
        v_frame = v_frame[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        v_frame = np.ascontiguousarray(v_frame)

        ######################

        # # Padded resize
        # v_frame = letterbox(v_frame, detect_args['imgsz'], stride=32)[0]
        # d_frame = v_frame.copy()

        # # Convert
        # v_frame = v_frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # # v_frame = np.ascontiguousarray(v_frame)

        tick_count = cv2.getTickCount()
        tick_freq = cv2.getTickFrequency()
        rows = v_frame.shape[0]
        cols = v_frame.shape[1]


        np.set_printoptions(threshold=sys.maxsize)
        with open('out_server.txt', 'w') as f:
            print(v_frame, file=f)  # Python 3.x
        raise("stop")


        if self.last_detect_tick_count == None or (tick_count - self.last_detect_tick_count) / tick_freq > self.OBJECT_DETECT_DELAY:
            self.last_detect_tick_count = tick_count
            
            img = torch.from_numpy(v_frame).to(detect_args['device'])
            img = img.half() if detect_args['half'] else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = detect_args['model'](img, augment=detect_args['augment'], visualize=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, detect_args['conf_thres'], detect_args['iou_thres'], detect_args['classes'], detect_args['agnostic_nms'], max_det=detect_args['max_det'])
            t2 = time_synchronized()

            # Apply Classifier
            if detect_args['classify']:
                pred = apply_classifier(pred, detect_args['modelc'], img, v_frame)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], f'{i}: ', v_frame.copy(), dataset.count
                # else:
                # p, s, im0, frame = path, '', v_frame.copy(), getattr(dataset, 'frame', 0)

                # p = Path(p)  # to Path
                s = ''
                s += '%gx%g ' % img.shape[2:]  # print string
                # gn = torch.tensor(im0.shape, requires_grad=False)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], v_frame.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {detect_args['names'][int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        d_names = detect_args['names']
                        label = None if detect_args['hide_labels'] else (detect_args['names'][c] if detect_args['hide_conf'] else f'{d_names[c]} {conf:.2f}')
                        # plot_one_box(xyxy, d_frame, label=label, color=colors(c, True), line_thickness=detect_args['line_thickness'])

                        t_x = int(xyxy[0])
                        t_y = int(xyxy[1])
                        t_right = int(xyxy[2])
                        t_bottom = int(xyxy[3])
                        
                        center_x = int((t_x + t_right) / 2)
                        center_y = int((t_y + t_bottom) / 2)

                        # Check if there's already a tracker here
                        intersect_existing_trackeers = False
                        for tracker_id, tracker_data in self.tracker_data_list.items():
                            (x_t, y_t, w_t, h_t) = tracker_data['bbox']
                            if center_x > x_t and center_x < x_t + w_t and center_y > y_t and center_y < y_t + h_t:
                                intersect_existing_trackeers = True
                                break

                        # Add new tracker for person
                        if not intersect_existing_trackeers:
                            track_bbox = (t_x, t_y, t_right - t_x, t_bottom - t_y)

                            # find least busy process
                            for iq in self.input_queues:
                                iq.put({'type': 'get_tracker_count'})
                            process_data = []
                            for j in range(len(self.output_queues)):
                                d = self.output_queues[j].get()
                                if 'tracker_count' in d:
                                    process_data.append({'idx': j, 'tracker_count': d['tracker_count']})
                                else:
                                    raise Exception('Subprocess wrong answer')
                            process_data = sorted(process_data, key=lambda k: k['tracker_count']) 
                            least_busy_process_idx = process_data[0]['idx']

                            # find vacant tracking ID
                            # self.vacant_tracker_id = self.vacant_tracker_id + 1
                            self.vacant_tracker_id = 0
                            while True:
                                if self.vacant_tracker_id in self.tracker_data_list:
                                    self.vacant_tracker_id += 1
                                    continue
                                else:
                                    break

                            # print('least_busy_process_idx' + str(least_busy_process_idx))
                            # add tracking data to main process and subprocess
                            # print(v_frame)
                            # print(track_bbox)
                            # self.tracker_data_list[self.vacant_tracker_id] = {'bbox': track_bbox, 'last_bbox': track_bbox, 'fail_count': 0, 'last_intersect_tick_count': 0}
                            # self.input_queues[least_busy_process_idx].put({'type': 'create_tracker', 'frame': v_frame, 'init_bbox': track_bbox, 'id': self.vacant_tracker_id })

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Update tracker
        if self.last_tracker_update_tick_count == None or (tick_count - self.last_tracker_update_tick_count) / tick_freq > self.TRACKER_UPDATE_DELAY:
            self.last_tracker_update_tick_count = tick_count

            for i in range(len(self.input_queues)):
                self.input_queues[i].put({'type':'update_tracker', 'frame': v_frame})

            for i in range(len(self.output_queues)):
                # TODO timeout?

                # id, bbox, last_bbox, fail_count
                res = self.output_queues[i].get()
                if 'type' in res and res['type'] == 'update_tracker':
                    for key_id, track_result in res['data'].items():
                        if key_id in self.tracker_data_list:
                            if track_result['track_ok'] is True:
                                self.tracker_data_list[key_id]['fail_count'] = 0
                                self.tracker_data_list[key_id]['last_bbox'] = self.tracker_data_list[key_id]['bbox'] 
                                self.tracker_data_list[key_id]['bbox'] = track_result['updated_bbox']
                            else: 
                                self.tracker_data_list[key_id]['fail_count'] = self.tracker_data_list[key_id]['fail_count'] + 1
                                
                        else:
                            raise Exception('Subprocess have tracking ID not tracked by main process')
                else:
                    raise Exception('Subprocess wrong answer')

        # Remove failed tracker
        for tracker_id in list(self.tracker_data_list):
        # for i in reversed(range(len(self.tracker_data_list))):
            if self.tracker_data_list[tracker_id]['fail_count'] > self.TRACKER_FAIL_COUNT_THRESHOLD:
                for iq in self.input_queues:
                    iq.put({'type': 'remove_tracker', 'id': tracker_id})
                del self.tracker_data_list[tracker_id]

        # Render tracker
        for tracker_id, tracker_data in self.tracker_data_list.items():
            (x, y, w, h) = tracker_data['bbox']
            if tracker_data['fail_count'] > 0:
                color = (0,0,255)
            else:
                color = (0,255,0)

                # Detect if track box intersect center of screen
                (prev_x, prev_y, prev_w, prev_h) = tracker_data['last_bbox']
                center_x = cols/2
                intersects_center = center_x > x and center_x < x + w
                prev_intersects_center = center_x > prev_x and center_x < prev_x + prev_w
                in_intersect_delay = (tick_count - tracker_data['last_intersect_tick_count']) / tick_freq < self.INTERSECT_DELAY
                
                if intersects_center:
                    tracker_data['last_intersect_tick_count'] = tick_count
                    if not in_intersect_delay and not prev_intersects_center:
                        if (x + w/2) > (prev_x + prev_w/2):
                            self.total_passed_objects += 1 if self.COUNT_RIGHT else -1
                        else:
                            self.total_passed_objects += -1 if self.COUNT_RIGHT else 1
                
                if in_intersect_delay:
                    color = (255,0,0)

            cv2.rectangle(d_frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(d_frame, str(tracker_id), (int(x + 5) , int(y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        cv2.line(d_frame, (int(d_frame.shape[0]/2), 0), (int(d_frame.shape[0]/2), d_frame.shape[1]), (0, 0, 255), thickness=2)
        new_frame = VideoFrame.from_ndarray(d_frame, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

async def index(request):
    content = open(os.path.join(ROOT, "server/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "server/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "server/demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        **kwargs
        ):

    #################################### setup detection model

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = False
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    modelc = None
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    global detect_args
    detect_args = {}
    detect_args['weights'] = weights
    detect_args['source'] = source
    detect_args['imgsz'] = imgsz
    detect_args['conf_thres'] = conf_thres
    detect_args['iou_thres'] = iou_thres
    detect_args['max_det'] = max_det
    detect_args['device'] = device
    detect_args['view_img'] = view_img
    detect_args['save_txt'] = save_txt
    detect_args['save_conf'] = save_conf
    detect_args['save_crop'] = save_crop
    detect_args['nosave'] = nosave
    detect_args['classes'] = classes
    detect_args['agnostic_nms'] = agnostic_nms
    detect_args['augment'] = augment
    detect_args['visualize'] = visualize
    detect_args['update'] = update
    detect_args['project'] = project
    detect_args['name'] = name
    detect_args['exist_ok'] = exist_ok
    detect_args['line_thickness'] = line_thickness
    detect_args['hide_labels'] = hide_labels
    detect_args['hide_conf'] = hide_conf
    detect_args['half'] = half

    detect_args['classify'] = classify
    detect_args['modelc'] = modelc
    detect_args['model'] = model
    detect_args['webcam'] = webcam
    detect_args['names'] = names




def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":


    #################################### detect args
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    #################################### server args
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")



    args = parser.parse_args()



    main(args)

    ############################ setup server    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
