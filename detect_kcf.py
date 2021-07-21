"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import numpy

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

import multiprocessing

def tracker_process(input_queue, output_queue):
    trackers = {}
    while True:
        data = input_queue.get()
        if data is not None:
            if data['type'] == 'create_tracker':
                tracker = cv2.TrackerKCF.create()
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
        ):
    # print('device')
    # print(device)
    # device = ''
    OBJECT_DETECT_DELAY = 1
    # OBJECT_DETECT_CONFIDENCE_THRESHOLD = 0.7
    TRACKER_UPDATE_DELAY = 0
    TRACKER_FAIL_COUNT_THRESHOLD = 20
    POOL_COUNT = min(8, multiprocessing.cpu_count())
    COUNT_RIGHT = True
    INTERSECT_DELAY = 0.5

    total_passed_objects = 0
    vacant_tracker_id = 1

    # id (number): {bbox, last bbox, fail_count}
    tracker_data_list = {}

    last_detect_tick_count = None
    last_tracker_update_tick_count = None

    fps = 0

    input_queues = [multiprocessing.Queue() for x in range(POOL_COUNT) ]
    output_queues = [multiprocessing.Queue() for x in range(POOL_COUNT) ] 
    for i in range(POOL_COUNT):
        p = multiprocessing.Process(target=tracker_process, args=(input_queues[i], output_queues[i], ))
        p.daemon = True
        p.start()

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
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
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        tick_count = cv2.getTickCount()
        tick_freq = cv2.getTickFrequency()
        v_frame = im0s[0]        

        # numpy.set_printoptions(threshold=sys.maxsize)
        # with open('out.txt', 'w') as f:
        #     print( v_frame, file=f)  # Python 3.x
        # raise("stop")

        rows = v_frame.shape[0]
        cols = v_frame.shape[1]
        cv2.line(v_frame, (int(cols/2), 0), (int(cols/2), rows), (0, 0, 255), thickness=2)

        if last_detect_tick_count == None or (tick_count - last_detect_tick_count) / tick_freq > OBJECT_DETECT_DELAY:
            last_detect_tick_count = tick_count
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_sync()
            pred = model(img,
                     augment=augment,
                     visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            
                            t_x = int(xyxy[0])
                            t_y = int(xyxy[1])
                            t_right = int(xyxy[2])
                            t_bottom = int(xyxy[3])
                         
                            center_x = int((t_x + t_right) / 2)
                            center_y = int((t_y + t_bottom) / 2)

                            # check if there's already a tracker here
                            intersect_existing_trackeers = False
                            for tracker_id, tracker_data in tracker_data_list.items():
                                (x_t, y_t, w_t, h_t) = tracker_data['bbox']
                                if center_x > x_t and center_x < x_t + w_t and center_y > y_t and center_y < y_t + h_t:
                                    intersect_existing_trackeers = True
                                    break

                            # add new tracker for person
                            if not intersect_existing_trackeers:
                                track_bbox = (t_x, t_y, t_right - t_x, t_bottom - t_y)

                                # find least busy process
                                for iq in input_queues:
                                    iq.put({'type': 'get_tracker_count'})
                                process_data = []
                                for j in range(len(output_queues)):
                                    d = output_queues[j].get()
                                    if 'tracker_count' in d:
                                        process_data.append({'idx': j, 'tracker_count': d['tracker_count']})
                                    else:
                                        raise Exception('Subprocess wrong answer')
                                process_data = sorted(process_data, key=lambda k: k['tracker_count']) 
                                least_busy_process_idx = process_data[0]['idx']

                                # find vacant tracking ID
                                # vacant_tracker_id = vacant_tracker_id + 1
                                vacant_tracker_id = 0
                                while True:
                                    if vacant_tracker_id in tracker_data_list:
                                        vacant_tracker_id += 1
                                        continue
                                    else:
                                        break

                                # print('least_busy_process_idx' + str(least_busy_process_idx))
                                # add tracking data to main process and subprocess
                                tracker_data_list[vacant_tracker_id] = {'bbox': track_bbox, 'last_bbox': track_bbox, 'fail_count': 0, 'last_intersect_tick_count': 0}
                                input_queues[least_busy_process_idx].put({'type': 'create_tracker', 'frame': v_frame, 'init_bbox': track_bbox, 'id': vacant_tracker_id })

                    
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

        
        # Update tracker
        if last_tracker_update_tick_count == None or (tick_count - last_tracker_update_tick_count) / tick_freq > TRACKER_UPDATE_DELAY:
            last_tracker_update_tick_count = tick_count

            for i in range(len(input_queues)):
                input_queues[i].put({'type':'update_tracker', 'frame': v_frame})

            for i in range(len(output_queues)):
                # TODO timeout?

                # id, bbox, last_bbox, fail_count
                res = output_queues[i].get()
                if 'type' in res and res['type'] == 'update_tracker':
                    for key_id, track_result in res['data'].items():
                        if key_id in tracker_data_list:
                            if track_result['track_ok'] is True:
                                tracker_data_list[key_id]['fail_count'] = 0
                                tracker_data_list[key_id]['last_bbox'] = tracker_data_list[key_id]['bbox'] 
                                tracker_data_list[key_id]['bbox'] = track_result['updated_bbox']
                            else: 
                                tracker_data_list[key_id]['fail_count'] = tracker_data_list[key_id]['fail_count'] + 1
                                
                        else:
                            raise Exception('Subprocess have tracking ID not tracked by main process')
                else:
                    raise Exception('Subprocess wrong answer')

        # Remove failed tracker
        for tracker_id in list(tracker_data_list):
        # for i in reversed(range(len(tracker_data_list))):
            if tracker_data_list[tracker_id]['fail_count'] > TRACKER_FAIL_COUNT_THRESHOLD:
                for iq in input_queues:
                    iq.put({'type': 'remove_tracker', 'id': tracker_id})
                del tracker_data_list[tracker_id]

        # Render tracker
        for tracker_id, tracker_data in tracker_data_list.items():
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
                in_intersect_delay = (tick_count - tracker_data['last_intersect_tick_count']) / tick_freq < INTERSECT_DELAY
                
                if intersects_center:
                    tracker_data['last_intersect_tick_count'] = tick_count
                    if not in_intersect_delay and not prev_intersects_center:
                        if (x + w/2) > (prev_x + prev_w/2):
                            total_passed_objects += 1 if COUNT_RIGHT else -1
                        else:
                            total_passed_objects += -1 if COUNT_RIGHT else 1
                
                if in_intersect_delay:
                    color = (255,0,0)

            cv2.rectangle(v_frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(v_frame, str(tracker_id), (int(x + 5) , int(y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        # Display
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(v_frame,'Count: ' + str(total_passed_objects) + ' Track: ' + str(len(tracker_data_list)),(10, 40),font, 1.5,(0, 255, 255),2)
        cv2.putText(v_frame, "FPS : " + str(int(fps)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1);
        # cv2.imshow('object counting', cv2.resize(v_frame, None, None, fx=2, fy=2))
        cv2.imshow('object counting', v_frame)
        cv2.waitKey(1)  # 1 millisecond

        fps = tick_freq / (cv2.getTickCount() - tick_count);

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


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


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)