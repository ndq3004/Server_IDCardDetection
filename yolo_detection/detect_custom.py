import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys
sys.path.insert(1, '/home/quan/server/yolo_detection')

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImagesCustom
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from dotmap import DotMap

class YoloPredictCustom:
    def __init__(self, modelName = None):
        self.modelName = modelName
        self.model = None
    def detect_custom_detect(self, save_img=False, imgCvt = None):
        opt = self.getopts()
        source, weights, view_img, save_txt, imgsz = opt['source'], opt['weights'], opt['view-img'], opt['save-txt'], opt['img-size']
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))
        # Directories
        save_dir = Path(increment_path(Path(opt['project']) / opt['name'], exist_ok=opt['exist-ok']))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        if self.model is not None:
            model = self.model
            print('Model loaded from cache!')
        else:
            print(weights)
            model = attempt_load(weights, map_location=device)  # load FP32 model
            self.model = model
            print('model loaded')
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            # dataset = LoadImages(source, img_size=imgsz)
            if imgCvt is None:
                print('imgCvt is not none!')
                dataset = LoadImages(source, img_size=imgsz)
            else:
                dataset = LoadImagesCustom(imgCvt, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt['augment'])[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=opt['classes'], agnostic=opt['agnostic-nms'])
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            # print(pred)
            # return pred
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = Path(path), '', im0s

                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # print('det', reversed(det)[:, :4].numpy())
                    return reversed(det)
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        print('*xyxy: ', *xyxy)
                        print('*xyxy: ', conf)
                        print('*xyxy: ', cls)
                        if True:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt['save-conf'] else (cls, *xywh)  # label format
                            # print(type(line))
                            # return line
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        print('save_path: ', save_path)
                        cv2.imwrite(save_path, im0)
                    elif dataset.mode == 'array':
                        cv2.imwrite(dataset.debugPath, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print('Done. (%.3fs)' % (time.time() - t0))


    def getopts(self):
        opt = {}
        if self.modelName is not None:
            opt['weights'] = '/home/quan/server/yolo_detection/models/' + self.modelName #'model.pt path(s)')
        else:
            # opt['weights'] = '/home/quan/server/yolo_detection/models/yolo_detect_card.pt' #'model.pt path(s)')
            raise Exception('modelName is Null!')
        opt['source'] = '/home/quan/server/yolo_detection/data/IMG_4989.JPG' #'source')  # file/folder, 0 for webcam
        opt['img-size'] = 640 #'inference size (pixels)')
        opt['conf-thres'] = 0.25 #'object confidence threshold')
        opt['iou-thres'] = 0.45 #'IOU threshold for NMS')
        opt['device'] = '' #'cuda device, i.e. 0 or 0,1,2,3 or cpu')
        opt['view-img'] = False #'store_true', help='display results')
        opt['save-txt'] =  False #'store_true', help='save results to *.txt')
        opt['save-conf'] = False #'store_true', help='save confidences in --save-txt labels')
        opt['classes'] = None #, nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        opt['agnostic-nms'] = False #action='store_true', help='class-agnostic NMS')
        opt['augment'] = False #  action='store_true', help='augmented inference')
        opt['update'] = False  #action='store_true', help='update all models')
        opt['project'] = 'runs/detect' #help='save results to project/name')
        opt['name'] = 'exp' #help='save results to project/name')
        opt['exist-ok'] = False # action='store_true', help='existing project/name ok, do not increment')
        return opt
        # opt = parser.parse_args()
        # print(opt)

        # with torch.no_grad():
        #     if opt.update:  # update all models (to fix SourceChangeWarning)
        #         for opt['weights'] in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #             detect()
        #             strip_optimizer(opt['weights'])
        #     else:
        #         detect()