import argparse

from seg_utils import *
from dec_utils import *
from seg_utils import seg_transforms, seg_dataset_kaggle
import time
from models import dec_net_seg, seg_net
import cv2


parser = argparse.ArgumentParser(description='Detection Training (MultiGPU)')
parser.add_argument('--testDir', default="DataSets/kaggle/11test", type=str, help='test image directory')
parser.add_argument('--annoDir', default="DataSets/kaggle/mask", type=str, help='annotation image directory')
parser.add_argument('--imgSuffix', default='.png', type=str, help='suffix of the input images')
parser.add_argument('--annoSuffix', default='.png', type=str, help='suffix of the annotation images')
parser.add_argument('--img_height', default=512, type=int, help='img height')
parser.add_argument('--img_width', default=512, type=int, help='img width')
parser.add_argument('--num_classes', default=2, type=int, help='dataset classes')
parser.add_argument('--top_k', default=500, type=int, help='the number of detections to keep')
parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.3, type=float, help='nms threshold')
parser.add_argument('--seg_thresh', default=0.5, type=float, help='segmentation threshold')
parser.add_argument('--dec_weights', default="dec_weights/end_model.pth", type=str, help='detection weights')
parser.add_argument('--seg_weights', default="seg_weights/end_model.pth", type=str, help='segmentation weights')


def test(args):
    #-----------------load detection model -------------------------
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dec_model = dec_net_seg.resnetssd50(pretrained=False, num_classes=args.num_classes)
    resume_dict = torch.load(args.dec_weights, map_location='cpu')
    resume_dict = {k: v for k, v in resume_dict.items()}
    dec_model.load_state_dict(resume_dict, strict=False)
    dec_model = dec_model.to(device)
    dec_model.eval()
    #-----------------load segmentation model -------------------------
    seg_model = seg_net.SEG_NET(num_classes=args.num_classes)
    seg_model.load_state_dict(torch.load(args.seg_weights, map_location='cpu'),strict=False)

    seg_model = seg_model.to(device)
    seg_model.eval()
    ##--------------------------------------------------------------
    data_transforms = seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                       seg_transforms.Resize(args.img_height, args.img_width),
                                       seg_transforms.ToTensor()])

    dsets = seg_dataset_kaggle.NucleiCell(args.testDir, args.annoDir, data_transforms,
                                          imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix)

    # for validation data -----------------------------------
    detector = Detect(num_classes=args.num_classes,
                      top_k=args.top_k,
                      conf_thresh=args.conf_thresh,
                      nms_thresh=args.nms_thresh,
                      variance=[0.1, 0.2])
    anchorGen = Anchors(args.img_height, args.img_width)
    anchors = anchorGen.forward()
    all_time = []
    for img_idx in range(len(dsets)):
        time_begin = time.time()
        print('loading {}/{} image'.format(img_idx, len(dsets)))
        ori_img = dsets.load_img(img_idx)
        black = cv2.cvtColor(np.zeros((ori_img.shape[0], ori_img.shape[1]),dtype=np.uint8),cv2.COLOR_GRAY2BGR)
        img = ori_img.astype(np.float32)
        img = cv2.resize(img, dsize=(512, 512))
        img = torch.from_numpy(img.copy().transpose((2, 0, 1)))
        inputs = img / 255
        h,w,c = ori_img.shape
        x = inputs.unsqueeze(0)

        x = x.to(device)
        locs, conf, feat_seg = dec_model(x)
        detections = detector(locs, conf, anchors)
        outputs = seg_model(detections, feat_seg)
        mask_patches, mask_dets = outputs
        all_time.append(time.time() - time_begin)
        # For batches
        for b_mask_patches, b_mask_dets in zip(mask_patches, mask_dets):

            nd = len(b_mask_dets)
            # Step1: rearrange mask_patches and mask_dets
            for d in range(nd):
                d_mask = np.zeros((args.img_height, args.img_width), dtype=np.float32)
                d_mask_det = b_mask_dets[d].data.cpu().numpy()
                d_mask_patch = b_mask_patches[d].data.cpu().numpy()
                d_bbox = d_mask_det[0:4]
                d_conf = d_mask_det[4]
                if d_conf < args.conf_thresh:
                    continue
                [y1, x1, y2, x2] = d_bbox
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), args.img_height - 1)
                x2 = np.minimum(np.int32(np.round(x2)), args.img_width - 1)
                d_mask_patch = cv2.resize(d_mask_patch, (x2 - x1 + 1, y2 - y1 + 1))
                d_mask_patch = np.where(d_mask_patch >= args.seg_thresh, 1., 0.)
                d_mask[y1:y2 + 1, x1:x2 + 1] = d_mask_patch
                d_mask = cv2.resize(d_mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

                #ori_img = map_mask_to_image(d_mask, ori_img, color=np.random.rand(3))
                black = map_mask_to_image(d_mask, black, color=np.random.rand(3))
        cv2.imwrite('TCGA_imgs/{}_gt.png'.format(img_idx), black)
    all_time = all_time[1:]
    print('avg time is {}'.format(np.mean(all_time)))
    print('FPS is {}'.format(1. / np.mean(all_time)))
    print('Finish')


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
