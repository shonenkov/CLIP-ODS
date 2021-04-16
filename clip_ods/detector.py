from itertools import chain, combinations

import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from ensemble_boxes import weighted_boxes_fusion
from PIL import ImageDraw

from .utils import IMAGENET_TEMPLATES
from . import clip


class AnchorImageDataset(Dataset):

    def __init__(self, image, coords, transforms):
        self.image = image.copy()
        self.coords = coords
        self.transforms = transforms

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return self.transforms(self.image.crop(coord))


class CLIPDetectorV0:

    def __init__(self, model, transforms, device):
        """
        First version, required improving of quality and speed :)
        """
        self.device = device
        self.model = model
        self.transforms = transforms
        self.model.to(device)
        self.model.eval()
        self.zero_text_embeddings = self.model.encode_text(
            clip.tokenize([template.format('') for template in IMAGENET_TEMPLATES]).to(self.device)
        )

    def get_anchor_features(self, img, coords, bs=32, quite=False):
        anchor_dataset = AnchorImageDataset(img, coords, self.transforms)
        anchor_loader = DataLoader(
            anchor_dataset,
            batch_size=bs,
            sampler=SequentialSampler(anchor_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=2,
        )
        if not quite:
            anchor_loader = tqdm(anchor_loader)
        anchor_features = []
        for anchor_batch in anchor_loader:
            with torch.no_grad():
                anchor_features_ = self.model.encode_image(anchor_batch.to(self.device))
                anchor_features_ /= anchor_features_.norm(dim=-1, keepdim=True)
                anchor_features.append(anchor_features_)
        return torch.vstack(anchor_features)

    def detect_by_text(
            self, texts, img, coords, anchor_features, *, tp_thr=0.0, fp_thr=-2.0, iou_thr=0.01, skip_box_thr=0.8,
    ):
        """
        :param texts: list of text query
        :param img: PIL of raw image
        :param coords: list of anchor coords Pascal/VOC format [[x1,y1,x2,y2], ...]
        :param anchor_features: pt tensor with anchor features of image
        :param tp_thr: threshold of true positive query, uses for full size image
        :param fp_thr: threshold of false positive query, uses for full size image
        :param iou_thr: parameter for ensemble boxes using WBF, see https://github.com/ZFTurbo/Weighted-Boxes-Fusion
        :param skip_box_thr: parameter for ensemble boxes using WBF, see https://github.com/ZFTurbo/Weighted-Boxes-Fusion
        :return: (img, result, thr)
        """
        zeroshot_weights = []
        with torch.no_grad():
            text_embeddings = []
            for text in texts:
                tokens = clip.tokenize([template.format(text) for template in IMAGENET_TEMPLATES]).to(self.device)
                text_embeddings.append(self.model.encode_text(tokens))

            text_embeddings = torch.stack(text_embeddings).mean(0)
            text_embeddings -= self.zero_text_embeddings
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.mean(dim=0)
            text_embeddings /= text_embeddings.norm()

            zeroshot_weights.append(text_embeddings)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
            logits = (anchor_features @ zeroshot_weights).reshape(-1)
            probas, indexes = torch.sort(logits, descending=True)

            img_features = self.model.encode_image(self.transforms(img).unsqueeze(0).to(self.device)).squeeze(0)
            thr = (img_features @ zeroshot_weights)[0].item()

        w, h = img.size
        boxes_list = []
        scores_list = []
        labels_list = []
        probas = probas.cpu().numpy()
        probas = probas - np.min(probas)
        probas = probas / max(0.2, np.max(probas))
        thr_indexes, = np.where(probas > skip_box_thr)
        if thr > fp_thr:
            if thr_indexes.shape[0] != 0:
                for best_index, proba in zip(indexes[thr_indexes], probas[thr_indexes]):
                    x1, y1, x2, y2 = list(coords[best_index])
                    x1, y1, x2, y2 = max(x1 / w, 0.0), max(y1 / h, 0.0), min(x2 / w, 1.0), min(y2 / h, 1.0)
                    boxes_list.append([x1, y1, x2, y2])
                    scores_list.append(proba)
                    labels_list.append(1)
            else:
                if thr > tp_thr:
                    best_index, proba = indexes[0], probas[0]
                    x1, y1, x2, y2 = list(coords[best_index])
                    x1, y1, x2, y2 = max(x1 / w, 0.0), max(y1 / h, 0.0), min(x2 / w, 1.0), min(y2 / h, 1.0)
                    boxes_list.append([x1, y1, x2, y2])
                    scores_list.append(proba)
                    labels_list.append(1)

        boxes, scores, labels = weighted_boxes_fusion(
            [boxes_list], [scores_list], [labels_list],
            weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        result = {'boxes': [], 'scores': [], 'labels': []}
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            result['boxes'].append([x1, y1, x2, y2])
            result['scores'].append(float(score))
            result['labels'].append(int(label))
            draw = ImageDraw.Draw(img)
            draw.rectangle((x1, y1, x2, y2), width=2, outline=(0, 0, 255))
        return img, result, thr


class CLIPDetectorV1:

    def __init__(self, model, transforms, device):
        """
        Detection and Segmentation using classic CV methods
        """
        self.device = device
        self.model = model
        self.transforms = transforms
        self.model.to(device)
        self.model.eval()
        self.zero_text_embeddings = self.model.encode_text(
            clip.tokenize([template.format('') for template in IMAGENET_TEMPLATES]).to(self.device)
        )

    def draw(self, img, results, label='', colour=(0, 0, 255), width=1, font_colour=(0, 0, 0), font_scale=1,
             font_thickness=1,
             T=0.6):
        """
        :param img:
        :param results:
        :param label:
        :param colour:
        :param width:
        :param font_scale:
        :param font_thickness:
        :param T: transparency
        :return:
        """
        img = np.array(img)
        R, G, B = colour
        for mask, (x1, y1, x2, y2) in zip(results['masks'], results['boxes']):
            img[y1:y2, x1:x2, 0][mask] = ((1 - T) * img[y1:y2, x1:x2, 0][mask]).astype(np.uint8) + np.uint8(R * T)
            img[y1:y2, x1:x2, 1][mask] = ((1 - T) * img[y1:y2, x1:x2, 1][mask]).astype(np.uint8) + np.uint8(G * T)
            img[y1:y2, x1:x2, 2][mask] = ((1 - T) * img[y1:y2, x1:x2, 2][mask]).astype(np.uint8) + np.uint8(B * T)

        for x1, y1, x2, y2 in results['boxes']:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colour, width)
            img = cv2.putText(img, label, (x1 + (x2 - x1) // 10, y1 + (y2 - y1) // 5), cv2.FONT_ITALIC,
                              font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
            img = cv2.putText(img, label, (x1 + (x2 - x1) // 10, y1 + (y2 - y1) // 5), cv2.FONT_ITALIC,
                              font_scale, font_colour, font_thickness, cv2.LINE_AA)
        return img

    def detect_by_text(
            self, texts, img, coords, anchor_features, masks=None, *,
            tp_thr=0.0, fp_thr=-2.0, iou_thr=0.01, skip_box_thr=0.8,
    ):
        """
        :param texts: list of text query
        :param img: PIL of raw image
        :param coords: list of anchor coords Pascal/VOC format [[x1,y1,x2,y2], ...]
        :param anchor_features: pt tensor with anchor features of image
        :param masks: list of bool masks for every anchor (with same order)
        :param tp_thr: threshold of true positive query, uses for full size image
        :param fp_thr: threshold of false positive query, uses for full size image
        :param iou_thr: threshold IoU for NMS
        :param skip_box_thr: threshold of score for skip box of anchor
        :return: (result, thr)
        """
        text_embeddings = self.get_text_embeddings(texts)
        with torch.no_grad():
            logits = (anchor_features @ text_embeddings).reshape(-1)
            probas, indexes = torch.sort(logits, descending=True)
            img_features = self.model.encode_image(self.transforms(img).unsqueeze(0).to(self.device)).squeeze(0)
            thr = (img_features @ text_embeddings).item()

        boxes = []
        scores = []
        if thr > fp_thr:
            probas = probas.cpu().numpy()
            probas = probas - np.min(probas)
            probas = probas / max(0.2, np.max(probas))
            thr_indexes, = np.where(probas > skip_box_thr)
            if thr_indexes.shape[0] == 0:
                if thr > tp_thr:
                    thr_indexes = thr_indexes[:1]
                else:
                    thr_indexes = []
            for best_index, proba in zip(indexes[thr_indexes], probas[thr_indexes]):
                x1, y1, x2, y2 = list(coords[best_index])
                boxes.append([x1, y1, x2, y2])
                scores.append(proba)

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32).to(self.device)
            labels = torch.ones(boxes.shape[0], dtype=torch.float32).to(self.device)
            scores = np.array(scores)
            indexes = nms(boxes, labels, iou_thr).cpu().numpy()
            boxes = boxes[indexes].cpu().numpy()
            scores = scores[indexes]

        result = {'boxes': [], 'scores': [], 'labels': [], 'masks': [], 'thr': thr}
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            if masks is not None:
                (x1, y1, x2, y2), mask = self._get_nearest_box_and_mask((x1, y1, x2, y2), coords, masks)
                result['masks'].append(mask)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            result['boxes'].append([x1, y1, x2, y2])
            result['scores'].append(float(score))
            result['labels'].append(1)

        return result

    def get_text_embeddings(self, texts):
        with torch.no_grad():
            text_embeddings = []
            for text in texts:
                tokens = clip.tokenize([template.format(text) for template in IMAGENET_TEMPLATES]).to(self.device)
                text_embeddings.append(self.model.encode_text(tokens))
            text_embeddings = torch.stack(text_embeddings).mean(0)
            text_embeddings -= self.zero_text_embeddings
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.mean(dim=0)
            text_embeddings /= text_embeddings.norm()
        return text_embeddings

    def get_coords_and_masks(self, pil_img, B=0.15, K_max_box_w=0.9, K_max_box_h=0.9,
                             K_min_box_w=0.03, K_min_box_h=0.03, iou_threshold=0.9):
        img = pil_img.copy()
        img = np.array(img)
        img = self._c_mean_shift(img)
        img = self._split_gray_img(img, n_labels=9)
        coords, masks = self._get_mixed_boxes_and_masks(
            img, B, K_max_box_w, K_max_box_h, K_min_box_w, K_min_box_h, iou_threshold
        )
        return coords, masks

    def get_anchor_features(self, img, coords, bs=32, quite=False):
        anchor_dataset = AnchorImageDataset(img, coords, self.transforms)
        anchor_loader = DataLoader(
            anchor_dataset,
            batch_size=bs,
            sampler=SequentialSampler(anchor_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=2,
        )
        if not quite:
            anchor_loader = tqdm(anchor_loader)
        anchor_features = []
        for anchor_batch in anchor_loader:
            with torch.no_grad():
                anchor_features_ = self.model.encode_image(anchor_batch.to(self.device))
                anchor_features_ /= anchor_features_.norm(dim=-1, keepdim=True)
                anchor_features.append(anchor_features_)
        return torch.vstack(anchor_features)

    def _c_mean_shift(self, image):
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.pyrMeanShiftFiltering(img, 16, 48)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    def _split_gray_img(self, image, n_labels=9):
        img = image.copy()
        step = 255 // n_labels
        t = list(np.arange(0, 255, step)) + [255]
        for i, (t1, t2) in enumerate(zip(t[:-1], t[1:])):
            img[(img >= t1) & (img < t2)] = t1
        return img

    def _get_mixed_boxes_and_masks(self, image, B, K_max_box_w, K_max_box_h, K_min_box_w, K_min_box_h, iou_threshold):
        img = image.copy()
        h, w = img.shape
        max_box_w, max_box_h, min_box_w, min_box_h = w * K_max_box_w, h * K_max_box_h, w * K_min_box_w, h * K_min_box_h

        out_boxes = []
        out_masks = []

        labels = np.unique(img)
        combs = self._get_combinations(labels)

        comb_indexes = []
        for i, comb in enumerate(combs):
            n_img = np.isin(img, np.array(comb)).astype(np.uint8) * 255
            n_img = self._clear_noise(n_img)
            m_boxes = self._get_boxes_from_mask(n_img, max_box_w, max_box_h, min_box_w, min_box_h)
            out_boxes.extend(m_boxes)
            comb_indexes.extend([i] * len(m_boxes))

        comb_indexes = np.array(comb_indexes)

        boxes = torch.tensor(out_boxes, dtype=torch.float32)
        labels = torch.ones(boxes.shape[0], dtype=torch.float32)

        indexes = nms(boxes, labels, iou_threshold)

        out_boxes = boxes[indexes].numpy().astype(np.int32)
        comb_indexes = comb_indexes[indexes.numpy()]

        for (x1, y1, x2, y2), comb_index in zip(out_boxes, comb_indexes):
            comb = combs[comb_index]
            n_img = np.isin(img, np.array(comb)).astype(np.uint8) * 255
            n_img = self._clear_noise(n_img)
            mask = n_img[y1:y2, x1:x2]

            h, w = mask.shape
            h_b = int(h * B)
            w_b = int(w * B)

            mask = mask.astype(np.bool)
            if mask[:h_b, :].sum() + mask[-h_b:, :].sum() + mask[:, :w_b].sum() + \
                    mask[:, -w_b:].sum() > 4 * h * w * B * (1 - B) * 0.5:
                mask = ~mask

            mask = mask.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask = mask.astype(np.bool)
            out_masks.append(mask)

        return out_boxes, out_masks

    @staticmethod
    def _get_combinations(array):
        combs = list(chain(*map(lambda x: combinations(array, x), range(0, len(array)+1))))
        return combs[1:]

    @staticmethod
    def _get_boxes_from_mask(mask, max_box_w, max_box_h, min_box_w, min_box_h):
        boxes = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours]
        for i, (x, y, w, h) in enumerate(bboxes):
            if (w < max_box_w and h < max_box_h) and (w > min_box_w and h > min_box_h):
                boxes.append([x, y, x + w, y + h])
        return boxes

    @staticmethod
    def _clear_noise(image):
        img = image.copy()
        e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erose = cv2.morphologyEx(img, cv2.MORPH_ERODE, e_kernel)
        d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.morphologyEx(erose, cv2.MORPH_DILATE, d_kernel)
        return dilate

    @staticmethod
    def _get_nearest_box_and_mask(box, gt_boxes, gt_masks):
        return sorted(zip(gt_boxes, gt_masks), key=lambda x: sum([abs(x[0][i] - box[i]) for i in range(4)]))[0]
