import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image, ImageDraw

from .utils import IMAGENET_TEMPLATES
from . import clip


class AnchorImageDataset(Dataset):

    def __init__(self, image, coords, transforms):
        self.image = image
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

    def get_anchor_clip_features(self, img, coords, bs=32, quite=False):
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
            self, search_texts, coords, anchor_features, image_path,
            *, iou_thr=0.01, skip_box_thr=0.1, proba_thr=0.65,
    ):
        zeroshot_weights = []
        with torch.no_grad():
            texts = []
            for search_text in search_texts:
                texts.extend([template.format(search_text) for template in IMAGENET_TEMPLATES])

            texts = clip.tokenize(texts).to(self.device)
            text_embeddings = self.model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()

            zeroshot_weights.append(text_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
            logits = (anchor_features @ zeroshot_weights).reshape(-1)
            probas, indexes = torch.sort(logits, descending=True)

        probas = probas.cpu().numpy()
        probas = probas - np.min(probas)
        probas = probas / max(0.2, np.max(probas))

        img = Image.open(image_path)

        w, h = img.size
        boxes_list = []
        scores_list = []
        labels_list = []
        thr_indexes, = np.where(probas > proba_thr)
        if thr_indexes.shape[0] != 0:
            for best_index, proba in zip(indexes[thr_indexes], probas[thr_indexes]):
                x1, y1, x2, y2 = list(coords[best_index])
                x1 /= w
                x2 /= w
                y1 /= h
                y2 /= h
                boxes_list.append([x1, y1, x2, y2])
                scores_list.append(proba)
                labels_list.append(1)

        boxes, scores, labels = weighted_boxes_fusion(
            [boxes_list], [scores_list], [labels_list],
            weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        result = {'boxes': [], 'scores': scores, 'labels': labels}
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            result['boxes'].append([x1, y1, x2, y2])
            draw = ImageDraw.Draw(img)
            draw.rectangle((x1, y1, x2, y2), width=4, outline=(0, 0, 255))
        return img, result

    def detect_coco(self):
        pass
