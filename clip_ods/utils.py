import numpy as np


IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def get_anchor_coords(image_w, image_h, count_w, count_h):
    step_w = image_w // count_w + 1
    step_h = image_h // count_h + 1

    w_coords = np.arange(0, image_w + step_w, step_w)
    h_coords = np.arange(0, image_h + step_h, step_h)

    coords = []
    for x1, x2 in zip(w_coords[:-1], w_coords[1:]):
        x1, x2 = min(x1, image_w), min(x2, image_w)
        for y1, y2 in zip(h_coords[:-1], h_coords[1:]):
            y1, y2 = min(y1, image_h), min(y2, image_h)
            coords.append((x1, y1, x2, y2))

            anchor_w = (x2 - x1) / 2
            anchor_h = (y2 - y1) / 2
            anchor_xc = (x2 + x1) // 2
            anchor_yc = (y2 + y1) // 2

            for coef_x, coef_y in [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (1, 2), (2, 1),
                (2, 3), (3, 2),
                (2, 4), (4, 2),
                (3, 1), (1, 3),
                (5, 4), (4, 5),
                (4, 1), (1, 4),
                (5, 1), (1, 5),
                (5, 3), (3, 5),
                (6, 4), (4, 6),
                (5, 8), (8, 5),
                (10, 2), (2, 10),
                (10, 4), (4, 10),
                (10, 6), (6, 10),
                (10, 8), (8, 10),
            ]:
                anc_x1 = max(anchor_xc - (anchor_w * coef_x), 0)
                anc_x2 = min(anchor_xc + (anchor_w * coef_x), image_w)
                anc_y1 = max(anchor_yc - (anchor_h * coef_y), 0)
                anc_y2 = min(anchor_yc + (anchor_h * coef_y), image_h)
                coords.append((anc_x1, anc_y1, anc_x2, anc_y2))

    return coords
