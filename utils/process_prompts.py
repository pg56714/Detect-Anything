import numpy as np

from utils.utils import (
    MUTLIMASK,
    draw_box_masks,
    draw_point_and_box_masks,
    draw_point_masks,
    get_box_inputs,
    get_point_inputs,
)


def segment_using_points(prompt_dict, model):
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    points = get_point_inputs(prompts)

    if len(points) == 0:
        return raw_image

    point_coords = points[..., :2]
    point_labels = points[..., 2]

    model.set_image(raw_image)
    masks, _, _ = model.predict(
        point_coords=np.array(point_coords),
        point_labels=np.array(point_labels),
        multimask_output=MUTLIMASK,
    )

    return draw_point_masks(raw_image, masks, points)


def segment_using_boxes(prompt_dict, model):
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    boxes = np.array(get_box_inputs(prompts))

    if len(boxes) == 0:
        return raw_image

    model.set_image(raw_image)
    masks, _, _ = model.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=MUTLIMASK,
    )

    return draw_box_masks(raw_image, masks, boxes)


def segment_using_points_and_boxes(prompt_dict, model):
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]

    boxes = get_box_inputs(prompts)
    points = get_point_inputs(prompts)

    if len(boxes) == 0 and len(points) == 0:
        return raw_image
    elif len(boxes) == 0:
        return segment_using_points(prompt_dict, model)
    elif len(points) == 0:
        return segment_using_boxes(prompt_dict, model)

    point_coords = points[..., :2]
    point_labels = points[..., 2]
    boxes = np.array(boxes)

    model.set_image(raw_image)
    masks, _, _ = model.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=boxes,
        multimask_output=MUTLIMASK,
    )

    return draw_point_and_box_masks(raw_image, masks, points, boxes)
