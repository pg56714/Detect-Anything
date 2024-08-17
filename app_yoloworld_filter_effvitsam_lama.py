import os
import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from typing import List, Tuple
from inference.models import YOLOWorld

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model

import warnings

warnings.filterwarnings("ignore")

from lama_inpaint import (
    inpaint_img_with_lama,
    build_lama_model,
    inpaint_img_with_builded_lama,
)

from utils import (
    load_img_to_array,
    save_array_to_img,
    dilate_mask,
    show_mask,
    show_points,
)

MARKDOWN = """
# YOLO-World + EfficientViT-SAM
Powered by Roboflow [Inference](https://github.com/roboflow/inference) and [Supervision](https://github.com/roboflow/supervision) and [YOLO-World](https://github.com/AILab-CVC/YOLO-World) and [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)
"""
IMAGE_EXAMPLES = [
    [
        os.path.join(os.path.dirname(__file__), "images/1224276_original.jpg"),
        "person wearing green clothes",
        0.2,
        0.5,
        True,
        # True,
        True,
        False,
    ],
    [
        os.path.join(os.path.dirname(__file__), "images/livingroom.jpg"),
        "table, lamp, dog, sofa, plant, clock, carpet, frame on the wall",
        0.05,
        0.5,
        True,
        # True,
        True,
        False,
    ],
    [
        os.path.join(os.path.dirname(__file__), "images/cat_and_dogs.jpg"),
        "cat, dog",
        0.2,
        0.5,
        True,
        # True,
        True,
        False,
    ],
]


# Load models
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/s")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/m")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/x")

# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-s")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-m")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-l")
# YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/v2-x")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = EfficientViTSamPredictor(
    create_sam_model(name="xl1", weight_url="./weights/xl1.pt").to(device).eval()
)

# build the lama model
lama_config = "./lama/configs/prediction/default.yaml"
lama_ckpt = "./weights/big-lama"
LAMA_MODEL = build_lama_model(lama_config, lama_ckpt, device=device)

# Load annotators
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def image_upload(image, image_resolution):
    if image is None:
        return None
    else:
        np_image = np.array(image, dtype=np.uint8)
        np_image = resize_image(np_image, image_resolution)
        return image


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    k = float(resolution) / min(H, W)
    H = int(np.round(H * k / 64.0)) * 64
    W = int(np.round(W * k / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(",")]


def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = True,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


def process_image(
    input_image: np.ndarray,
    categories: str,
    confidence_threshold: float,
    nms_threshold: float,
    with_confidence: bool = True,
    with_segmentation: bool = True,
    reverse_filter: bool = False,
    with_dilate_kernel: bool = True,
    image_resolution: int = 512,
    dilate_kernel_size: int = 15,
) -> tuple:
    global exclude_positions

    # Resize the input image to the desired resolution
    resized_image = resize_image(input_image, image_resolution)

    # Preparation.
    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)

    # Object detection
    results = YOLO_WORLD_MODEL.infer(resized_image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )

    # Segmentation
    if with_segmentation:
        sam.set_image(resized_image, image_format="RGB")
        masks = []
        for xyxy in detections.xyxy:
            mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
            mask = mask.squeeze()  # Reduce dimensions
            masks.append(mask)

        detections.mask = np.array(masks)
        # print(f"detections.mask: {detections.mask}")

    if reverse_filter and exclude_positions:
        detections = filter_detections_by_position(detections, exclude_positions)

    exclude_positions = [(box[0], box[1], box[2], box[3]) for box in detections.xyxy]

    # Combine masks into one mask image
    if with_segmentation and detections.mask.size > 0:
        combined_mask = np.logical_or.reduce(detections.mask, axis=0)
        mask_image = combined_mask.astype(np.uint8) * 255

        # Create RGBA images
        rgba_image = np.zeros((*resized_image.shape[:2], 4), dtype=np.uint8)
        rgba_image[..., :3] = resized_image  # Set RGB channels

        rgba_image[..., 3] = mask_image  # Set alpha channel from the mask
        image_bg_removal = rgba_image

        bg_image = np.zeros((*resized_image.shape[:2], 4), dtype=np.uint8)
        bg_image[..., :3] = resized_image
        bg_image[..., 3] = np.where(combined_mask, 0, 255)

        if with_dilate_kernel:
            mask_image = dilate_mask(mask_image, dilate_kernel_size)
            image_bg_removal[..., 3] = mask_image  # Dilate alpha channel only

            binary_mask = combined_mask.astype(np.uint8)
            dilated_mask = dilate_mask(binary_mask, dilate_kernel_size)
            bg_image[..., 3] = np.where(dilated_mask, 0, 255)

    else:
        mask_image = np.zeros(
            resized_image.shape[:2], dtype=np.uint8
        )  # Return an empty mask if no segmentation
        image_bg_removal = np.zeros(
            (*resized_image.shape[:2], 4), dtype=np.uint8
        )  # Empty RGBA image
        bg_image = resized_image.copy()

    # Annotation
    output_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence,
    )

    return (
        cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB),
        mask_image,
        image_bg_removal,
        bg_image,
    )


# atol 誤差值 先設定 5
def filter_detections_by_position(
    detections: sv.Detections,
    positions: List[Tuple[float, float, float, float]],
    tol: float = 5,
) -> sv.Detections:
    filtered_indices = [
        i
        for i, box in enumerate(detections.xyxy)
        if not any(np.allclose(box, pos, atol=tol) for pos in positions)
    ]

    # debug
    # filtered_indices = []
    # for i, box in enumerate(detections.xyxy):
    #     print(f"i: {i}")
    #     print(f"box: {box}")
    #     keep = True
    #     for pos in positions:
    #         print(f"pos: {pos}")
    #         if np.allclose(box, pos, atol=tol):
    #             keep = False
    #             break
    #     if keep:
    #         filtered_indices.append(i)

    return detections[filtered_indices]


def get_inpainted_img(image, mask):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = cv2.resize(
        mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    img_inpainted = inpaint_img_with_builded_lama(
        LAMA_MODEL, image, mask, lama_config, device=device
    )
    return img_inpainted


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.005,
    step=0.01,
    label="Confidence Threshold",
    #     info=(
    #         "The confidence threshold for the YOLO-World model. Lower the threshold to "
    #         "reduce false negatives, enhancing the model's sensitivity to detect "
    #         "sought-after objects. Conversely, increase the threshold to minimize false "
    #         "positives, preventing the model from identifying objects it shouldn't."
    #     ),
)

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.5,
    step=0.01,
    label="IoU Threshold",
    # info=(
    #     "The Intersection over Union (IoU) threshold for non-maximum suppression. "
    #     "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
    #     "making the detection process stricter. On the other hand, increase the value "
    #     "to allow more overlapping bounding boxes, accommodating a broader range of "
    #     "detections."
    # ),
)

with_confidence_component = gr.Checkbox(
    value=True,
    label="Display Confidence",
    # info=("Whether to display the confidence of the detected objects."),
)

with_segmentation_component = gr.Checkbox(
    value=True,
    label="With Segmentation",
    # info=("Whether to run EfficientViT-SAM for instance segmentation."),
)


reverse_filter_component = gr.Checkbox(
    value=False,
    label="Toggle Reverse Filter",
    #     info=(
    #         "Toggle the reverse filter. When on, it excludes detections overlapping stored target positions."
    #     ),
)

with_dilate_kernel = gr.Checkbox(
    value=True,
    label="With Dilate Kernel",
)

image_resolution = gr.Slider(
    label="Image Resolution for Stable Diffusion",
    minimum=256,
    maximum=1024,
    value=512,
    step=64,
)

dilate_kernel_size = gr.Slider(
    label="Dilate Kernel Size",
    minimum=0,
    maximum=30,
    value=15,
    step=1,
)

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column(scale=6, variant="panel"):
            with gr.Row():
                gr.Markdown("## Upload an image.")
                gr.Markdown("## Output image.")
            with gr.Row():
                input_image_component = gr.Image(
                    type="numpy",
                    label="Input Image",
                    show_label=False,
                )
                yolo_world_output_image_component = gr.Image(
                    type="numpy",
                    label="Output image",
                    interactive=False,
                    show_label=False,
                )
            with gr.Row():
                confidence_threshold_component.render()
                iou_threshold_component.render()
                image_resolution.render()
                dilate_kernel_size.render()

            with gr.Row():
                with_confidence_component.render()
                with_segmentation_component.render()
                reverse_filter_component.render()
                with_dilate_kernel.render()

        with gr.Column(scale=4, variant="panel"):
            with gr.Row():
                gr.Markdown("## Control Panel")
            image_categories_text_component = gr.Textbox(
                label="Categories",
                placeholder="you can input multiple words with comma (,)",
            )
            submit_button_component = gr.Button(value="Submit", variant="primary")
            lama = gr.Button("Inpaint Image", variant="primary")
            clear_button_image = gr.Button(value="Reset", variant="secondary")

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Mask")
            with gr.Row():
                mask_image = gr.Image(
                    type="numpy",
                    label="Mask Image",
                    interactive=False,
                    show_label=False,
                )

        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Background Removal")
            with gr.Row():
                image_bg_removal = gr.Image(
                    type="numpy",
                    label="Image Background Removal",
                    show_label=False,
                    interactive=False,
                )

        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Object Removal")
            with gr.Row():
                image_ob_removal = gr.Image(
                    type="numpy",
                    label="Image Object Removal",
                    show_label=False,
                    interactive=False,
                )

        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Removed with Mask")
            with gr.Row():
                img_rm_with_mask = gr.Image(
                    type="numpy",
                    label="Image Removed with Mask",
                    interactive=False,
                    show_label=False,
                )

    gr.Examples(
        # fn=process_image,
        examples=IMAGE_EXAMPLES,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_confidence_component,
            with_segmentation_component,
            reverse_filter_component,
            with_dilate_kernel,
            image_resolution,
            dilate_kernel_size,
        ],
        outputs=[yolo_world_output_image_component, mask_image],
    )

    input_image_component.upload(
        image_upload,
        inputs=[input_image_component, image_resolution],
        outputs=[input_image_component],
    )

    submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_confidence_component,
            with_segmentation_component,
            reverse_filter_component,
            with_dilate_kernel,
            image_resolution,
            dilate_kernel_size,
        ],
        outputs=[
            yolo_world_output_image_component,
            mask_image,
            image_bg_removal,
            image_ob_removal,
        ],
    )

    lama.click(
        get_inpainted_img,
        inputs=[input_image_component, mask_image],
        outputs=[img_rm_with_mask],
    )

    def reset(*args):
        return [None for _ in args]

    clear_button_image.click(
        reset,
        inputs=[
            input_image_component,
            yolo_world_output_image_component,
            image_bg_removal,
            image_ob_removal,
            mask_image,
            img_rm_with_mask,
        ],
        outputs=[
            input_image_component,
            yolo_world_output_image_component,
            image_bg_removal,
            image_ob_removal,
            mask_image,
            img_rm_with_mask,
        ],
    )

demo.launch(debug=False, show_error=True)
