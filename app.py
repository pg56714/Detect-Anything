import os
import warnings
import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from typing import List, Tuple
from scipy.spatial.distance import cdist

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from inference.models import YOLOWorld
from lama_inpaint import (
    build_lama_model,
    inpaint_img_with_builded_lama,
)
from stable_diffusion_inpaint import fill_img_with_sd, replace_img_with_sd
from utils import dilate_mask

warnings.filterwarnings("ignore")

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
sam_objShadow = EfficientViTSamPredictor(
    create_sam_model(name="xl1", weight_url="./weights/400_00948.pt").to(device).eval()
)

# build the lama model
lama_config = "./lama/configs/prediction/default.yaml"
lama_ckpt = "./weights/big-lama"
LAMA_MODEL = build_lama_model(lama_config, lama_ckpt, device=device)

# Load annotators
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# ================================================================

# Initialize global variables for filtering detections
available_classes = []


def filter_detections(conditions, detections):
    if not conditions or detections is None:
        return detections

    # print(f"Conditions: {conditions}")
    # print(f"Detections class_name: {detections.data.get('class_name', [])}")

    filtered_indices = []
    for i, (class_name, confidence) in enumerate(
        zip(detections.data.get("class_name", []), detections.confidence)
    ):
        for condition in conditions:
            if (
                class_name == condition["class"]
                and condition["score_min"] <= confidence <= condition["score_max"]
            ):
                filtered_indices.append(i)

    return detections[filtered_indices]


def add_condition(selected_class, score_min, score_max, existing_conditions):
    if not selected_class:
        return (
            existing_conditions,
            "Selected category is empty! Please select a category before adding conditions.",
        )

    for condition in existing_conditions:
        if condition["class"] == selected_class:
            return (
                existing_conditions,
                "Condition for this category already exists! Please delete it before adding a new one.",
            )

    existing_conditions.append(
        {"class": selected_class, "score_min": score_min, "score_max": score_max}
    )
    return existing_conditions, "Condition added successfully!"


def delete_condition(selected_class, existing_conditions):
    updated_conditions = [
        cond for cond in existing_conditions if cond["class"] != selected_class
    ]
    return updated_conditions, "Condition deleted successfully!"


def add_to_dropdown(input_text, available_classes):
    if available_classes is None:
        available_classes = []

    new_classes = [cls.strip() for cls in input_text.split(",") if cls.strip()]
    updated_classes = list(set(available_classes + new_classes))

    return (
        updated_classes,
        gr.update(choices=updated_classes, interactive=True),
        "Category added successfully!",
    )


# ================================================================


def image_upload(image, image_resolution):
    if image is None:
        return None
    else:
        np_image = np.array(image, dtype=np.uint8)
        if np_image.ndim == 2:
            np_image = np.stack((np_image,) * 3, axis=-1)
        elif np_image.shape[2] == 4:
            np_image = np_image[:, :, :3]
        np_image = resize_image(np_image, image_resolution)
        return np_image


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
    conditions: list,
    with_confidence: bool = True,
    with_segmentation: bool = True,
    reverse_filter: bool = False,
    with_dilate_kernel: bool = True,
    with_objShadow: bool = False,
    image_resolution: int = 512,
    dilate_kernel_size: int = 15,
) -> tuple:
    global exclude_positions

    # Resize image
    resized_image = resize_image(input_image, image_resolution)

    # Process categories
    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)

    # YOLO-World Detection
    results = YOLO_WORLD_MODEL.infer(resized_image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )

    # Apply filter condition (score ranges)
    if conditions:
        detections = filter_detections(conditions, detections)

    # EfficientViT-SAM Segmentation
    if with_segmentation:
        sam.set_image(resized_image, image_format="RGB")
        masks = []
        for xyxy in detections.xyxy:
            mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
            masks.append(mask.squeeze())
        detections.mask = np.array(masks)

    # Reverse filter with dynamic tol
    if reverse_filter and exclude_positions:
        # 自動推導 tol
        current_positions = [tuple(box) for box in detections.xyxy]
        tol = auto_calculate_tol_by_center_distance(
            exclude_positions, current_positions
        )
        # print(f"tol: {tol}")
        detections = filter_detections_by_position(detections, exclude_positions, tol)

    # 儲存目前位置供下次比較
    exclude_positions = [tuple(box) for box in detections.xyxy]

    # 合併 Mask
    if with_segmentation and detections.mask.size > 0:
        combined_mask = np.logical_or.reduce(detections.mask, axis=0)
        mask_image = (combined_mask * 255).astype(np.uint8)

        rgba_image = np.zeros((*resized_image.shape[:2], 4), dtype=np.uint8)
        rgba_image[..., :3] = resized_image
        rgba_image[..., 3] = mask_image

        bg_image = np.zeros_like(rgba_image)
        bg_image[..., :3] = resized_image
        bg_image[..., 3] = np.where(combined_mask, 0, 255)

        if with_dilate_kernel:
            dilated_mask = dilate_mask(mask_image, dilate_kernel_size)
            rgba_image[..., 3] = dilated_mask
            bg_image[..., 3] = np.where(dilated_mask, 0, 255)
    else:
        mask_image = np.zeros(resized_image.shape[:2], dtype=np.uint8)
        rgba_image = np.zeros((*resized_image.shape[:2], 4), dtype=np.uint8)
        bg_image = resized_image.copy()

    # ObjShadow mask
    if with_objShadow:
        sam_objShadow.set_image(resized_image, image_format="RGB")
        masks = []
        for xyxy in detections.xyxy:
            mask, _, _ = sam_objShadow.predict(box=xyxy, multimask_output=False)
            masks.append(mask.squeeze())
        obj_mask = (np.any(masks, axis=0) * 255).astype(np.uint8)
        final_mask = cv2.bitwise_or(mask_image, obj_mask)
        final_mask = (final_mask > 0).astype(np.uint8) * 255
    else:
        obj_mask = None
        final_mask = None

    # 標註輸出
    output_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(output_image, detections, categories, with_confidence)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return (
        output_image,
        mask_image,
        rgba_image,
        bg_image,
        obj_mask,
        final_mask,
    )


def auto_calculate_tol_by_center_distance(
    positions_1: List[Tuple[float, float, float, float]],
    positions_2: List[Tuple[float, float, float, float]],
) -> float:
    if not positions_1 or not positions_2:
        return 5.0

    def get_centers(positions):
        centers = np.array(
            [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in positions]
        )
        # print(f"👉 中心點計算: {centers.tolist()}")
        return centers

    centers_1 = get_centers(positions_1)
    centers_2 = get_centers(positions_2)
    dist_matrix = cdist(centers_1, centers_2)
    # print(f"\n📏 歐幾里得距離矩陣:\n{np.round(dist_matrix, 5)}\n")

    all_distances = dist_matrix.flatten()
    distance_threshold = np.median(all_distances) * 2

    matched_indices_2 = set()
    diffs = []

    for i, row in enumerate(dist_matrix):
        j = np.argmin(row)
        if row[j] > distance_threshold:
            continue
        if j in matched_indices_2:
            continue
        matched_indices_2.add(j)

        box1 = np.array(positions_1[i])
        box2 = np.array(positions_2[j])
        diff = np.abs(box1 - box2)
        max_diff = diff.max()

        # print(f"🔗 比對: positions_1[{i}] 和 positions_2[{j}]")
        # print(f"距離: {row[j]:.5f}")
        # print(f"box1: {np.round(box1, 5).tolist()}")
        # print(f"box2: {np.round(box2, 5).tolist()}")
        # print(f"座標差異: {np.round(diff, 5).tolist()}")
        # print(f"最大座標差異 (作為 tol 候選): {max_diff:.5f}\n")

        diffs.append(max_diff)

    if not diffs:
        # print("⚠️ 無有效配對，回傳預設容忍值 5.0\n")
        return 5.0

    tol = np.ceil(np.max(diffs) * 100) / 100
    # print(f"✅ 計算完成，回傳最大容忍差異 tol = {tol}\n")
    return tol


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


def get_fill_img_with_sd(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    np_image = resize_image(image, image_resolution)
    mask = cv2.resize(
        mask, (np_image.shape[1], np_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    img_fill = fill_img_with_sd(np_image, mask, text_prompt, device=device)
    img_fill = img_fill.astype(np.uint8)
    return img_fill


def get_replace_img_with_sd(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    np_image = resize_image(image, image_resolution)
    mask = cv2.resize(
        mask, (np_image.shape[1], np_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    img_replaced = replace_img_with_sd(np_image, mask, text_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced


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

with_objShadow = gr.Checkbox(
    value=False,
    label="With Object Shadow",
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
                with_objShadow.render()

        with gr.Column(scale=4, variant="panel"):
            with gr.Row():
                gr.Markdown("## Control Panel")
            image_categories_text_component = gr.Textbox(
                label="Categories",
                placeholder="you can input multiple words with comma (,)",
            )
            submit_button_component = gr.Button(value="Submit", variant="primary")
            lama = gr.Button("Inpaint Image", variant="primary")

            text_prompt = gr.Textbox(label="Text Prompt with SD")
            fill_sd = gr.Button("Fill Anything with SD", variant="primary")
            replace_sd = gr.Button("Replace Anything with SD", variant="primary")
            clear_button_image = gr.Button(value="Reset", variant="secondary")

    with gr.Row(variant="panel"):
        with gr.Column(scale=4):
            # Select category and confidence score range
            class_dropdown = gr.Dropdown(
                choices=[], label="Select Category", interactive=False
            )
            with gr.Row():
                score_min_slider = gr.Slider(
                    0, 1, step=0.01, label="Minimum Confidence Score"
                )
                score_max_slider = gr.Slider(
                    0, 1, step=0.01, label="Maximum Confidence Score"
                )

        with gr.Column(scale=2):
            with gr.Row():
                add_class_button = gr.Button("Add Category")
                add_button = gr.Button("Add Filter Condition")
                delete_button = gr.Button("Delete Filter Condition")

        with gr.Column(scale=4):
            # Status display
            conditions = gr.State([])  # Stores filter conditions
            condition_display = gr.Textbox(
                label="Current Filter Conditions", interactive=False
            )
            status_display = gr.Textbox(label="Status", interactive=False)

    # Event: Add Category
    add_class_button.click(
        add_to_dropdown,
        inputs=[image_categories_text_component, gr.State(available_classes)],
        outputs=[gr.State(available_classes), class_dropdown, status_display],
    )

    # Event: Add Condition
    add_button.click(
        add_condition,
        inputs=[class_dropdown, score_min_slider, score_max_slider, conditions],
        outputs=[conditions, status_display],
    )

    # Event: Delete Condition
    delete_button.click(
        delete_condition,
        inputs=[class_dropdown, conditions],
        outputs=[conditions, status_display],
    )

    # Update Condition Display
    conditions.change(
        lambda conds: "\n".join(
            [
                f"{c['class']} (Score Range: {c['score_min']}-{c['score_max']})"
                for c in conds
            ]
        ),
        inputs=[conditions],
        outputs=[condition_display],
    )

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

        # with gr.Column():
        #     with gr.Row():
        #         gr.Markdown("## Image Removed with Mask")
        #     with gr.Row():
        #         img_rm_with_mask = gr.Image(
        #             type="numpy",
        #             label="Image Removed with Mask",
        #             interactive=False,
        #             show_label=False,
        #         )

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## ObjShadow Mask")
            with gr.Row():
                objShadow_mask_image = gr.Image(
                    type="numpy",
                    label="ObjShadow Mask",
                    interactive=False,
                    show_label=False,
                )

        with gr.Column():
            with gr.Row():
                gr.Markdown("## ObjShadow Mask Filtered Image")
            with gr.Row():
                objShadow_mask_image_filtered = gr.Image(
                    type="numpy",
                    label="ObjShadow Mask Filtered Image",
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

                # img_rm_with_mask_test = gr.Image(
                #     type="numpy",
                #     label="Image Removed with Mask",
                #     interactive=False,
                #     show_label=False,
                # )

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Fill Anything with Mask")
            with gr.Row():
                img_fill_with_mask = gr.Image(
                    type="numpy",
                    label="Image Fill Anything with Mask",
                    interactive=False,
                )

        with gr.Column():
            with gr.Row():
                gr.Markdown("## Replace Anything with Mask")
            with gr.Row():
                img_replace_with_mask = gr.Image(
                    type="numpy",
                    label="Image Replace Anything with Mask",
                    interactive=False,
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
            conditions,
            with_confidence_component,
            with_segmentation_component,
            reverse_filter_component,
            with_dilate_kernel,
            with_objShadow,
            image_resolution,
            dilate_kernel_size,
        ],
        outputs=[
            yolo_world_output_image_component,
            mask_image,
            image_bg_removal,
            image_ob_removal,
            objShadow_mask_image,
            objShadow_mask_image_filtered,
        ],
    )

    # lama.click(
    #     get_inpainted_img,
    #     inputs=[input_image_component, mask_image],
    #     outputs=[img_rm_with_mask],
    # )

    # lama.click(
    #     get_inpainted_img,
    #     inputs=[input_image_component, objShadow_mask_image_filtered],
    #     outputs=[img_rm_with_mask_test],
    # )

    lama.click(
        lambda img, mask, obj_shadow_mask, obj_shadow_flag: (
            get_inpainted_img(img, obj_shadow_mask)
            if obj_shadow_flag
            else get_inpainted_img(img, mask)
        ),
        inputs=[
            input_image_component,  # Input image
            mask_image,  # General mask
            objShadow_mask_image_filtered,  # ObjShadow mask
            with_objShadow,  # Object Shadow flag
        ],
        outputs=[img_rm_with_mask],  # Conditional output
    )

    fill_sd.click(
        get_fill_img_with_sd,
        inputs=[input_image_component, mask_image, image_resolution, text_prompt],
        outputs=[img_fill_with_mask],
    )

    replace_sd.click(
        get_replace_img_with_sd,
        inputs=[input_image_component, mask_image, image_resolution, text_prompt],
        outputs=[img_replace_with_mask],
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
            img_fill_with_mask,
            img_replace_with_mask,
            objShadow_mask_image,
            objShadow_mask_image_filtered,
        ],
        outputs=[
            input_image_component,
            yolo_world_output_image_component,
            image_bg_removal,
            image_ob_removal,
            mask_image,
            img_rm_with_mask,
            img_fill_with_mask,
            img_replace_with_mask,
            objShadow_mask_image,
            objShadow_mask_image_filtered,
        ],
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

demo.launch(debug=False, show_error=True)
