from typing import List, Tuple
import cv2
import gradio as gr
import numpy as np
import supervision as sv
from inference.models import YOLOWorld

MARKDOWN = """
# YoloWorldTest

Powered by Roboflow [Inference](https://github.com/roboflow/inference) and 
[Supervision](https://github.com/roboflow/supervision).
"""

IMAGE_EXAMPLES = [
    [
        "https://media.roboflow.com/dog.jpeg",
        "dog, eye, nose, tongue, car",
        0.005,
        0.1,
        True,
        False,
    ],
    [
        "./images/1224276_original.jpg",
        "person wearing green clothes",
        0.2,
        0.5,
        True,
        False,
    ],
]

# Load models
YOLO_WORLD_MODEL = YOLOWorld(model_id="yolo_world/l")

# BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()  # Corrected based on your latest script
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# 儲存要排除的物件位置
exclude_positions = []


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
    iou_threshold: float,
    with_confidence: bool = True,
    reverse_filter: bool = True,
) -> np.ndarray:
    global exclude_positions

    categories = process_categories(categories)
    YOLO_WORLD_MODEL.set_classes(categories)
    results = YOLO_WORLD_MODEL.infer(
        input_image, confidence=confidence_threshold, nms=iou_threshold
    )
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(threshold=iou_threshold)

    if isinstance(input_image, str):
        input_image = cv2.imread(input_image)

    results = YOLO_WORLD_MODEL.infer(
        input_image, confidence=confidence_threshold, nms=iou_threshold
    )
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(threshold=iou_threshold)

    if reverse_filter and exclude_positions:
        detections = filter_detections_by_position(detections, exclude_positions)

    exclude_positions = [(box[0], box[1], box[2], box[3]) for box in detections.xyxy]

    # print(f"detections1: {detections}")
    # print(f"exclude_positions: {exclude_positions}")

    output_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_image = annotate_image(
        input_image=output_image,
        detections=detections,
        categories=categories,
        with_confidence=with_confidence,
    )

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


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


confidence_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.005,
    step=0.01,
    label="Confidence Threshold",
    info=(
        "The confidence threshold for the YOLO-World model. Lower the threshold to "
        "reduce false negatives, enhancing the model's sensitivity to detect "
        "sought-after objects. Conversely, increase the threshold to minimize false "
        "positives, preventing the model from identifying objects it shouldn't."
    ),
)

iou_threshold_component = gr.Slider(
    minimum=0,
    maximum=1.0,
    value=0.5,
    step=0.01,
    label="IoU Threshold",
    info=(
        "The Intersection over Union (IoU) threshold for non-maximum suppression. "
        "Decrease the value to lessen the occurrence of overlapping bounding boxes, "
        "making the detection process stricter. On the other hand, increase the value "
        "to allow more overlapping bounding boxes, accommodating a broader range of "
        "detections."
    ),
)

with_confidence_component = gr.Checkbox(
    value=True,
    label="Display Confidence",
    info=("Whether to display the confidence of the detected objects."),
)

reverse_filter_component = gr.Checkbox(
    value=False,
    label="Toggle Reverse Filter",
    info=(
        "Toggle the reverse filter. When on, it excludes detections overlapping stored target positions."
    ),
)

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="numpy", label="Input Image")
            with gr.Accordion("Configuration", open=False):
                confidence_threshold_component.render()
                iou_threshold_component.render()
                with gr.Row():
                    with_confidence_component.render()
                    reverse_filter_component.render()

            image_categories_text_component = gr.Textbox(
                label="Categories",
                placeholder="you can input multiple words with comma (,)",
            )
        yolo_world_output_image_component = gr.Image(
            type="numpy", label="YOLO-WORLD Output"
        )
    submit_button_component = gr.Button(value="Submit", scale=1, variant="primary")
    gr.Examples(
        fn=process_image,
        examples=IMAGE_EXAMPLES,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_confidence_component,
        ],
        outputs=[
            yolo_world_output_image_component,
        ],
    )

    submit_button_component.click(
        fn=process_image,
        inputs=[
            input_image_component,
            image_categories_text_component,
            confidence_threshold_component,
            iou_threshold_component,
            with_confidence_component,
            reverse_filter_component,
        ],
        outputs=[
            yolo_world_output_image_component,
        ],
    )

demo.launch(debug=False, show_error=True)
