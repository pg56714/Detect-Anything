import gradio as gr

# 初始化模型檢測的類別選項和信賴分數範圍
available_classes = []  # 假設的類別
detection_results = [...]  # 模型的檢測結果列表


def filter_detections(conditions):
    # 根據條件過濾檢測結果
    filtered_results = []
    for condition in conditions:
        filtered_results += [
            det
            for det in detection_results
            if det["class"] == condition["class"]
            and condition["score_min"] <= det["score"] <= condition["score_max"]
        ]
    return filtered_results  # 返回篩選後的檢測結果或圖像


def add_condition(selected_class, score_min, score_max, existing_conditions):
    # 檢查是否有選擇類別
    if not selected_class:
        return existing_conditions, "選擇類別為空！請先選擇一個類別後再新增條件。"

    # 檢查該類別是否已經有篩選條件
    for condition in existing_conditions:
        if condition["class"] == selected_class:
            return existing_conditions, "該類別的條件已存在！請刪除後重新新增。"

    # 新增條件
    existing_conditions.append(
        {"class": selected_class, "score_min": score_min, "score_max": score_max}
    )
    return existing_conditions, "成功新增條件！"


def delete_condition(selected_class, existing_conditions):
    # 刪除指定類別的條件
    updated_conditions = [
        cond for cond in existing_conditions if cond["class"] != selected_class
    ]
    return updated_conditions, "成功刪除條件！"


def add_to_dropdown(input_text, available_classes):
    if available_classes is None:
        available_classes = []

    # 解析輸入內容，去掉多餘的空白並去重
    new_classes = [cls.strip() for cls in input_text.split(",") if cls.strip()]
    updated_classes = list(set(available_classes + new_classes))

    # 回傳更新後的類別列表並啟用下拉選單
    return (
        updated_classes,
        gr.update(choices=updated_classes, interactive=True),
        "成功新增類別！",
    )


# 設置 Gradio 介面
with gr.Blocks() as demo:
    gr.Markdown("### 動態條件篩選器")

    # 新增類別輸入框
    new_class_input = gr.Textbox(label="新增類別（用逗號分隔）")

    # 選擇類別與信賴分數範圍
    class_dropdown = gr.Dropdown(choices=[], label="選擇類別", interactive=False)
    score_min_slider = gr.Slider(0, 1, step=0.01, label="信賴分數最小值")
    score_max_slider = gr.Slider(0, 1, step=0.01, label="信賴分數最大值")

    # 按鈕
    add_class_button = gr.Button("新增類別")
    add_button = gr.Button("新增篩選條件")
    delete_button = gr.Button("刪除篩選條件")

    # 狀態顯示
    conditions = gr.State([])  # 儲存篩選條件
    condition_display = gr.Textbox(label="當前篩選條件", interactive=False)
    status_display = gr.Textbox(label="狀態", interactive=False)

    # 事件：新增類別
    add_class_button.click(
        add_to_dropdown,
        inputs=[new_class_input, gr.State(available_classes)],
        outputs=[gr.State(available_classes), class_dropdown, status_display],
    )

    # 事件：新增條件
    add_button.click(
        add_condition,
        inputs=[class_dropdown, score_min_slider, score_max_slider, conditions],
        outputs=[conditions, status_display],
    )

    # 事件：刪除條件
    delete_button.click(
        delete_condition,
        inputs=[class_dropdown, conditions],
        outputs=[conditions, status_display],
    )

    # 更新條件顯示
    conditions.change(
        lambda conds: "\n".join(
            [
                f"{c['class']} (分數範圍: {c['score_min']}-{c['score_max']})"
                for c in conds
            ]
        ),
        inputs=[conditions],
        outputs=[condition_display],
    )

demo.launch()
