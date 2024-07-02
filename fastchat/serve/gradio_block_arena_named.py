"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time
import requests

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 64)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown(choices=models, value=model_left, visible=True),
        gr.Dropdown(choices=models, value=model_right, visible=True),
    )

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def leftvote_last_response(
        state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def rightvote_last_response(
        state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def tievote_last_response(
        state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def bothbad_vote_last_response(
        state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
                states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {get_ip(request)}")
    return (
            [None] * num_sides
            + [None] * num_sides
            + [""]
            + [invisible_btn] * 4
            + [disable_btn] * 2
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (named). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(
        state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    m = {"CPU": "vicuna-7b-v1.5", "GPU": "vicuna-7b-v1.5-GTX4090"}
    model_selector0 = m[model_selector0]
    model_selector1 = m[model_selector1]
    ip = get_ip(request)
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + ["", None]
                + [
                    no_change_btn,
                ]
                * 6
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[1].conv.get_prompt()
    all_conv_text = (
            all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list)
    if flagged:
        logger.info(f"violate moderation (named). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [CONVERSATION_LIMIT_MSG]
                + [
                    no_change_btn,
                ]
                * 6
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                disable_btn,
            ]
            * 6
    )


def bot_response_multi(
        state0,
        state1,
        temperature,
        top_p,
        max_new_tokens,
        request: gr.Request,
        progress=gr.Progress(),
):
    logger.info(f"bot_response_multi (named). ip: {get_ip(request)}")

    progress(0, desc="Starting")
    progress(0.05)

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    model_tpy = []
    for i in range(num_sides):
        token_per_yield = 1
        if states[i].model_name in [
            "gemini-pro",
            "gemma-1.1-2b-it",
            "gemma-1.1-7b-it",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "snowflake-arctic-instruct",
        ]:
            token_per_yield = 30
        elif states[i].model_name in [
            "qwen-max-0428",
            "qwen1.5-110b-chat",
        ]:
            token_per_yield = 7
        elif states[i].model_name in [
            "qwen2-72b-instruct",
        ]:
            token_per_yield = 4
        model_tpy.append(token_per_yield)

    chatbots = [None] * num_sides
    scores = [None] * num_sides
    points = [None] * num_sides
    score_data = [None] * num_sides
    point_data = [None] * num_sides
    gpt = None
    iters = 0
    while True:
        if iters <= 10:
            progress(0.1 * iters, desc="Generating")
        stop = True
        iters += 1
        for i in range(num_sides):
            if "GTX" not in states[i].model_name and iters % 3 != 1:
                stop = False
                continue
            try:
                # yield fewer times if chunk size is larger
                if model_tpy[i] == 1 or (iters % model_tpy[i] == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                score_data[i] = chatbots[i][0]
                point_data[i] = chatbots[i][0][1]
        yield states + chatbots + scores + points + [gpt] + [disable_btn] * 6
        if stop:
            r = requests.post("http://45.32.109.227:5000/score", json=score_data)
            j = r.json()
            for i in range(num_sides):
                scores[i] = j[i]['score']
            r = requests.post("http://45.32.109.227:5000/similarity", json=point_data)
            j = r.json()
            points = [x[0] for x in j]
            r = requests.post("http://45.32.109.227:5000/gpt4_score", json=score_data)
            j = r.json()
            gpt = j["scores"]
            print(gpt)
            yield states + chatbots + scores + points + [gpt] + [disable_btn] * 6
            break


def flash_buttons():
    btn_updates = [
        [disable_btn] * 4 + [enable_btn] * 2,
        [enable_btn] * 6,
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.3)


def build_side_by_side_ui_named(models):
    notice_markdown = """
# ⚔️  AOS-Playground: A front-end AI project framework built on gardio

- AOS-Playground is convenient for AI developers to test the comparison of node inference performance of different 
frameworks. 
- Support worker (GPU & CPU) node registration, run gguf format models, respond to inference requests from 
front-end users.

## 🤖 Choose models to compare
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides
    scores = [None] * num_sides
    points = [None] * num_sides
    gpt = None

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    llms = []
    for m in models:
        if m.split("-")[0] not in llms:
            llms.append(m.split("-")[0])

    with gr.Column():
        gr.Dropdown(
            label="Models",
            choices=llms,
        )

    mm = ["CPU", "GPU"]

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Radio(
                        choices=mm,
                        value=models[i],
                        interactive=True,
                        show_label=False,
                        container=False,
                    )

        with gr.Row():
            for i in range(num_sides):
                # label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        # label=label,
                        elem_id=f"chatbot",
                        height=650,
                        show_copy_button=True,
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    scores[i] = gr.Textbox(label="Inference Quality Score", info="Based on sbert")

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    points[i] = gr.Textbox(label="Inference result similarity points", visible=(i == 1),
                                           info="A Sbert-based "
                                                "similarity analysis is provided (-1, 1). The closer the"
                                                "value is to 1, the more relevant it is.")

        with gr.Column():
            gpt = gr.Textbox(label="GPT4-Score", info="Based on GPT")

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners
    btn_list = [
        regenerate_btn,
        clear_btn,
    ]
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + scores + points + [gpt] + btn_list,
    ).then(
        flash_buttons, [], btn_list
    ).then(

    )
    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + scores + points + [gpt] + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
