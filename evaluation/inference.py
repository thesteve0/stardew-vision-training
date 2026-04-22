#!/usr/bin/env python3
"""Model loading and single-image inference for Qwen2.5-VL."""

import logging
import os

import torch
from PIL import Image

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluation.prompt import build_messages
from evaluation.tool_parser import ParsedPrediction, parse_model_output

EVAL_IMAGE_SIZE = (1600, 1200)

logger = logging.getLogger(__name__)


def load_model(
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2.5-VL model and processor.

    Uses FP16 (required for ROCm gfx1151, no BF16 support).
    """
    logger.info(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
    )
    model.eval()
    model = torch.compile(model)
    logger.info("Model loaded and compiled")
    return model, processor


def run_inference(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    max_new_tokens: int = 512,
) -> ParsedPrediction:
    """Run inference on a single screenshot and parse the tool call."""
    img = Image.open(image_path).convert("RGB").resize(EVAL_IMAGE_SIZE)
    messages = build_messages(img)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    raw_output = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    return parse_model_output(raw_output)


def run_inference_traced(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    max_new_tokens: int = 512,
) -> tuple[ParsedPrediction, str]:
    """Like run_inference but also returns the full prompt text for debugging."""
    img = Image.open(image_path).convert("RGB").resize(EVAL_IMAGE_SIZE)
    messages = build_messages(img)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    raw_output = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    return parse_model_output(raw_output), text
