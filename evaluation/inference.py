#!/usr/bin/env python3
"""Model loading and single-image inference for Qwen2.5-VL."""

import logging
import os
from pathlib import Path

import torch
from PIL import Image

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from evaluation.prompt import build_messages
from evaluation.tool_parser import ParsedPrediction, parse_model_output

EVAL_IMAGE_SIZE = (1600, 1200)
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

logger = logging.getLogger(__name__)


def _is_lora_adapter(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def load_model(
    model_path: str = BASE_MODEL,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2.5-VL model and processor.

    If model_path is a LoRA adapter directory, loads the base model
    and applies the adapter. Uses FP16 (required for ROCm gfx1151).
    """
    if _is_lora_adapter(model_path):
        logger.info(f"Loading base model: {BASE_MODEL}")
        processor = AutoProcessor.from_pretrained(BASE_MODEL)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info(f"Applying LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        logger.info(f"Loading model: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
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
