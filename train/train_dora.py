#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import jittor as jt
import jittor.nn as nn
import argparse
import copy
import logging
import math
import torch
import os
import warnings
from pathlib import Path

import numpy as np
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from jittor import transform
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from JDiffusion import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers import DDPMScheduler,DDIMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_process",
        type=int,
        default=1
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--attribute_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--without_painting",
        default=False,
        required=False,
        action="store_true"
        # help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=5, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # logger is not available yet
    if args.class_data_dir is not None:
        warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
    if args.class_prompt is not None:
        warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        attribute_prompt,
        tokenizer,
        without_painting,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.without_painting = without_painting
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.attribute_prompt = attribute_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        #图像增强
        self.image_transforms = transform.Compose(
            [
                transform.Resize(size),
                transform.CenterCrop(size) if center_crop else transform.RandomCrop(size),
                # transform.RandomAffine(degrees=10),
                transform.ToTensor(),
                transform.ImageNormalize([0.5], [0.5]),
            ]
        )
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        #GUO
        file_name=str(self.instance_images_path[index % self.num_instance_images]).split("/")[-1]
        file_name = file_name.rsplit('.', 1)[0]
        #file_name = file_name.replace('_', ' ')
        name = str(file_name)
        if self.without_painting:
            new_prompt=self.attribute_prompt+' of '+str(file_name)
        else:
            new_prompt=self.attribute_prompt+' painting of '+str(file_name)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, new_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask


        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                new_class_prompt=self.class_prompt+'a photo of '+str(file_name)
                # print(new_class_prompt)
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, new_class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask
 
        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = jt.stack(pixel_values)
    pixel_values = pixel_values.float()

    input_ids = jt.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    A = text_encoder.device
    text_input_ids = input_ids.to(text_encoder.device)
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, lora_alpha,original_weight):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_A = jt.randn((input_dim, rank), requires_grad=True) * 0.01
        self.lora_B = jt.randn((rank, output_dim), requires_grad=True) * 0.01
        self.original_weight = original_weight
        self.m = jt.Var(jt.ones((1, output_dim)))  # 可训练的长度 m
        self.m = original_weight["weight"].norm(p=2,dim=0,keepdim=True)
        self.linear1 = nn.Linear(input_dim, output_dim)
    def execute(self, x):
        # 从 original_weight 计算范数
        
        # 计算 lora 输出
        # with jt.no_grad():
        original_weight = self.original_weight["weight"]
        
        
        norm = jt.norm(original_weight, p=2, dim=0, keepdim=True)  # 计算范数
        directional_component_ori = original_weight/norm
        new_ori_weight = self.m * directional_component_ori # 使用可训练的 m
        self.linear1.weight = new_ori_weight
        # 处理偏置
        if self.original_weight["bias"] is not None:
            self.linear1.bias = self.original_weight["bias"]
        a = self.linear1(x)
        return a+jt.matmul(jt.matmul(self.m*x/norm, self.lora_A), self.lora_B)#jt.matmul(x, new_weight.transpose())+self.original_weight["bias"] if self.original_weight["bias"] is not None else jt.matmul(x, new_weight.transpose())# 返回最终输出
class CustomUNet(UNet2DConditionModel):
    def __init__(self, base_unet, rank, target_modules):

        self.base_unet = base_unet
        self.lora_layers = {}
        # 复制 base_unet 的属性到当前 UNet
        self.__dict__.update(base_unet.__dict__)

        for name, layer in base_unet.named_modules():
            flag = 0
            real_name = name.split(".")[-1]
            if real_name.endswith('0'):
                if name.endswith("to_out.0"):
                    real_name = '.'.join(name.split(".")[-2:])
                    flag = 1
            if real_name in target_modules:
                print(name)
                input_dim = layer.in_features   # 假设层有 in_features 属性
                output_dim = layer.out_features   # 假设层有 out_features 属性
                original_weight={}
                original_weight["weight"] = layer.weight
                original_weight["bias"] = layer.bias
                lora_layer = LoRALayer(input_dim, output_dim, rank, rank, original_weight)
                self.lora_layers[name] = lora_layer
                # 创建一个新的层，包含原层和 LoRA 层的逻辑
                new_layer = nn.Module()
                new_layer.add_module('mylora_layer', lora_layer)
                new_layer.execute = self.create_execute_with_lora(new_layer)
                
                # 替换原有层
                parent_module = self.get_parent_module(name)
                if flag ==1:
                    parent_module.layers['0']= new_layer
                else:
                    setattr(parent_module, real_name, new_layer)
        self.set_grad()
    def set_grad(self):
        """激活所有 new_layer 中 mylora_layer 的梯度"""
        for name, module in self.named_modules():
            if hasattr(module, 'mylora_layer'):
                module.mylora_layer.lora_A.requires_grad = True
                module.mylora_layer.lora_B.requires_grad = True
                module.mylora_layer.m.requires_grad = True
    def create_execute_with_lora(self, new_layer):
        def execute(x,scale = None):
            lora_output = new_layer.mylora_layer(x)
            return lora_output
        return execute

    def get_parent_module(self, name):
        parent_name = '.'.join(name.split('.')[:-1])
        if parent_name:
            return dict(self.named_modules())[parent_name]
        return None
        # for name, layer in base_unet.named_modules():
        #     real_name = name.split(".")[-1]
        #     if real_name in target_modules:
        #         input_dim = layer.in_features   # 假设层有 input_dim 属性
        #         output_dim = layer.out_features   # 假设层有 output_dim 属性
        #         self.lora_layers[name] = LoRALayer(input_dim, output_dim, rank, rank)

def main(args):
    jt.flags.use_cuda = 1
    jt.misc.set_global_seed(args.seed)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
  
    # We only train the additional adapter LoRA layers
   #if vae is not None:
   #    vae.requires_grad_(False)
   #text_encoder.requires_grad_(False)
   #unet.requires_grad_(False) 
    # trigger_word = "style_00"
    
    # tokenizer.add_tokens(trigger_word)
    # example_ =  text_encoder.get_input_embeddings().weight.data[0]
    # embedding_vector = jt.zeros_like(example_)  
    # text_encoder.resize_token_embeddings(len(tokenizer))

    # # 将嵌入向量赋值到 text_encoder 的嵌入层
    # token_id = tokenizer.convert_tokens_to_ids(trigger_word)
    # tmp = text_encoder.get_input_embeddings().weight.data
    # tmp[token_id] = embedding_vector
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = jt.float32
    print(unet)
    for name, param in unet.named_parameters():
        assert param.requires_grad == False, name
    # now we will add new LoRA weights to the attention layers
    target_modules=["to_k", "to_q", "to_v","to_out.0","proj_out","proj_in"]
    # Optimizer creation
    unet = CustomUNet(unet,args.rank,target_modules)
    for name,param in unet.named_parameters():
        print(name, param.requires_grad)
    optimizer = AdamW(
        list(unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None
    args.with_prior_preservation=False
    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        attribute_prompt=args.attribute_prompt,
        without_painting = args.without_painting if args.without_painting else False,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt ,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.num_process,
        num_training_steps=args.max_train_steps * args.num_process,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    tracker_config = vars(copy.deepcopy(args))
    tracker_config.pop("validation_images")

    # Train!
    total_batch_size = args.train_batch_size * args.num_process * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            this = getattr(unet, 'down_blocks')[2].attentions[0].proj_in
            print(this.mylora_layer.lora_A)
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = jt.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = jt.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), 
            ).to(device=model_input.device)
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            

            # Get the text embedding for conditioning
    
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )





            # encoder_hidden_states[0][-1]=jt.ones_like(encoder_hidden_states[0][-1])







            if unet.config.in_channels == channels * 2:
                noisy_model_input = jt.cat([noisy_model_input, noisy_model_input], dim=1)

            if args.class_labels_conditioning == "timesteps":
                class_labels = timesteps
            else:
                class_labels = None

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                class_labels=class_labels,
                return_dict=False,
            )[0]

            # if model predicts variance, throw away the prediction. we will only train on the
            # simplified training objective. This means that all schedulers using the fine tuned
            # model must be configured to use one of the fixed variance variance types.
            if model_pred.shape[1] == 6:
                model_pred, _ = jt.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            # if args.with_prior_preservation:
            #     # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            #     model_pred, model_pred_prior = jt.chunk(model_pred, 2, dim=0)
            #     target, target_prior = jt.chunk(target, 2, dim=0)

            #     # Compute instance loss
            #     loss = nn.smooth_l1_loss(target,model_pred)

            #     # Compute prior loss
            #     prior_loss = nn.smooth_l1_loss(target_prior,model_pred_prior,)

            #     # Add the prior loss to the instance loss.
            #     loss = loss + args.prior_loss_weight * prior_loss
            # else:
            #     loss = nn.smooth_l1_loss(target,model_pred)


            #loss=nn.l1_loss(model_pred,target)
            #loss = nn.mse_loss(model_pred, target)
            loss = nn.smooth_l1_loss(target,model_pred)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            
            logs = {"loss": loss.detach().item()}
            #logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    def save_all_lora_weights(model, file_path):
        state_dict = {}
        for name, lora_layer in model.lora_layers.items():
            state_dict[name] = {
                'lora_A': lora_layer.lora_A,
                'lora_B': lora_layer.lora_B,
                'm':lora_layer.m,
                'lora_alpha': lora_layer.lora_alpha,
                'rank': lora_layer.rank,
            }
        jt.save(state_dict, file_path)
    save_all_lora_weights(unet,args.output_dir+"/lora.pth")
    unet = unet.to(jt.float32)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
