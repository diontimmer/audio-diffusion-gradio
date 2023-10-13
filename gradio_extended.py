import gc
import glob
import json
import os

import gradio as gr
import numpy as np
import pytorch_lightning as pl
import torch
from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from prefigure.prefigure import push_wandb_config
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import importlib
import random

from stable_audio_tools.data.dataset import create_dataloader_from_configs_and_args
from stable_audio_tools.inference.generation import (
    generate_diffusion_cond,
    generate_diffusion_uncond,
)
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.training import (
    create_demo_callback_from_config,
    create_training_wrapper_from_config,
)
from stable_audio_tools.training.utils import copy_state_dict
from types import SimpleNamespace
import traceback
from tqdm import tqdm
import timeit
from tyler import tyler_patch, TYLER
import subprocess
import sys

import wandb

from gradio.components.base import FormComponent


class ToolButton(FormComponent, gr.Button):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, *args, **kwargs):
        classes = kwargs.pop("elem_classes", [])
        super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

    def get_block_name(self):
        return "button"


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(
            command, shell=True, env=os.environ if custom_env is None else custom_env
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}"""
            )

        return ""

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=os.environ if custom_env is None else custom_env,
    )

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(args, desc=None):
    index_url = os.environ.get("INDEX_URL", "")
    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    return run(
        f'"{sys.executable}" -m pip {args} --prefer-binary{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
    )


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package.split("==")[0])
    except ModuleNotFoundError:
        return False

    return spec is not None


class stable_audio_interface:
    def __init__(
        self,
        init_model_dirs=[],
        hidden=[],
        init_model_ckpt=None,
        init_model_config=None,
        init_pretransform_ckpt=None,
        init_device=None,
        extensions_folder="./extensions",
    ):
        self.current_set_model_path = None
        self.current_set_model_config = None
        self.current_set_pretransform_path = None
        self.current_loaded_model = None
        self.current_loaded_model_path = None
        self.current_loaded_pretransform_path = None
        self.current_loaded_model_config = None
        self.current_wrapped = True
        self.embedded_config = False
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_device = init_device if init_device else self.current_device
        self.fp16 = True if "cuda" in self.current_device else False
        self.init_model_dirs = init_model_dirs
        self.aec_ui = None
        self.txt2audio_ui = None
        self.blank_ui = None
        self.tab_control = None
        self.info_box = None
        self.interface = None
        self.device_selector = None
        self.fp16_checkbox = None
        self.config_selector = None
        self.pretransform_selector = None
        self.model_selector = None
        self.timing_acc = None
        self.sec_start_slider = None
        self.sec_total_slider = None
        self.cfg_scale_slider = None
        self.cfg_rescale_slider = None
        self.conditioning_section = None
        self.sigma_min = None
        self.prompt_box = None
        self.init_audio_input = None
        self.audio_outputs = None
        self.send_to_init_n = None
        self.sample_size_slider = None
        self.max_outputs = 36
        self.hidden = hidden  # ["train", "settings", "extensions"]
        self.extensions_folder = extensions_folder
        self.loaded_send_tos = []

        if init_model_ckpt is not None:
            self.prepare_initial_model(
                init_model_ckpt, init_model_config, init_pretransform_ckpt
            )

    def load_examples(self):
        examples = None
        examples_path = os.path.join(os.path.dirname(__file__), "examples.txt")
        if os.path.exists(examples_path):
            with open(examples_path) as f:
                examples = f.read().splitlines()
        return examples

    def load_tags(self):
        tags = None
        tags_path = os.path.join(os.path.dirname(__file__), "tags.txt")
        if os.path.exists(tags_path):
            with open(tags_path) as f:
                tags = f.read().splitlines()
        return tags

    def make_loadable_vars(self):
        return [
            self.info_box,
            self.cfg_scale_slider,
            self.cfg_rescale_slider,
            self.conditioning_section,
            self.sigma_min,
            self.aec_ui,
            self.txt2audio_ui,
            self.blank_ui,
            self.tab_control,
            self.timing_acc,
            self.sec_start_slider,
            self.sec_total_slider,
            self.prompt_box,
            self.sample_size_slider,
            self.device_selector,
            self.fp16_checkbox,
            self.model_selector,
            self.config_selector,
            self.pretransform_selector,
        ]

    def prepare_initial_model(self, ckpt, config, pretransform):
        load_success = self.load_model(config, ckpt, pretransform)
        if load_success:
            self.current_set_model_path = ckpt
            self.current_set_pretransform_path = pretransform
            self.current_set_model_config = (
                config if config else self.current_set_model_config
            )  # check if embedded
            print(f"Initial load complete.")

    def refresh_model(self):
        if self.current_loaded_model_path:
            self.load_model(
                self.current_loaded_model_config,
                self.current_loaded_model_path,
                self.current_loaded_pretransform_path,
            )
            gr.Info("Reloaded model!")
        else:
            gr.Warning("No model loaded!")

        return self.on_load()

    def create_audio_output(self, *args, **kwargs):
        kwargs["interactive"] = False
        # kwargs["show_download_button"] = False
        return gr.Audio(*args, **kwargs)

    def on_unload(self):
        self.unload_model()
        return self.on_load() + [None, None, None]

    def create_send_to_ui(self, output_component):
        with gr.Accordion("⬇️ Import from output", open=False):
            with gr.Column():
                send_n = gr.Number(
                    label="Import #",
                    value=1,
                    precision=1,
                    minimum=1,
                )
                send_to_button = gr.Button(f"⬇️ Import")
                send_to_button.click(
                    self.send_to_fn,
                    inputs=self.audio_outputs + [send_n],
                    outputs=output_component,
                    api_name=False,
                    queue=False,
                )

    def create_audio_input(self):
        with gr.Group():
            input_audio = gr.Audio(label="Input Audio", elem_id="small_audio")
            self.create_send_to_ui(input_audio)
            clear_btn = gr.ClearButton(components=[input_audio], variant="secondary")

        return input_audio

    def create_send_to_input_list(self, input_components):
        with gr.Row():
            with gr.Group():
                send_n = gr.Number(label="Send #", value=1, precision=1, minimum=1)
                send_to_button = gr.Button("Send to init audio", scale=1)
                send_to_button.click(
                    self.send_to_fn,
                    inputs=input_components + [send_n],
                    outputs=[self.init_audio_input],
                    api_name=False,
                    queue=False,
                )

    def add_tag_to_promptbox(self, prompt, tag):
        return tag[0] if prompt is None else f"{prompt}, {tag[0]}"

    def unload_model(self):
        if self.current_loaded_model:
            print("Unloading model..")
            self.current_loaded_model.to("cpu")
            del self.current_loaded_model
            torch.cuda.empty_cache()
            gc.collect()
            self.current_loaded_model = None
            # self.current_loaded_model_config = None
            self.current_loaded_model_path = None
            self.current_loaded_pretransform_path = None
            # self.embedded_config = False
            # self.current_set_model_config = None
            # self.current_set_model_path = None
            return True
        else:
            return False

    def dynamic_find_modules(self, prefix="EXT_"):
        modules = []
        if os.path.exists(self.extensions_folder):
            for folder in os.listdir(self.extensions_folder):
                folder_path = os.path.join(self.extensions_folder, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.lower().startswith(prefix.lower()) and file.endswith(
                            ".py"
                        ):
                            module_path = os.path.join(folder_path, file)
                            spec = importlib.util.spec_from_file_location(
                                module_path, module_path
                            )
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            modules.append(module)
                            if hasattr(module, "interface"):
                                module.interface = self
                            if hasattr(module, "make_ui"):
                                module.make_ui()
        else:
            print("No extensions folder found.")
        return modules

    def load_extensions(self):
        extensions = self.dynamic_find_modules()
        for extension in extensions:
            self.find_installs(extension)
        if extensions:
            print(f"Loaded {len(extensions)} extensions.")
        return extensions

    def send_to_fn(self, *inputs):
        slider = inputs[-1]
        audios = inputs[:-1]
        index = int(slider - 1)
        # check if audio[index] exists
        if index < len(audios):
            return audios[index]
        else:
            return None

    def find_installs(self, checked_module):
        if hasattr(checked_module, "install_modules"):
            for module in checked_module.install_modules:
                checked = module
                if hasattr(checked_module, "install_aliases") and module in list(
                    checked_module.install_aliases.keys()
                ):
                    checked = checked_module.aliases[module]
                installed = is_installed(checked)
                if not installed:
                    install_command = f"install {module}"
                    run_pip(install_command, desc=f"{module}")
                else:
                    print(f"{module} already installed.")

    def reset_sample_size(self):
        return self.current_loaded_model_config["sample_size"]

    def calculate_sample_size(self, sample_size):
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        sample_rate = self.current_loaded_model_config["sample_rate"]
        total_seconds = sample_size / sample_rate
        formatted_time = format_time(total_seconds)
        print(f"{sample_size} samples = {formatted_time}")
        gr.Info(f"{sample_size} samples = {formatted_time}")

    def load_model(
        self,
        model_config,
        model_ckpt_path,
        pretrans_select_value="",
        load_pretransform_only=False,
    ):
        try:
            if not load_pretransform_only and self.current_loaded_model:
                if self.embedded_config:
                    emb = True
                self.unload_model()
                if emb:
                    self.embedded_config = True
            if (
                not self.current_loaded_model_config
                and not self.embedded_config
                and not model_config
            ):
                print("No config file provided")
                gr.Error("No config file provided")
                return False

            if not self.embedded_config and self.current_set_model_config != "internal":
                with open(self.current_set_model_config) as f:
                    model_config = json.load(f)
            else:
                model_config = self.current_loaded_model_config

            if not load_pretransform_only:
                print(f"Creating model from config")
                model = create_model_from_config(model_config)

                model.to(self.current_device).eval().requires_grad_(False)

                print(f"Loading model checkpoint from {model_ckpt_path}")
                # Load checkpoint
                ckpt = torch.load(model_ckpt_path, map_location=self.current_device)
                model.load_state_dict(
                    ckpt["state_dict"],
                    strict=False,
                )
                self.current_wrapped = (
                    True if "optimizer_states" in list(ckpt.keys()) else False
                )

            else:
                model = self.current_loaded_model

            if pretrans_select_value:
                print(f"Loading pretransform from {pretrans_select_value}")
                aec_weights = torch.load(pretrans_select_value, map_location="cpu")[
                    "state_dict"
                ]
                model.pretransform.load_state_dict(aec_weights, strict=False)

            model = model.half() if self.fp16 else model

            print(f"Done loading model")
            self.current_loaded_model = model
            self.current_loaded_model_config = model_config
            self.current_loaded_model_path = model_ckpt_path
            self.current_loaded_pretransform_path = (
                pretrans_select_value if pretrans_select_value else None
            )

            return True
        except Exception as e:
            gr.Error(f"Error loading model: {e}")
            print(e)
            traceback.print_exc()
            return False

    def generate(
        self,
        prompt,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        tame=False,
        steps=250,
        seed=-1,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.03,
        sigma_max=50,
        sample_size=65536,
        cfg_rescale=0.7,
        init_audio=None,
        init_noise_level=1.0,
        batch_size=1,
        split_decode=True,
        tiled_encode=False,
        tiled_encode_window_length=65536,
        tiled_decode=False,
        tiled_decode_window_length=65536,
        preview_every=None,
        progress=gr.Progress(track_tqdm=True),
    ):
        start = timeit.default_timer()
        og_encode = None

        try:
            if self.fp16:
                print("Generating using float16 precision.")

            use_init = init_audio is not None

            preview_images = []

            model_conditionings = (
                self.current_loaded_model_config.get("model", {})
                .get("diffusion", {})
                .get("cross_attention_cond_ids", [])
            )

            all_conditioning = [
                {
                    "prompt": prompt,
                    "seconds_start": seconds_start,
                    "seconds_total": seconds_total,
                }
            ]

            # Filter out conditionings that aren't in the model

            conditioning = [
                {k: v for k, v in cond.items() if k in model_conditionings}
                for cond in all_conditioning
            ] * batch_size

            seed = int(seed)

            if self.current_loaded_model.pretransform is not None:
                if tiled_encode:
                    og_encode = tyler_patch(
                        module=self.current_loaded_model.pretransform,
                        function_name="encode",
                        window_length=tiled_encode_window_length,
                    )

            if not use_init:
                init_audio = None

            if init_audio is not None:
                in_sr, init_audio = init_audio

                # Turn into torch tensor, converting from int16 to float32
                init_audio = torch.from_numpy(init_audio).float().div(32767)

                if init_audio.dim() == 1:
                    init_audio = init_audio.unsqueeze(0)  # [1, n]
                elif init_audio.dim() == 2:
                    init_audio = init_audio.transpose(0, 1)  # [n, 2] -> [2, n]
                if self.fp16:
                    print("Converting init audio to fp16")
                    init_audio = init_audio.half()

                init_audio = (in_sr, init_audio)

            def progress_callback(callback_info):
                global preview_images
                denoised = callback_info["denoised"]
                current_step = callback_info["i"]
                sigma = callback_info["sigma"]

                if (current_step - 1) % preview_every == 0:
                    if self.current_loaded_model.pretransform is not None:
                        denoised = (
                            self.current_loaded_model.pretransform.decode(denoised)
                            if not tiled_decode
                            else TYLER(
                                denoised,
                                self.current_loaded_model.pretransform.decode,
                                window_length=65536
                                // self.current_loaded_model.pretransform.downsampling_ratio,
                            )
                        )

                    denoised = rearrange(denoised, "b d n -> d (b n)")

                    denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

                    audio_spectrogram = audio_spectrogram_image(
                        denoised,
                        sample_rate=self.current_loaded_model_config["sample_rate"],
                    )

                    preview_images.append(
                        (audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})")
                    )

            seed = random.randint(0, 2**32 - 1) if seed == -1 else seed

            if (
                self.current_loaded_model_config.get("model_type", {})
                == "diffusion_cond"
            ):
                latents = generate_diffusion_cond(
                    self.current_loaded_model,
                    conditioning=conditioning,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    batch_size=batch_size,
                    sample_size=sample_size,
                    sample_rate=self.current_loaded_model_config["sample_rate"],
                    seed=seed,
                    device=torch.device(self.current_device),
                    sampler_type=sampler_type,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    init_audio=init_audio,
                    init_noise_level=init_noise_level,
                    callback=progress_callback if preview_every is not None else None,
                    scale_phi=cfg_rescale,
                    return_latents=True,
                )

            else:
                latents = generate_diffusion_uncond(
                    self.current_loaded_model,
                    steps=steps,
                    batch_size=batch_size,
                    sample_size=sample_size,
                    seed=seed,
                    device=torch.device(self.current_device),
                    sampler_type=sampler_type,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    init_audio=init_audio,
                    init_noise_level=init_noise_level,
                    callback=progress_callback if preview_every is not None else None,
                    return_latents=True,
                )

            if self.fp16:
                latents = latents.half()

            if og_encode:
                self.current_loaded_model.pretransform.encode = og_encode

            with torch.no_grad():
                if split_decode and not tiled_decode:
                    print(f"Split decoding batch of {len(latents)} samples.")
                    fakes = []
                    for fake_latent in tqdm(
                        latents, desc="Decoding..", total=len(latents)
                    ):
                        fake_latent = fake_latent.unsqueeze(0)
                        fake = self.current_loaded_model.pretransform.decode(
                            fake_latent
                        )
                        fakes.append(fake.cpu())
                        del fake
                        torch.cuda.empty_cache()
                        gc.collect()
                    audios = torch.cat(fakes, dim=0)
                else:
                    audios = (
                        self.current_loaded_model.pretransform.decode(latents)
                        if not tiled_decode
                        else TYLER(
                            latents,
                            self.current_loaded_model.pretransform.decode,
                            window_length=tiled_decode_window_length
                            // self.current_loaded_model.pretransform.downsampling_ratio,
                            show_tqdm=True,
                        )
                    )

            # audio = rearrange(audio, "b d n -> d (b n)")

            np_audios = []

            audio_specs = []

            for audio in audios:
                if self.fp16:
                    audio = audio.float()
                if tame:
                    audio = audio.clamp(-1, 1)
                audio = audio.mul(32767).to(torch.int16).cpu()

                audio_spectrogram = audio_spectrogram_image(
                    audio, sample_rate=self.current_loaded_model_config["sample_rate"]
                )

                audio_specs.append(audio_spectrogram)
                np_audios.append(
                    (self.current_loaded_model_config["sample_rate"], audio.numpy().T)
                )

            # if numpy audios is less than max outputs, pad with empty audio
            np_audios += [
                (self.current_loaded_model_config["sample_rate"], np.zeros(1))
            ] * (self.max_outputs - len(np_audios))

            end = timeit.default_timer()
            gr.Info(f"Generated audio in {end - start:.2f} seconds.")

            return np_audios + [audio_specs, *preview_images]
        except Exception as e:
            gr.Error(f"Error generating audio: {e}")
            print(e)
            traceback.print_exc()
            if og_encode:
                self.current_loaded_model.pretransform.encode = og_encode
            return [None] * (self.max_outputs + 2)

    def create_sampling_ui(self):
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary", scale=1)
                    batch_size_slider = gr.Slider(
                        minimum=1,
                        maximum=self.max_outputs,
                        step=1,
                        value=3,
                        label="📦 Batch size",
                        scale=6,
                    )
                self.conditioning_section = gr.Column(variant="panel", visible=False)
                with self.conditioning_section:
                    self.prompt_box = gr.Textbox(
                        show_label=False, placeholder="Prompt", scale=6, lines=2
                    )

                    tags = self.load_tags()
                    tag_dropdown = gr.Dropdown(
                        tags,
                        label="Add tags",
                        multiselect=True,
                        filterable=True,
                        visible=tags,
                    )
                    tag_dropdown.select(
                        self.add_tag_to_promptbox,
                        inputs=[self.prompt_box, tag_dropdown],
                        outputs=self.prompt_box,
                        api_name=False,
                        queue=False,
                    ).then(
                        # fn is lambda that returns None to clear the tag dropdown
                        fn=lambda: gr.update(value=[]),
                        inputs=None,
                        outputs=tag_dropdown,
                        api_name=False,
                        queue=False,
                    )

                    examples = self.load_examples()
                    if examples:
                        inp_ex = []
                        for example in examples:
                            inp_ex.append([example])
                        gr.Examples(
                            inp_ex,
                            label="Examples",
                            inputs=[self.prompt_box],
                        )
                    self.timing_acc = gr.Accordion(
                        "⌛ Timing Controls", open=False, visible=False
                    )
                    with self.timing_acc:
                        with gr.Column(variant="compact"):
                            gr.Markdown(
                                """
                            The diffusion model will try to generate a segment of an imaginary track of set total seconds, 
                            starting at the set start second. Lower start seconds will make it try to generate intros, 
                            higher start seconds will make it try to generate drops and bridges.
                                """
                            )
                            # Timing controls
                            self.sec_start_slider = gr.Slider(
                                minimum=0,
                                maximum=512,
                                step=1,
                                value=30,
                                label="Imaginary Start Seconds",
                            )
                            self.sec_total_slider = gr.Slider(
                                minimum=0,
                                maximum=512,
                                step=1,
                                value=200,
                                label="Imaginary Total Seconds",
                            )
                with gr.Accordion("⚙️ Advanced", open=False):
                    # Tame checkbox
                    tame_checkbox = gr.Checkbox(
                        label="Tame",
                        value=False,
                    )
                    with gr.Row():
                        # Steps slider
                        steps_nr = gr.Number(
                            minimum=1,
                            maximum=500,
                            value=75,
                            label="Steps",
                            scale=1,
                            precision=1,
                        )

                        # CFG scale
                        self.cfg_scale_slider = gr.Slider(
                            minimum=0.0,
                            maximum=25.0,
                            step=0.1,
                            value=7.5,
                            label="CFG scale",
                        )

                    with gr.Group():
                        with gr.Row():
                            # sample size slider
                            self.sample_size_slider = gr.Slider(
                                minimum=32768,
                                maximum=32768 * 1024,
                                step=32768,
                                value=32768,
                                label="Sample size",
                                scale=6,
                            )
                            # reset sample_size button
                            with gr.Group():
                                reset_sample_size_button = gr.Button("🔃 Reset", scale=1)
                                reset_sample_size_button.click(
                                    fn=self.reset_sample_size,
                                    outputs=[self.sample_size_slider],
                                    api_name=False,
                                    queue=False,
                                )
                                calculate_sample_size_button = gr.Button(
                                    "⌚ Calculate Seconds", scale=1
                                )
                                calculate_sample_size_button.click(
                                    fn=self.calculate_sample_size,
                                    inputs=[self.sample_size_slider],
                                    api_name=False,
                                )

                    with gr.Accordion("Sampler params", open=True):
                        # Seed
                        with gr.Row():
                            seed_box = gr.Number(
                                label="Seed (set to -1 for random seed)",
                                value=-1,
                                precision=1,
                                scale=6,
                            )
                            sampler_type_dropdown = gr.Dropdown(
                                [
                                    "dpmpp-2m-sde",
                                    "k-heun",
                                    "k-lms",
                                    "k-dpmpp-2s-ancestral",
                                    "k-dpm-2",
                                    "k-dpm-fast",
                                ],
                                label="Sampler type",
                                value="dpmpp-2m-sde",
                                scale=6,
                            )

                        # Sampler params
                        with gr.Row():
                            self.sigma_min = gr.Number(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.3,
                                label="Sigma min",
                                scale=1,
                            )
                            sigma_max_nr = gr.Number(
                                minimum=0.0,
                                maximum=500.0,
                                value=100,
                                label="Sigma max",
                                scale=1,
                            )
                            self.cfg_rescale_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.3,
                                step=0.01,
                                value=0.35,
                                label="CFG rescale amount",
                                scale=6,
                            )
                        with gr.Accordion("Tiled Decoding", open=False):
                            with gr.Row():
                                split_decode_checkbox = gr.Checkbox(
                                    label="Split decode",
                                    value=True,
                                )
                                tiled_decode_checkbox = gr.Checkbox(
                                    label="Tiled decode",
                                    value=False,
                                )
                                tiled_decode_window_length_slider = gr.Slider(
                                    label="Tiled window samples",
                                    minimum=32768,
                                    maximum=32768 * 32,
                                    step=32768,
                                    value=65536,
                                )

                with gr.Accordion("🎵 Init audio", open=False):
                    with gr.Accordion("Tiled Encoding", open=False):
                        tiled_encode_checkbox = gr.Checkbox(
                            label="Tiled encode", value=False
                        )
                        tiled_encode_window_length_slider = gr.Slider(
                            label="Tiled window samples",
                            minimum=32768,
                            maximum=32768 * 32,
                            step=32768,
                            value=65536,
                        )
                    with gr.Group():
                        self.init_audio_input = gr.Audio(
                            label="Init audio", elem_id="small_audio"
                        )
                        init_noise_level_slider = gr.Slider(
                            minimum=0.0,
                            maximum=100.0,
                            step=0.01,
                            value=6.0,
                            label="Init noise level",
                        )

            with gr.Column(variant="panel"):
                self.audio_outputs = [
                    self.create_audio_output(
                        label=f"Output #{i+1}",
                        visible=False if i > 2 else True,
                    )
                    for i in range(self.max_outputs)
                ]
                with gr.Accordion("🌈 Spectrograms", open=False):
                    audio_spectrogram_output = gr.Gallery(
                        label="Output spectrogram", show_label=False
                    )
                self.create_send_to_input_list(self.audio_outputs)

            generate_button.click(
                self.set_output_batch_size,
                inputs=[batch_size_slider],
                outputs=self.audio_outputs,
                api_name=False,
            )

        generate_button.click(
            fn=self.generate,
            inputs=[
                self.prompt_box,
                self.sec_start_slider,
                self.sec_total_slider,
                self.cfg_scale_slider,
                tame_checkbox,
                steps_nr,
                seed_box,
                sampler_type_dropdown,
                self.sigma_min,
                sigma_max_nr,
                self.sample_size_slider,
                self.cfg_rescale_slider,
                self.init_audio_input,
                init_noise_level_slider,
                batch_size_slider,
                split_decode_checkbox,
                tiled_encode_checkbox,
                tiled_encode_window_length_slider,
                tiled_decode_checkbox,
                tiled_decode_window_length_slider,
            ],
            outputs=self.audio_outputs + [audio_spectrogram_output],
            api_name="generate",
        )

    def create_txt2audio_ui(self):
        with gr.Box(visible=False) as ui:
            self.create_sampling_ui()

        return ui

    def autoencoder_process(
        self,
        audio,
        tiled_encode,
        tiled_decode,
        skip_decode,
        tiled_window_length,
        # tiled_hop_divisor,
        progress=gr.Progress(track_tqdm=True),
    ):
        if audio is None:
            gr.Error("No audio provided.")
            return None
        start = timeit.default_timer()
        if self.fp16:
            print("Processing using float16 precision.")
        in_sr, audio = audio

        audio = torch.from_numpy(audio).float().div(32767)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.transpose(0, 1)

        audio_length = audio.shape[1]

        # Pad to multiple of model's downsampling ratio
        pad_length = (
            self.current_loaded_model.downsampling_ratio
            - (audio_length % self.current_loaded_model.downsampling_ratio)
        ) % self.current_loaded_model.downsampling_ratio
        audio = F.pad(audio, (0, pad_length))

        audio = prepare_audio(
            audio,
            in_sr=in_sr,
            target_sr=self.current_loaded_model_config["sample_rate"],
            target_length=audio.shape[1],
            target_channels=self.current_loaded_model.io_channels,
            device=self.current_device,
        )

        if self.fp16:
            audio = audio.half()

        with torch.no_grad():
            original_length = audio.shape[-1]
            audio = (
                TYLER(
                    audio,
                    self.current_loaded_model.encode,
                    window_length=tiled_window_length,
                    use_tqdm=True,
                )
                if tiled_encode
                else self.current_loaded_model.encode(audio)
            )

            if not skip_decode:
                if self.fp16:
                    audio = audio.half()

                audio = (
                    TYLER(
                        audio,
                        self.current_loaded_model.decode,
                        window_length=tiled_window_length
                        // self.current_loaded_model.downsampling_ratio,
                        use_tqdm=True,
                    )
                    if tiled_decode
                    else self.current_loaded_model.decode(audio)
                )
                audio = audio[..., :original_length]
                if self.fp16:
                    audio = audio.float()
                audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            else:
                audio = audio.cpu()
        audio_output = (
            self.current_loaded_model_config["sample_rate"],
            audio.numpy().T,
        )

        time = timeit.default_timer() - start
        gr.Info(f"Processed audio in {time:.2f} seconds.")

        return audio_output

    def create_autoencoder_ui(self):
        with gr.Box(visible=False) as ui:
            with gr.Row(equal_height=False):
                with gr.Group():
                    skip_decode_checkbox = gr.Checkbox(label="Skip decode", value=False)
                    with gr.Accordion("Tiled Processing", open=True):
                        with gr.Row():
                            tiled_encode_checkbox = gr.Checkbox(
                                label="Tiled encode", value=False
                            )
                            tiled_decode_checkbox = gr.Checkbox(
                                label="Tiled decode", value=False
                            )
                        tiled_window_length_slider = gr.Slider(
                            label="Tiled window samples",
                            minimum=32768,
                            maximum=32768 * 32,
                            step=32768,
                            value=65536,
                        )
                        # tiled_hop_divisor_slider = gr.Slider(
                        #     label="Tiled hop divisor",
                        #     minimum=1,
                        #     maximum=16,
                        #     step=0.1,
                        #     value=4,
                        # )
                with gr.Column(variant="panel"):
                    input_audio = gr.Audio(label="Input audio")
                    output_audio = self.create_audio_output(label="Output audio")
                    process_button = gr.Button("Process", variant="primary", scale=1)
                    process_button.click(
                        fn=self.autoencoder_process,
                        inputs=[
                            input_audio,
                            tiled_encode_checkbox,
                            tiled_decode_checkbox,
                            skip_decode_checkbox,
                            tiled_window_length_slider,
                            # tiled_hop_divisor_slider,
                        ],
                        outputs=output_audio,
                        api_name="aec_process",
                    )

        return ui

    def check_ready_state(
        self, mod_select_value, cfg_select_value, pretrans_select_value=""
    ):
        # Initialize variables
        model_filename = os.path.basename(mod_select_value) if mod_select_value else ""
        pretransform_filename = (
            os.path.basename(pretrans_select_value) if pretrans_select_value else ""
        )
        same_ckpt = self.current_loaded_model_path == mod_select_value

        same_pretransform = (
            self.current_loaded_pretransform_path == pretrans_select_value
        )

        self.current_set_model_path = mod_select_value
        self.current_set_pretransform_path = pretrans_select_value

        # Check if model and config are selected
        if mod_select_value is None:
            return False, gr.HTML(
                self.create_info_table(msg="🔴 Please select a model file!")
            )
        elif not mod_select_value.endswith(".ckpt"):
            return False, gr.HTML(
                self.create_info_table(
                    model_filename=model_filename,
                    pretransform_filename=pretransform_filename,
                    msg="🔴 Please select a valid model checkpoint (.ckpt) file!",
                )
            )

        if not self.embedded_config and not self.current_set_model_config:
            if cfg_select_value is None:
                return False, gr.HTML(
                    self.create_info_table(msg="🔴 Please select a config file!")
                )

            self.current_set_model_config = cfg_select_value

            # Load model configuration
            with open(cfg_select_value) as f:
                model_config = json.load(f)
        else:
            model_config = self.current_loaded_model_config

        if "model_type" not in model_config:
            return False, gr.HTML(
                self.create_info_table(
                    model_filename=model_filename,
                    pretransform_filename=pretransform_filename,
                    msg="🔴 Please select a valid config file!",
                )
            )

        # Load the model if it's different from the currently loaded one
        if not same_ckpt or not same_pretransform:
            load_success = self.load_model(
                model_config,
                mod_select_value,
                pretrans_select_value,
                load_pretransform_only=same_ckpt and not same_pretransform,
            )
            if not load_success:
                return False, gr.HTML(
                    self.create_info_table(
                        model_filename=model_filename,
                        pretransform_filename=pretransform_filename,
                        model_config=model_config,
                        msg="🔴 Error loading model!",
                    )
                )

        # Model is successfully loaded and ready
        return True, gr.HTML(
            self.create_info_table(
                model_filename=model_filename,
                pretransform_filename=pretransform_filename,
                model_config=model_config,
                msg="🟢 Ready!",
            )
        )

    def set_output_batch_size(self, slider):
        return [gr.Audio(visible=True)] * slider + [gr.Audio(visible=False)] * (
            self.max_outputs - slider
        )

    def set_visibility_by_model_type(self, model_type):
        aec_visibility = False
        txt2audio_visibility = False
        blank_visibility = False

        if model_type in ["diffusion_uncond", "diffusion_cond"]:
            txt2audio_visibility = True
        elif model_type == "autoencoder":
            aec_visibility = True

        return [
            gr.Box(visible=aec_visibility),
            gr.Box(visible=txt2audio_visibility),
            gr.Box(visible=blank_visibility),
        ]

    def set_global_values(
        self, mod_select_value, cfg_select_value, pretrans_select_value=""
    ):
        ready, info_box = self.check_ready_state(
            mod_select_value, cfg_select_value, pretrans_select_value
        )
        sec_slider_default = gr.Slider(visible=False)
        show_prompt = False

        visibilities = [gr.Box(visible=False)] * 2 + [gr.Box(visible=True)]
        sec_slider_updates = [
            gr.Accordion(label="⌛ Timing Controls", visible=False),
            sec_slider_default,
            sec_slider_default,
        ]
        shown_tab = (
            "model_settings" if "settings" not in self.hidden else "model_process"
        )
        loaded_sample_size = 65536

        cond_updates = [
            gr.Slider(visible=False),
            gr.Slider(visible=False),
            gr.Column(visible=False),
            gr.Number(0.05),
        ]

        if ready:
            model_type = self.current_loaded_model_config["model_type"]
            shown_tab = "model_process" if not self.current_wrapped else shown_tab
            visibilities = (
                self.set_visibility_by_model_type(model_type)
                if not self.current_wrapped
                else visibilities
            )
            loaded_sample_size = self.current_loaded_model_config.get(
                "sample_size", 65536
            )

            cond = model_type == "diffusion_cond"

            cond_updates = [
                gr.Slider(visible=cond),
                gr.Slider(visible=cond),
                gr.Column(visible=cond),
                gr.Number(0.3 if cond else 0.05),
            ]

            # Update second sliders based on conditional types
            cond_types = (
                self.current_loaded_model_config.get("model", {})
                .get("diffusion", {})
                .get("cross_attention_cond_ids", [])
            )

            if "seconds_start" in cond_types or "seconds_total" in cond_types:
                sec_slider_updates[0] = gr.Accordion(visible=True)

                if "seconds_start" in cond_types:
                    sec_slider_updates[1] = gr.Slider(visible=True)
                if "seconds_total" in cond_types:
                    sec_slider_updates[2] = gr.Slider(visible=True)
            if "prompt" in cond_types:
                show_prompt = True

        return (
            [info_box]
            + cond_updates  # cond switch
            + visibilities  # model type generation UIs
            + [gr.Tabs(selected=shown_tab)]  # settings
            + sec_slider_updates  # seconds sliders
            + [gr.Textbox(visible=show_prompt)]  # prompt
            + [gr.Slider(value=loaded_sample_size)]  # sample size
        )

    def on_model_change(self, model_selector, config_selector, pretransform_selector):
        config = config_selector.name if config_selector else None
        pretrans_select_values = (
            pretransform_selector if pretransform_selector else None
        )
        return self.set_global_values(model_selector, config, pretrans_select_values)

    def on_device_change(self, device_selector):
        """Set the current device for processing."""

        def set_device(device_name):
            """Utility function to set the device and log the info."""
            if self.current_device != device_name:
                self.current_device = device_name
                if self.current_loaded_model:
                    gr.Info(f"Moving model to {device_name}")
                    self.current_loaded_model.to(self.current_device)
                gr.Info(f"Device set to {device_name}")
            return

        # For standard device names like 'cuda', 'cpu', and 'mps'.
        if device_selector in {"cuda", "cpu", "mps"}:
            set_device(device_selector)
            return

        # For specific CUDA devices like 'cuda:0', 'cuda:1', etc.
        if device_selector.startswith("cuda:"):
            try:
                device_id = int(device_selector.split(":")[1])
            except ValueError:
                gr.Error("Invalid device!")
                return

            if 0 <= device_id < torch.cuda.device_count():
                set_device(device_selector)
                return
            else:
                gr.Error("Invalid device!")
                return

    def on_fp16_change(self, fp16_checkbox):
        if self.fp16 == fp16_checkbox:
            return
        self.fp16 = fp16_checkbox
        infostring = (
            "Precision set to float16." if self.fp16 else "Precision set to float32."
        )
        if self.current_loaded_model:
            self.current_loaded_model.half() if self.fp16 else self.current_loaded_model.float()
        gr.Info(infostring)
        return

    def get_models(self):
        models = []
        print(f"Looking for models in {self.init_model_dirs}")
        for model_dir in self.init_model_dirs:
            models += glob.glob(model_dir + "/**/*.ckpt", recursive=True)

        print(f"Found {len(models)} models")
        return models + [""]

    def get_models_tupled(self):
        models = self.get_models()
        models = [model for model in models if model != ""]
        tupled = [os.path.splitext(os.path.basename(model))[0] for model in models]
        tupled = [("None", "")] + list(zip(tupled, models))
        return tupled

    def refresh_models(self):
        return [gr.Dropdown(choices=self.get_models_tupled())] * 2

    def autofind_config(self, model_path, config_selector):
        self.embedded_config = False
        # check inside the model
        if not model_path:
            return None

        # check if .ckpt or .safetensors is in model_path string
        if ".ckpt" not in model_path and ".safetensors" not in model_path:
            return None

        json_path = model_path.replace(".ckpt", ".json")

        if os.path.isfile(json_path):
            with open(json_path) as f:
                model_config = json.load(f)
                if "model_type" in model_config:
                    self.current_set_model_config = json_path
                    self.current_loaded_model_config = model_config
                    print(f"Auto-located config file {json_path}")
                    gr.Info(f"Auto-located config file {json_path}")
                    return json_path

        # check if its inside the model

        checked = torch.load(model_path, map_location="cpu")
        if "model_config" in list(checked.keys()):
            self.current_set_model_config = "internal"
            self.current_loaded_model_config = checked["model_config"]
            self.embedded_config = True
            print(f"Auto-located config in model.")
            return None

        self.current_set_model_config = (
            config_selector.name if config_selector else None
        )

        return self.current_set_model_config

    def unwrap_model(self):
        config_loaded = (
            True
            if self.current_set_model_config or self.current_loaded_model_config
            else False
        )
        if not config_loaded or not self.current_set_model_path:
            gr.Error("Please select a model and config file first!")
            print("Please select a model and config file first!")
            return [gr.Dropdown(choices=self.get_models_tupled())] * 2
        try:
            gr.Info(f"Unwrapping model with config {self.current_set_model_config}")
            if self.current_set_model_config != "internal":
                with open(self.current_set_model_config) as f:
                    model_config = json.load(f)
            else:
                model_config = self.current_loaded_model_config

            model = (
                create_model_from_config(model_config)
                if self.current_loaded_model is None
                else self.current_loaded_model
            )

            model_type = model_config.get("model_type", None)

            assert (
                model_type is not None
            ), "model_type must be specified in model config"

            training_config = model_config.get("training", None)

            if model_type == "autoencoder":
                from stable_audio_tools.training.autoencoders import (
                    AutoencoderTrainingWrapper,
                )

                ema_copy = None

                if training_config.get("use_ema", False):
                    ema_copy = create_model_from_config(model_config)
                    ema_copy = create_model_from_config(
                        model_config
                    )  # I don't know why this needs to be called twice but it broke when I called it once

                    # Copy each weight to the ema copy
                    for name, param in model.state_dict().items():
                        if isinstance(param, Parameter):
                            # backwards compatibility for serialized parameters
                            param = param.data
                        ema_copy.state_dict()[name].copy_(param)

                use_ema = training_config.get("use_ema", False)

                training_wrapper = AutoencoderTrainingWrapper.load_from_checkpoint(
                    self.current_set_model_path,
                    autoencoder=model,
                    strict=False,
                    loss_config=training_config["loss_configs"],
                    use_ema=training_config["use_ema"],
                    ema_copy=ema_copy if use_ema else None,
                )
            elif model_type == "diffusion_uncond":
                from stable_audio_tools.training.diffusion import (
                    DiffusionUncondTrainingWrapper,
                )

                training_wrapper = DiffusionUncondTrainingWrapper.load_from_checkpoint(
                    self.current_set_model_path, model=model, strict=False
                )
            elif model_type == "diffusion_autoencoder":
                from stable_audio_tools.training.diffusion import (
                    DiffusionAutoencoderTrainingWrapper,
                )

                training_wrapper = (
                    DiffusionAutoencoderTrainingWrapper.load_from_checkpoint(
                        self.current_set_model_path, model=model, strict=False
                    )
                )
            elif model_type == "diffusion_cond":
                from stable_audio_tools.training.diffusion import (
                    DiffusionCondTrainingWrapper,
                )

                training_wrapper = DiffusionCondTrainingWrapper.load_from_checkpoint(
                    self.current_set_model_path, model=model, strict=False
                )
            elif model_type == "diffusion_cond_inpaint":
                from stable_audio_tools.training.diffusion import (
                    DiffusionCondInpaintTrainingWrapper,
                )

                training_wrapper = (
                    DiffusionCondInpaintTrainingWrapper.load_from_checkpoint(
                        self.current_set_model_path, model=model, strict=False
                    )
                )
            else:
                raise ValueError(f"Unknown model type {model_type}")

            new_name = self.current_set_model_path.replace(".ckpt", "_unwrapped.ckpt")

            training_wrapper.export_model(new_name)

            # hack in model config
            print("Embedding config..")
            set_config = torch.load(new_name, map_location="cpu")
            set_config["model_config"] = model_config
            torch.save(set_config, new_name)

            gr.Info(f"Unwrapped model to {new_name}")
            print(f"Unwrapped model to {new_name}")
            training_wrapper.to("cpu")
            del training_wrapper
            torch.cuda.empty_cache()
            gc.collect()

            return [gr.Dropdown(choices=self.get_models_tupled())] * 2
        except Exception as e:
            gr.Error(f"Error unwrapping model: {e}")
            print(f"Error unwrapping model: {e}")
            traceback.print_exc()
            return [gr.Dropdown(choices=self.get_models_tupled())] * 2

    # Selector

    def setup_selector_changes(self, widget):
        widget.select(
            self.on_model_change,
            inputs=[
                self.model_selector,
                self.config_selector,
                self.pretransform_selector,
            ],
            outputs=[
                self.info_box,
                self.cfg_scale_slider,
                self.cfg_rescale_slider,
                self.conditioning_section,
                self.sigma_min,
                self.aec_ui,
                self.txt2audio_ui,
                self.blank_ui,
                self.tab_control,
                self.timing_acc,
                self.sec_start_slider,
                self.sec_total_slider,
                self.prompt_box,
                self.sample_size_slider,
            ],
            api_name=False,
        )

    def setup_general_selector(self, label, change_fn):
        selector = gr.Number(label=label, precision=1)
        selector.change(
            change_fn,
            inputs=[selector],
            outputs=[],
            api_name=False,
            queue=False,
        )
        return selector

    def setup_textbox_selector(self, label, change_fn):
        selector = gr.Textbox(label=label, scale=1)
        selector.change(
            change_fn,
            inputs=[selector],
            outputs=[],
            api_name=False,
            queue=False,
        )
        return selector

    def create_selector_header(self, visible=True):
        with gr.Box(visible=visible) as header:
            with gr.Group():
                # Setup top row containing info, model, and config selectors.
                with gr.Row(variant="compact"):
                    with gr.Column():
                        self.info_box = gr.HTML(self.create_info_table())
                        self.device_selector = self.setup_textbox_selector(
                            "Device", self.on_device_change
                        )
                        self.fp16_checkbox = gr.Checkbox(
                            label="FP16", value=False, scale=1
                        )
                        self.fp16_checkbox.change(
                            self.on_fp16_change,
                            inputs=[self.fp16_checkbox],
                            outputs=[],
                            api_name=False,
                            queue=False,
                        )
                    with gr.Column():
                        with gr.Row(equal_height=False):
                            self.model_selector = gr.Dropdown(
                                value="",
                                label="Model Ckpt",
                                scale=1,
                                choices=self.get_models_tupled(),
                            )

                        self.pretransform_selector = gr.Dropdown(
                            value="",
                            label="Pretransform Ckpt",
                            scale=1,
                            choices=self.get_models_tupled(),
                        )
                        self.config_selector = gr.File(
                            label="Config File", scale=1, file_types=[".json"]
                        )
                        with gr.Row():
                            unwrap_button = gr.Button(
                                "Unwrap", variant="primary", scale=6
                            )
                            refresh_models_button = gr.Button(
                                "🔃", scale=1, elem_id="small_button"
                            )
                            refresh_button = gr.Button("Reload", scale=1)
                            unload_button = gr.Button("Unload", scale=1)
                            unwrap_button.click(
                                fn=self.unwrap_model,
                                outputs=[
                                    self.model_selector,
                                    self.pretransform_selector,
                                ],
                                api_name=False,
                            )
                            refresh_models_button.click(
                                fn=self.refresh_models,
                                outputs=[
                                    self.model_selector,
                                    self.pretransform_selector,
                                ],
                                api_name=False,
                                queue=False,
                            )
                            refresh_button.click(
                                fn=self.refresh_model,
                                outputs=self.make_loadable_vars(),
                                api_name=False,
                            )
                            unload_button.click(
                                fn=self.on_unload,
                                outputs=self.make_loadable_vars()
                                + [
                                    self.model_selector,
                                    self.pretransform_selector,
                                    self.config_selector,
                                ],
                                api_name=False,
                            )

                        self.model_selector.select(
                            self.autofind_config,
                            inputs=[self.model_selector, self.config_selector],
                            outputs=[self.config_selector],
                            api_name=False,
                        )

                        # Setting up change events for selectors
                        self.setup_selector_changes(self.model_selector)
                        self.setup_selector_changes(self.pretransform_selector)
                        self.setup_selector_changes(self.config_selector)
                with gr.Accordion(
                    "Train Selected Config",
                    open=False,
                    visible=not "train" in self.hidden,
                ):
                    tr_name = gr.Textbox(
                        "My Audio Model", label="Project Name", scale=1
                    )
                    tr_wandb_key = gr.Textbox(
                        "",
                        label="Wandb API Key",
                        scale=1,
                        type="password",
                    )
                    with gr.Row():
                        tr_batch_size = gr.Number(
                            3, label="Batch Size", scale=1, precision=1
                        )
                        tr_accum_batches = gr.Number(
                            4, label="Accum Batches", scale=1, precision=1
                        )
                    with gr.Row():
                        tr_num_gpus = gr.Number(
                            1, label="Num GPUs", scale=1, precision=1
                        )
                        tr_num_workers = gr.Number(
                            8, label="Num Workers", scale=1, precision=1
                        )
                        tr_num_nodes = gr.Number(
                            1, label="Num Nodes", scale=1, precision=1
                        )
                    tr_strategy = gr.Textbox("", label="Strategy", scale=1)
                    tr_seed = gr.Number(42, label="Seed", scale=1, precision=1)
                    tr_precision = gr.Dropdown(
                        value="16", label="Precision", scale=1, choices=["16", "32"]
                    )
                    tr_checkpoint_every = gr.Number(
                        250, label="Checkpoint Every", scale=1, precision=1
                    )

                    ckpt_action = gr.Radio(
                        value="Train From Scratch",
                        label="Resume/Restart",
                        scale=1,
                        choices=[
                            "Train From Scratch",
                            "Resume From Loaded Model",
                            "Restart From Loaded Pretrained",
                        ],
                    )
                    tr_dataset_config = gr.File(
                        label="Dataset Config", scale=1, file_types=[".json"]
                    )
                    gr.Markdown(
                        """## Note: Initial selected model, pretransform and config will be used for training."""
                    )
                    train_button = gr.Button("Train", variant="primary", scale=1)
                    train_button.click(
                        fn=self.train_model,
                        inputs=[
                            tr_name,
                            tr_wandb_key,
                            tr_batch_size,
                            tr_num_gpus,
                            tr_num_nodes,
                            tr_strategy,
                            tr_num_workers,
                            tr_seed,
                            tr_precision,
                            tr_accum_batches,
                            tr_checkpoint_every,
                            ckpt_action,
                            tr_dataset_config,
                        ],
                        outputs=[],
                        api_name=False,
                    )

        return header

    def train_model(
        self,
        name,
        wandb_key,
        batch_size,
        num_gpus,
        num_nodes,
        strategy,
        num_workers,
        seed,
        precision,
        accum_batches,
        checkpoint_every,
        ckpt_action,
        dataset_config,
    ):
        try:
            # args list to dict then to simple namespace
            args = SimpleNamespace(
                **{
                    "name": name,
                    "wandb_key": wandb_key,
                    "batch_size": batch_size,
                    "num_gpus": num_gpus,
                    "num_nodes": num_nodes,
                    "strategy": strategy,
                    "num_workers": num_workers,
                    "seed": seed,
                    "precision": int(precision),
                    "accum_batches": accum_batches,
                    "checkpoint_every": checkpoint_every,
                    "ckpt_action": ckpt_action,
                    "dataset_config": dataset_config.name,
                    "pretransform_ckpt_path": self.current_set_pretransform_path,
                }
            )

            args.pretrained_ckpt_path = (
                self.current_set_model_path
                if args.ckpt_action == "Restart From Loaded Pretrained"
                else ""
            )
            args.ckpt_path = (
                self.current_set_model_path
                if args.ckpt_action == "Resume From Loaded Model"
                else ""
            )

            print(args)

            if not self.current_set_model_config:
                gr.Error("Please select a config file first!")
                return

            # unpack args into dict
            gr.Info("Starting training, please see console for output..")

            class ExceptionCallback(pl.Callback):
                def on_exception(self, trainer, module, err):
                    print(f"{type(err).__name__}: {err}")

            class ModelConfigEmbedderCallback(pl.Callback):
                def __init__(self, model_config):
                    self.model_config = model_config

                def on_save_checkpoint(
                    self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint
                ) -> None:
                    checkpoint["model_config"] = self.model_config

            torch.manual_seed(args.seed)

            # Get JSON config from args.model_config
            with open(self.current_set_model_config) as f:
                model_config = json.load(f)

            with open(args.dataset_config) as f:
                dataset_config = json.load(f)

            train_dl = create_dataloader_from_configs_and_args(
                model_config, args, dataset_config
            )

            model = create_model_from_config(model_config)

            if args.pretrained_ckpt_path:
                copy_state_dict(
                    model, torch.load(args.pretrained_ckpt_path)["state_dict"]
                )

            if self.current_set_pretransform_path:
                print("Loading pretransform from checkpoint")
                model.pretransform.load_state_dict(
                    torch.load(self.current_set_pretransform_path)["state_dict"]
                )
                print("Done loading pretransform from checkpoint")

            training_wrapper = create_training_wrapper_from_config(model_config, model)

            exc_callback = ExceptionCallback()
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                every_n_train_steps=args.checkpoint_every, save_top_k=-1
            )

            demo_callback = create_demo_callback_from_config(
                model_config, demo_dl=train_dl
            )

            config_embedder_callback = ModelConfigEmbedderCallback(model_config)

            if args.wandb_key:
                wandb.login(key=args.wandb_key)
                wandb_logger = pl.loggers.WandbLogger(project=args.name)
                wandb_logger.watch(training_wrapper)
                # Combine args and config dicts
                args_dict = vars(args)
                args_dict.update({"model_config": model_config})
                args_dict.update({"dataset_config": dataset_config})
                push_wandb_config(wandb_logger, args_dict)
            else:
                wandb_logger = False

            # Set multi-GPU strategy if specified
            if args.strategy:
                if args.strategy == "deepspeed":
                    from pytorch_lightning.strategies import DeepSpeedStrategy

                    strategy = DeepSpeedStrategy(
                        stage=2,
                        contiguous_gradients=True,
                        overlap_comm=True,
                        reduce_scatter=True,
                        reduce_bucket_size=5e8,
                        allgather_bucket_size=5e8,
                        load_full_weights=True,
                    )
                else:
                    strategy = args.strategy
            else:
                strategy = "ddp" if args.num_gpus > 1 else "auto"

            trainer = pl.Trainer(
                devices=args.num_gpus,
                accelerator="gpu",
                num_nodes=args.num_nodes,
                strategy=strategy,
                precision=args.precision,
                accumulate_grad_batches=args.accum_batches,
                callbacks=[
                    ckpt_callback,
                    demo_callback,
                    exc_callback,
                    config_embedder_callback,
                ],
                logger=wandb_logger,
                log_every_n_steps=1,
                max_epochs=10000000,
                default_root_dir=args.save_dir,
            )

            trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path)
        except Exception as e:
            gr.Error(f"Error training model, please check console!")
            traceback.print_exc()
            return

    def create_ui(self, theme):
        theme = gr.Theme.from_hub(theme).set() if theme != "default" else None
        with gr.Blocks(
            theme=theme, css="style.css", title="Audio Diffusion Gradio"
        ) as ui:
            self.tab_control = gr.Tabs()
            with self.tab_control:
                with gr.Tab("Model Process", id="model_process"):
                    self.aec_ui = self.create_autoencoder_ui()
                    self.txt2audio_ui = self.create_txt2audio_ui()
                    self.blank_ui = gr.Box(visible=True)
                    with self.blank_ui:
                        gr.Markdown(
                            """# ⚠️ No processable model loaded! Please load an unwrapped model to start processing. (Did you forget to unwrap?)"""
                        )
                if not "settings" in self.hidden:
                    with gr.Tab(
                        "Loaded Model/Global Settings",
                        id="model_settings",
                    ):
                        header = self.create_selector_header()
                else:
                    header = self.create_selector_header(visible=False)

                if "extensions" not in self.hidden:
                    self.load_extensions()

            self.interface = ui
            ui.load(
                self.on_load,
                outputs=self.make_loadable_vars(),
                api_name=False,
                queue=False,
            )
        return ui

    def on_load(self):
        if self.current_loaded_model:
            return_data = self.set_global_values(
                self.current_set_model_path,
                self.current_set_model_config,
                self.current_set_pretransform_path,
            )  # infobox, cond switch, visibilities of the UIs, close settings, sec sliders
            return_data = return_data + [
                self.current_device,
                self.fp16,
                self.current_set_model_path
                if self.current_loaded_model_path
                else "None",
                self.current_set_model_config if not self.embedded_config else None,
                self.current_set_pretransform_path,
            ]  # add global settings
        else:
            return_data = [
                gr.HTML(self.create_info_table()),  # default info table
                gr.Slider(visible=False),  # default cond switch cfg scale
                gr.Slider(visible=False),  # default cond switch cfg rescale
                gr.Column(visible=False),  # default cond switch box
                gr.Number(0.3),  # default cond switch box
                gr.Box(visible=False),  # default txt2audio
                gr.Box(visible=False),  # default autoencoder
                gr.Box(visible=True),  # default blank
                gr.Tabs(
                    selected="model_settings"
                    if "settings" not in self.hidden
                    else "model_process"
                ),  # open settings
                gr.Accordion(
                    label="⌛ Timing Controls", visible=False
                ),  # default timing accordion
                gr.Slider(visible=False),  # default seconds start
                gr.Slider(visible=False),  # default seconds total
                gr.Textbox(visible=False),  # default prompt
                gr.Slider(value=65536),  # default sample size
                self.current_device,
                self.fp16,
                self.current_set_model_path,
                self.current_set_model_config if not self.embedded_config else None,
                self.current_set_pretransform_path,
            ]

        return return_data

    def create_info_table(
        self,
        model_filename="None",
        pretransform_filename="None",
        model_config={
            "model_type": "None",
            "sample_rate": "None",
            "sample_size": "None",
            "audio_channels": "None",
        },
        msg="🔴 Please set your files.",
    ):
        if model_config["model_type"] == "None":
            wraptext = ""
        else:
            wraptext = " (Wrapped)" if self.current_wrapped else " (Unwrapped)"
        return f"""
        <div style="font-family: Arial, sans-serif;">
            <h1>{msg}</h1>
            <table style="
                border-collapse: collapse;
                width: 100%;
                border: none;
                text-align: left;
                ">
                <tr>
                    <td style="padding: 8px;">Filename</td>
                    <td style="padding: 8px;">{model_filename}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Custom Pretransform</td>
                    <td style="padding: 8px;">{pretransform_filename if pretransform_filename else " "}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Model Type</td>
                    <td style="padding: 8px;">{model_config["model_type"]}{wraptext}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Native Sample Rate</td>
                    <td style="padding: 8px;">{model_config["sample_rate"]}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Native Chunk Size</td>
                    <td style="padding: 8px;">{model_config["sample_size"]}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Native N Channels</td>
                    <td style="padding: 8px;">{model_config["audio_channels"]}</td>
                </tr>
            </table>
        </div>
        """


def main(args):
    # make authfile if not exists
    if not os.path.isfile(args.authfile):
        from getpass import getpass

        print("No authfile found, creating one and registering initial user.")

        # prompt user for initial username and password
        input_username = input("Enter registration username: ")
        input_password = getpass("Enter registration password: ")
        with open(args.authfile, "w") as f:
            f.write(f"{input_username}:{input_password}")
    torch.manual_seed(42)
    if args.hidden:
        print(f"Hidden functions: {args.hidden}")

    def auth(username, password):
        with open(args.authfile) as f:
            authfile = f.readlines()
        for line in authfile:
            line_username = line.split(":")[0]
            line_password = line.split(":")[1].strip()
            if username == line_username and password == line_password:
                return True

    # remove the dirs from model_dir that dont exist or are duplicates
    args.model_dir = list(set(args.model_dir))
    args.model_dir = [x for x in args.model_dir if os.path.isdir(x)]

    interface = stable_audio_interface(
        init_model_dirs=args.model_dir,
        hidden=args.hidden,
        init_model_ckpt=args.init_model_ckpt,
        init_model_config=args.init_model_config,
        init_pretransform_ckpt=args.init_pretransform_ckpt,
        init_device=args.init_device,
        extensions_folder=args.extensions_folder,
    ).create_ui(theme=args.theme)
    interface.queue(api_open=False)
    interface.launch(
        share=True if not args.no_share else False,
        auth=auth if not args.no_auth else None,
        server_name=args.listen,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gradio interface")
    parser.add_argument(
        "--listen",
        type=str,
        help="Listen on a specific host",
        required=False,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port", type=int, help="Port to listen on", required=False, default=7860
    )
    # add list for model_dirs
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to model directory",
        required=False,
        default=["./models"],
        action="append",
    )
    parser.add_argument(
        "--hidden",
        type=str,
        help="Disable function. Options: 'train', 'settings', 'extensions'",
        required=False,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--init-model-ckpt",
        type=str,
        help="Path to initial model checkpoint",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--init-model-config",
        type=str,
        help="Path to initial model config",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--init-pretransform-ckpt",
        type=str,
        help="Path to initial pretransform checkpoint",
        required=False,
        default=None,
    )
    # init device
    parser.add_argument(
        "--init-device",
        type=str,
        help="Initial device to use",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--authfile",
        type=str,
        help="Path to authfile",
        required=False,
        default="logins.txt",
    )

    parser.add_argument(
        "--theme",
        type=str,
        help="Theme to use",
        required=False,
        default="freddyaboulton/dracula_revamped",
    )
    parser.add_argument(
        "--extensions-folder",
        type=str,
        help="Path to extensions folder",
        required=False,
        default="./extensions",
    )
    # no auth
    parser.add_argument(
        "--no-auth",
        help="Disable authentication",
        required=False,
        action="store_true",
    )
    # no share
    parser.add_argument(
        "--no-share",
        help="Disable sharing",
        required=False,
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
