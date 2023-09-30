import gradio as gr
import torch
from aeiou.viz import audio_spectrogram_image

DEMUCS_MODELS = ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"]

# initialize interface
interface = None

# required modules and import aliases for modules (to prevent infinite reinstall)
install_modules = ["demucs"]
install_aliases = {}


def separate_audio(audio, model, progress=gr.Progress(track_tqdm=True)):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if audio is None:
        gr.Error("Please upload an audio clip to separate.")
        return [None] * 4
    sr, wav = audio
    model = get_model(model)

    wav = torch.from_numpy(wav.T).float()

    model.cpu()
    model.eval()
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    sources = apply_model(
        model,
        wav[None],
        device=interface.current_device,
        shifts=1,
        split=True,
        overlap=0.25,
        progress=True,
        num_workers=8,
    )[0]
    sources *= ref.std()
    sources += ref.mean()
    audios = []
    images = []
    for source in list(sources):
        images.append(
            audio_spectrogram_image(
                source,
                sample_rate=sr,
            )
        )
        audios.append((sr, source.numpy().T))
    # audios is drums, bass, other, vocals
    # reorder to vocals, drums, other, bass
    audios = [audios[3], audios[0], audios[2], audios[1]]
    return audios + [images]


def make_ui():
    if interface:
        with gr.Tab(label="Demucs Separation") as UI:
            with gr.Row(equal_height=False):
                with gr.Column():
                    input_audio = interface.create_audio_input()
                    with gr.Group():
                        model = gr.Dropdown(
                            choices=DEMUCS_MODELS,
                            label="Model",
                            value=DEMUCS_MODELS[0],
                        )
                        process_button = gr.Button(value="Process", variant="primary")
                with gr.Column(variant="panel"):
                    output_vocals = interface.create_audio_output(label="🎤 Vocals")
                    output_drums = interface.create_audio_output(label="🥁 Drums")
                    output_other = interface.create_audio_output(label="🎹 Other")
                    output_bass = interface.create_audio_output(label="🎸 Bass")
                    outputs = [
                        output_vocals,
                        output_drums,
                        output_other,
                        output_bass,
                    ]
                    with gr.Accordion("🌈 Spectrograms", open=False):
                        spec = gr.Gallery(label="Output spectrogram", show_label=False)
                    interface.create_send_to_input_list(outputs)
                process_button.click(
                    fn=separate_audio,
                    inputs=[input_audio, model],
                    outputs=outputs + [spec],
                    api_name="demucs_separate",
                )
