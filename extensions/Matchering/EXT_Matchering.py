import gradio as gr
from aeiou.viz import audio_spectrogram_image
import traceback

# initialize interface
interface = None

# required modules and import aliases for modules (to prevent infinite reinstall)
install_modules = ["matchering", "soundfile"]
install_aliases = {}


def master_audio(audio, reference_audio, progress=gr.Progress(track_tqdm=True)):
    from matchering import Config
    from matchering.stages import main

    try:
        if audio is None:
            gr.Error("Please upload an audio clip to master.")
            return [None] * 2
        elif reference_audio is None:
            gr.Error("Please upload a reference audio clip.")
            return [None] * 2
        sr, wav = audio
        sr_ref, wav_ref = reference_audio

        config = Config()

        config.internal_sample_rate = sr

        # to float64
        wav = wav.astype("float64")
        wav_ref = wav_ref.astype("float64")

        # Process
        _, result_no_limiter, _ = main(
            wav,
            wav_ref,
            config,
            need_default=False,
            need_no_limiter=True,
            need_no_limiter_normalized=False,
        )

        del wav_ref
        del wav

        return (sr, result_no_limiter)
    except Exception as e:
        print(e)
        traceback.print_exc()
        gr.Error(f"Error processing audio: {e}")
        return [None] * 3


def make_ui():
    if interface:
        with gr.Tab(label="Matchering Mastering") as UI:
            with gr.Row(equal_height=False):
                with gr.Column():
                    input_audio = interface.create_audio_input()
                    with gr.Group():
                        reference_audio = gr.Audio(
                            label="Reference audio",
                            elem_id="small_audio",
                        )
                        process_button = gr.Button(value="Process", variant="primary")
                with gr.Column(variant="panel"):
                    output = interface.create_audio_output(label="🔊 Mastered Output")

                process_button.click(
                    fn=master_audio,
                    inputs=[input_audio, reference_audio],
                    outputs=output,
                    api_name="matchering_mastering",
                )
