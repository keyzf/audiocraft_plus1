# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen, MultiBandDiffusion

version = "2.0.0"

theme = gr.themes.Base(
    primary_hue="lime",
    secondary_hue="lime",
    neutral_hue="neutral",
).set(
    button_primary_background_fill_hover='*primary_500',
    button_primary_background_fill_hover_dark='*primary_500',
    button_secondary_background_fill_hover='*primary_500',
    button_secondary_background_fill_hover_dark='*primary_500'
)

MODEL = None  # Last used model
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
print(IS_BATCHED)
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


def load_diffusion():
    global MBD
    print("loading MBD")
    MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, melodies, duration, progress=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    if any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
            return_tokens=USE_DIFFUSION
        )
    else:
        outputs = MODEL.generate(texts, progress=progress, return_tokens=USE_DIFFUSION)
    if USE_DIFFUSION:
        outputs_diffusion = MBD.tokens_to_wav(outputs[1])
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
            print(f'wav: {file.name}')
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
        print(f'video: {video}')
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return res


def predict_full(model, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    def _progress(generated, to_generate):
        progress((min(generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, _ = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    if USE_DIFFUSION:
        return videos[0], None, videos[1], None
    return videos[0], None, None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def ui_full(launch_kwargs):
    with gr.Blocks(title='Audiocraft Plus', theme=theme) as interface:
        gr.Markdown(
            """
            # Audiocraft Plus V2.0.0

            ## An All-in-One Audiocraft WebUI

            Thanks to: facebookresearch, Camenduru, rkfg, oobabooga, AlexHK and GrandaddyShmax
            """
        )
        with gr.Tab("MusicGen"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Generation"):
                        with gr.Accordion("Structure Prompts", open=False):
                            with gr.Column():
                                with gr.Row():
                                    struc_prompts = gr.Checkbox(label="Enable", value=False, interactive=True, container=False)
                                    bpm = gr.Number(label="BPM", value=120, interactive=True, scale=1, precision=0)
                                    key = gr.Dropdown(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"], label="Key", value="C", interactive=True)
                                    scale = gr.Dropdown(["Major", "Minor"], label="Scale", value="Major", interactive=True)
                                with gr.Row():
                                    global_prompt = gr.Text(label="Global Prompt", interactive=True, scale=3)
                        with gr.Column():
                            text = gr.Text(label="Input Text", interactive=True)
                        with gr.Row():
                            duration = gr.Slider(minimum=1, maximum=300, value=10, step=1, label="Duration", interactive=True)

                    with gr.Tab("Audio"):
                        with gr.Row():
                            with gr.Column():
                                input_type = gr.Radio(["file", "mic"], value="file", label="Input Type (optional)", interactive=True)
                                mode = gr.Radio(["melody", "sample"], label="Input Audio Mode (optional)", value="sample", interactive=True)
                            melody = gr.Audio(source="upload", type="numpy", label="Input Audio (optional)", interactive=True)

                    with gr.Tab("Settings"):
                        with gr.Row():
                            model = gr.Radio(["melody", "small", "medium", "large", "custom"], label="Model", value="large", interactive=True, scale=1)
                        with gr.Row():
                            decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                               label="Decoder", value="Default", interactive=True)
                        with gr.Row():
                            topk = gr.Number(label="Top-k", value=250, interactive=True)
                            topp = gr.Number(label="Top-p", value=0, interactive=True)
                            temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                            cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                    with gr.Row():
                        submit = gr.Button("Generate", variant="primary")
                        # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                        _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Column() as c:
                    with gr.Tab("Output"):
                        output = gr.Video(label="Generated Music")
                        #audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
                        diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")
                        audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')
                    with gr.Tab("Wiki"):
                        gr.Markdown(
                            """
                            - **[Generate (button)]:**  
                            Generates the music with the given settings and prompts.

                            - **[Interrupt (button)]:**  
                            Stops the music generation as soon as it can, providing an incomplete output.

                            ---

                            ### Generation Tab:

                            #### Structure Prompts:

                            This feature helps reduce repetetive prompts by allowing you to set global prompts  
                            that will be used for all prompt segments.

                            - **[Structure Prompts (checkbox)]:**  
                            Enable/Disable the structure prompts feature.

                            - **[BPM (number)]:**  
                            Beats per minute of the generated music.

                            - **[Key (dropdown)]:**  
                            The key of the generated music.

                            - **[Scale (dropdown)]:**  
                            The scale of the generated music.

                            - **[Global Prompt (text)]:**  
                            Here write the prompt that you wish to be used for all prompt segments.

                            #### Multi-Prompt: 
                            
                            This feature allows you to control the music, adding variation to different time segments.  
                            You have up to 10 prompt segments. the first prompt will always be 30s long  
                            the other prompts will be [30s - overlap].  
                            for example if the overlap is 10s, each prompt segment will be 20s.

                            - **[Prompt Segments (number)]:**  
                            Amount of unique prompt to generate throughout the music generation.

                            - **[Prompt/Input Text (prompt)]:**  
                            Here describe the music you wish the model to generate.

                            - **[Repeat (number)]:**  
                            Write how many times this prompt will repeat (instead of wasting another prompt segment on the same prompt).

                            - **[Time (text)]:**  
                            The time of the prompt segment.

                            - **[Calculate Timings (button)]:**  
                            Calculates the timings of the prompt segments.

                            - **[Duration (number)]:**  
                            How long you want the generated music to be (in seconds).

                            - **[Overlap (number)]:**  
                            How much each new segment will reference the previous segment (in seconds).  
                            For example, if you choose 20s: Each new segment after the first one will reference the previous segment 20s  
                            and will generate only 10s of new music. The model can only process 30s of music.

                            - **[Seed (number)]:**  
                            Your generated music id. If you wish to generate the exact same music,  
                            place the exact seed with the exact prompts  
                            (This way you can also extend specific song that was generated short).

                            - **[Random Seed (button)]:**  
                            Gives "-1" as a seed, which counts as a random seed.

                            - **[Copy Previous Seed (button)]:**  
                            Copies the seed from the output seed (if you don't feel like doing it manualy).

                            ---

                            ### Audio Tab:

                            - **[Input Type (selection)]:**  
                            `File` mode allows you to upload an audio file to use as input  
                            `Mic` mode allows you to use your microphone as input

                            - **[Input Audio Mode (selection)]:**  
                            `Melody` mode only works with the melody model: it conditions the music generation to reference the melody  
                            `Sample` mode works with any model: it gives a music sample to the model to generate its continuation.

                            - **[Trim Start and Trim End (numbers)]:**  
                            `Trim Start` set how much you'd like to trim the input audio from the start  
                            `Trim End` same as the above but from the end

                            - **[Input Audio (audio file)]:**  
                            Input here the audio you wish to use with "melody" or "sample" mode.

                            ---

                            ### Customization Tab:

                            - **[Background Color (color)]:**  
                            Works only if you don't upload image. Color of the background of the waveform.

                            - **[Bar Color Start (color)]:**  
                            First color of the waveform bars.

                            - **[Bar Color End (color)]:**  
                            Second color of the waveform bars.

                            - **[Background Image (image)]:**  
                            Background image that you wish to be attached to the generated video along with the waveform.

                            - **[Height and Width (numbers)]:**  
                            Output video resolution, only works with image.  
                            (minimum height and width is 256).
                            
                            ---

                            ### Settings Tab:

                            - **[Output Audio Channels (selection)]:**  
                            With this you can select the amount of channels that you wish for your output audio.  
                            `mono` is a straightforward single channel audio  
                            `stereo` is a dual channel audio but it will sound more or less like mono  
                            `stereo effect` this one is also dual channel but uses tricks to simulate a stereo audio.

                            - **[Output Audio Sample Rate (dropdown)]:**  
                            The output audio sample rate, the model default is 32000.

                            - **[Model (selection)]:**  
                            Here you can choose which model you wish to use:  
                            `melody` model is based on the medium model with a unique feature that lets you use melody conditioning  
                            `small` model is trained on 300M parameters  
                            `medium` model is trained on 1.5B parameters  
                            `large` model is trained on 3.3B parameters  
                            `custom` model runs the custom model that you provided.

                            - **[Custom Model (selection)]:**  
                            This dropdown will show you models that are placed in the `models` folder  
                            you must select `custom` in the model options in order to use it.

                            - **[Refresh (button)]:**  
                            Refreshes the dropdown list for custom model.

                            - **[Base Model (selection)]:**  
                            Choose here the model that your custom model is based on.

                            - **[Top-k (number)]:**  
                            is a parameter used in text generation models, including music generation models. It determines the number of most likely next tokens to consider at each step of the generation process. The model ranks all possible tokens based on their predicted probabilities, and then selects the top-k tokens from the ranked list. The model then samples from this reduced set of tokens to determine the next token in the generated sequence. A smaller value of k results in a more focused and deterministic output, while a larger value of k allows for more diversity in the generated music.

                            - **[Top-p (number)]:**  
                            also known as nucleus sampling or probabilistic sampling, is another method used for token selection during text generation. Instead of specifying a fixed number like top-k, top-p considers the cumulative probability distribution of the ranked tokens. It selects the smallest possible set of tokens whose cumulative probability exceeds a certain threshold (usually denoted as p). The model then samples from this set to choose the next token. This approach ensures that the generated output maintains a balance between diversity and coherence, as it allows for a varying number of tokens to be considered based on their probabilities.
                            
                            - **[Temperature (number)]:**  
                            is a parameter that controls the randomness of the generated output. It is applied during the sampling process, where a higher temperature value results in more random and diverse outputs, while a lower temperature value leads to more deterministic and focused outputs. In the context of music generation, a higher temperature can introduce more variability and creativity into the generated music, but it may also lead to less coherent or structured compositions. On the other hand, a lower temperature can produce more repetitive and predictable music.

                            - **[Classifier Free Guidance (number)]:**  
                            refers to a technique used in some music generation models where a separate classifier network is trained to provide guidance or control over the generated music. This classifier is trained on labeled data to recognize specific musical characteristics or styles. During the generation process, the output of the generator model is evaluated by the classifier, and the generator is encouraged to produce music that aligns with the desired characteristics or style. This approach allows for more fine-grained control over the generated music, enabling users to specify certain attributes they want the model to capture.
                            """
                        )
                    with gr.Tab("Changelog"):
                        gr.Markdown(
                            """
                            ## Changelog:

                            ### V1.2.8c

                            - Implemented Reverse compatibility for audio info tab with previous versions



                            ### V1.2.8b

                            - Fixed the error when loading default models



                            ### V1.2.8a

                            - Adapted Audio info tab to work with the new structure prompts feature

                            - Now custom models actually work, make sure you select the correct base model



                            ### V1.2.8

                            - Now you will also recieve json file with metadata of generated audio

                            - Added error messages in Audio Info tab

                            - Added structure prompts: you can select bpm, key and global prompt for all prompts

                            - Added time display next to each prompt, can be calculated with "Calculate Timings" button



                            ### V1.2.7

                            - When sending generated audio to Input Audio, it will send a backup audio with default settings  
                            (best for continuos generation)

                            - Added Metadata to generated audio (Thanks to AlexHK ♥)

                            - Added Audio Info tab that will display the metadata of the input audio

                            - Added "send to Text2Audio" button in Audio Info tab

                            - Generated audio is now stored in the "output" folder (Thanks to AlexHK ♥)

                            - Added an output area with generated files and download buttons

                            - Enhanced Stereo effect (Thanks to AlexHK ♥)



                            ### V1.2.6

                            - Added option to generate in stereo (instead of only mono)

                            - Added dropdown for selecting output sample rate (model default is 32000)



                            ### V1.2.5a

                            - Added file cleaner (This comes from the main facebookresearch repo)

                            - Reorganized a little, moved audio to a seperate tab



                            ### V1.2.5

                            - Gave a unique lime theme to the webui
                            
                            - Added additional output for audio only

                            - Added button to send generated audio to Input Audio

                            - Added option to trim Input Audio



                            ### V1.2.4

                            - Added mic input (This comes from the main facebookresearch repo)



                            ### V1.2.3

                            - Added option to change video size to fit the image you upload



                            ### V1.2.2

                            - Added Wiki, Changelog and About tabs



                            ### V1.2.1

                            - Added tabs and organized the entire interface

                            - Added option to attach image to the output video

                            - Added option to load fine-tuned models (Yet to be tested)



                            ### V1.2.0

                            - Added Multi-Prompt



                            ### V1.1.3

                            - Added customization options for generated waveform



                            ### V1.1.2

                            - Removed sample length limit: now you can input audio of any length as music sample



                            ### V1.1.1

                            - Improved music sample audio quality when using music continuation



                            ### V1.1.0

                            - Rebuilt the repo on top of the latest structure of the main MusicGen repo
                            
                            - Improved Music continuation feature



                            ### V1.0.0 - Stable Version

                            - Added Music continuation
                            """
                        )
                    with gr.Tab("About"):
                        gr.Markdown(
                            """
                            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
                            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
                            
                            ## MusicGen+ is an extended version of the original MusicGen by facebookresearch. 
                            
                            ### Repo: https://github.com/GrandaddyShmax/audiocraft_plus/tree/plus

                            ---
                            
                            ### This project was possible thanks to:

                            #### GrandaddyShmax - https://github.com/GrandaddyShmax

                            #### Camenduru - https://github.com/camenduru

                            #### rkfg - https://github.com/rkfg

                            #### oobabooga - https://github.com/oobabooga
                            
                            #### AlexHK - https://github.com/alanhk147
                            """
                        )
        
        submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                     show_progress=False).then(predict_full, inputs=[model, decoder, text, melody, duration, topk, topp,
                                                                     temperature, cfg_coef],
                                               outputs=[output, diffusion_output, audio_diffusion])
        input_type.change(toggle_audio_src, input_type, [melody], queue=False, show_progress=False)
        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(source="upload", type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
        submit.click(predict_batched, inputs=[text, melody],
                     outputs=[output, audio_output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output]
        )
        gr.Markdown("""
        ### More details

        The model will generate 12 seconds of audio based on the description you provided.
        You can optionally provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        All samples are generated with the `melody` model.

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    # Show the interface
    if IS_BATCHED:
        global USE_DIFFUSION
        USE_DIFFUSION = False
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)