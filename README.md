# audio-diffusion-gradio (alpha)

 The Audio Diffusion Gradio Interface is a user-friendly graphical user interface (GUI) that simplifies the process of working with audio diffusion models, autoencoders, diffusion autoencoders, and  various models trainable using the stable-audio-tools package. This interface not only streamlines your audio diffusion tasks but also provides a modular extension system, enabling users to easily integrate additional functionalities. This software is a WIP and alot of functionality could have bugs.

## Features
- Load and unwrap diffusion models, autoencoders, diffusion autoencoders & more.
- Seamless integration with the stable-audio-tools package.
- Easily extend functionality through a modular system.
- Comes with two active extensions:
    Demucs: A powerful source separation model for audio.
    Matchering: An audio mastering tool for automated audio processing.

## Command line args:
- `--listen` (Optional, Default: "127.0.0.1"): Listen on a specific host.
- `--port` (Optional, Default: 7860): Port to listen on.
- `--model-dir` (Optional, Default: ["./models"]): Path to model directory. (You can use multiple `--model-dir` flags for multiple directories)
- `--hidden` (Optional, Default: []): Disable functions. Options: 'train', 'settings', 'extensions'. (You can use multiple `--hidden` flags to disable multiple functions)
- `--init-model-ckpt` (Optional, Default: None): Path to the initial model checkpoint.
- `--init-model-config` (Optional, Default: None): Path to the initial model configuration file.
- `--init-pretransform-ckpt` (Optional, Default: None): Path to the initial pretransform checkpoint.
- `--init-device` (Optional, Default: None): Initial device to use.
- `--authfile` (Optional, Default: "logins.txt"): Path to the authentication file.
- `--theme` (Optional, Default: "freddyaboulton/dracula_revamped"): Theme to use.
- `--extensions-folder` (Optional, Default: "./extensions"): Path to the extensions folder.
- `--no-auth` (Optional, Default: False): Disable authentication (use this flag to disable).
- `--no-share` (Optional, Default: False): Disable sharing (use this flag to disable).