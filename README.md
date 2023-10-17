# audio-diffusion-gradio (alpha)

![adg](https://www.dropbox.com/scl/fi/6l0j8nsbitbagbzcetd6v/audio-diffusion-gradio.png?rlkey=6zzx3mjxq4gemj1dgbjkvjsdw&raw=1)
 The Audio Diffusion Gradio Interface is a user-friendly graphical user interface (GUI) made in Gradio that simplifies the process of working with audio diffusion models, autoencoders, diffusion autoencoders, and  various models trainable using the stable-audio-tools package. This interface not only streamlines your audio diffusion tasks but also provides a modular extension system, enabling users to easily integrate additional functionalities. This software is a WIP and alot of functionality could have bugs.
 The GUI will ask for an initial login+password combination for access to the gui. This will be stored in plain-text format in ```logins.txt```. Additional users can be added in the format username:password.

 To run, please launch start_windows or start_unix.sh depending on your platform. This will create a virtual environment for you. If you prefer to run your own python, you are free to install the ```requirements.txt``` file and run ```gradio_extended.py``` using your own interpreter.
 Place your loadable models in the ```./models``` folder to access them through the GUI. All models need their configs to load properly. You can select the config using the menu, or place the .json alongside the model with the exact same name. You can also embed the model config dictionary directly into the file as a top-level key as ```model_config``` and it will be able to read that as well. The internal trainer will embed the model config automatically.

 The examples and tags components can be activated and filled by creating an 'examples.txt' or 'tags.txt' file and placing an element you would want to view on every newline.

## Features
- Load and unwrap diffusion models, autoencoders, diffusion autoencoders & more.
- Seamless integration with the stable-audio-tools package.
- Easily extend functionality through a modular system.
- Comes with two active extensions:
    Demucs: A powerful source separation model for audio.
    Matchering: An audio mastering tool for automated audio processing.

## Command line options:
 The GUI can be launched using various options. These options can be directly set if launching gradio_extended.py without a venv. If using a venv, they can be set at the top of the ```launch_script.py``` file.
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