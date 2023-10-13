# harmonai_gradio (alpha)
 Decked-out gradio client for stable-audio-tools.
 This software is a work in progress. 
 More features and documentation to come.

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