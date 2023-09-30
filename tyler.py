import torch
from typing import Callable
import gc
from tqdm import tqdm

"""
TYLER: Utility for processing audio in a windowed fashion using tiled processing.

"""


def tyler_patch(module, function_name: str, window_length: int = 65536):
    """
    Patches a function in a module to apply it in a windowed fashion.

    Args:
        module (module): The module to patch.
        function_name (str): The name of the function to patch.
        window_length (int, optional): The length of the window to process the audio tensor in. Defaults to 65536.

    Returns:
        Callable: The original function.
    """
    original_function = getattr(module, function_name)

    def patched_function(x):
        return TYLER(x, original_function, window_length=window_length)

    setattr(module, function_name, patched_function)
    return original_function


def TYLER(
    audio: torch.Tensor,
    fn: Callable,
    window_length: int = 65536,
    show_tqdm: bool = False,
):
    """
    Applies a function to an audio tensor in a windowed fashion.

    Args:
        audio (torch.Tensor): The audio tensor to process. [(Batch), Channels, Time]
        fn (Callable): The function to apply to the audio tensor.
        window_length (int, optional): The length of the window to process the audio tensor in. Defaults to 65536.
        use_tqdm (bool, optional): Whether to use tqdm to display progress. Defaults to False.

    Returns:
        torch.Tensor: The processed audio tensor. [Batch, Channels, Time]
    """
    print(
        f"TYLER | Processing audio of shape {audio.shape} with function '{fn.__name__}' using a window length of {window_length}"
    )

    device = audio.device
    if audio.dim() != 3:
        audio = audio.unsqueeze(0)

    output_tensors = []
    audio = (
        tqdm(audio, desc="Split Batch Process", unit="Batch N") if show_tqdm else audio
    )

    for single_audio in audio:
        total_len = single_audio.shape[1]
        processed_audio = None
        start_idx = 0

        inner_tqdm = None
        if show_tqdm:
            inner_tqdm = tqdm(
                total=total_len,
                desc=f"Tiled Process: {fn.__name__}",
                unit="Samples",
                leave=False,
            )

        while start_idx < total_len:
            end_idx = min(start_idx + window_length, total_len)
            pad_len = window_length - (end_idx - start_idx)
            chunk = torch.nn.functional.pad(
                single_audio[:, start_idx:end_idx], (0, pad_len)
            )
            decoded_chunk = fn(chunk.unsqueeze(0)).squeeze(0)
            if processed_audio is None:
                processed_audio = torch.zeros((decoded_chunk.shape[0], 0)).to(device)

            processed_audio = torch.cat((processed_audio, decoded_chunk), dim=1)
            start_idx += window_length

            if inner_tqdm:
                inner_tqdm.update(window_length)

            del chunk, decoded_chunk
            torch.cuda.empty_cache()
            gc.collect()

        if inner_tqdm:
            inner_tqdm.close()

        output_tensors.append(processed_audio.unsqueeze(0))
        processed_audio.detach()
        processed_audio.to("cpu")
        del processed_audio
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(output_tensors, dim=0).to(device)
