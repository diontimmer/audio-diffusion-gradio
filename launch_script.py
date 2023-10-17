# this scripts installs necessary requirements and launches main program
import subprocess
import os
import importlib.util
import sys
import urllib.request
from shutil import which
import pkg_resources

cmd_args = [
    # "--listen=192.0.0.1"
    # "--no-auth",
    # "--no-share",
    # "--init-model-ckpt=./dnb_112300.ckpt"
]

# ****************************************************************************
# *                                   UTIL                                   *
# ****************************************************************************

python = sys.executable
print(f"Launching using Python: {python}")
platform = sys.platform


def prRed(skk):
    print(f"\033[91m{skk}\033[00m")


def prGreen(skk):
    print(f"\033[92m{skk}\033[00m")


def prYellow(skk):
    print(f"\033[93m{skk}\033[00m")


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


def is_installed(package):
    if package in aliases:
        package = aliases[package]
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_pip(args, desc=None, live=True):
    index_url = os.environ.get("INDEX_URL", "")
    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    return run(
        f'"{sys.executable}" -m pip {args} --prefer-binary{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
        live=live,
    )


def upgrade_pip():
    return run_pip("install --upgrade pip", desc="Upgrading pip")


def get_package_version(package):
    if package in aliases:
        package = aliases[package]
    try:
        version = pkg_resources.get_distribution(package).version
        return version
    except pkg_resources.DistributionNotFound:
        return None


# ****************************************************************************
# *                                  CONFIG                                  *
# ****************************************************************************

SKIP_INSTALL = False

torch_command = (
    "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121"
    if platform == "win32"
    else "pip install torch torchaudio"
)

main_script_path = "gradio_extended.py"
pre_torch_packages = []
post_torch_packages = [
    "xformers",
    "stable_audio_tools",
    "gradio==3.48.0",
]

aliases = {}


prGreen("Starting launch script...")


if __name__ == "__main__":
    print("Checking pip upgrade..")
    upgrade_pip()
    # INSTALL

    if not SKIP_INSTALL:
        # pre torch packages
        if pre_torch_packages:
            for package in pre_torch_packages:
                package_name = package.split("==")[0]
                if not is_installed(package_name):
                    run_pip(f"install {package}", package)
                else:  # check version
                    version = get_package_version(package_name)
                    print(f"Version of {package_name}: {version}")
                    if (
                        version is not None
                        and "==" in package
                        and version != package.split("==")[1]
                    ):
                        run_pip(f"install {package}", package_name)

        # TORCH INSTALL
        if not is_installed("torch") and torch_command is not None:
            run(
                f'"{sys.executable}" -m {torch_command}',
                "Installing torch.",
                "Couldn't install torch",
                live=True,
            )

        # post torch packages
        if post_torch_packages:
            for package in post_torch_packages:
                package_name = package.split("==")[0]
                if not is_installed(package_name):
                    run_pip(f"install {package}", package)
                else:  # check version
                    version = get_package_version(package_name)
                    print(f"Version of {package_name}: {version}")
                    if (
                        version is not None
                        and "==" in package
                        and version != package.split("==")[1]
                    ):
                        run_pip(f"install {package}", package_name)

        if which("ffmpeg") is None:
            if platform == "win32":
                prYellow("No FFMPEG detected! Opening installer..")
                urllib.request.urlretrieve(
                    "https://github.com/icedterminal/ffmpeg-installer/releases/download/5.1.0.20220727/FFmpeg_Essentials.msi",
                    "ffmpeg_installer.msi",
                )
                subprocess.run("ffmpeg_installer.msi", shell=True)
                os.remove("ffmpeg_installer.msi")

            elif platform == "linux":
                prRed(
                    'No FFMPEG detected! Please update pkgs and run "sudo apt install ffmpeg" to install before using the GUI!'
                )
                exit()

            elif platform == "darwin":
                prRed(
                    'No FFMPEG detected! Please use homebrew "brew install ffmpeg" to install before using the GUI!'
                )
                exit()

    # LAUNCH
    run(
        f'"{sys.executable}" {main_script_path} {" ".join(cmd_args)}',
        "Launch success! Starting main script, this might take a bit..",
        "MAIN SCRIPT CRASH",
        live=True,
    )
