# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/setup.py
import questionary
import subprocess
import logging

# config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# function create virtual env use venv
# @param env_path: the path of virtual env
def create_env(env_path):
    try:
        result = subprocess.run(["python3", "-m", "venv", env_path])
        if result.returncode == 0:
            logging.info("create virtual env success")
        else:
            logging.error("create virtual env failed")
    except Exception as e:
        logging.error("create virtual env failed")
        logging.error(e)
        exit(1)

# install pytorch in virtual env
# @param env_path: the path of virtual env
# @param device: the device that user want to use
# @param pytorch_version: the version of pytorch
# @param cuda_version: the version of cuda
def install_pytorch(env_path, device, pytorch_version, cuda_version):
    # gpu + pytorch 2.0.1 + cuda 11.8: 
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # gpu + pytorch 2.0.1 + cuda 11.7:
    # pip install torch torchvision torchaudio
    # gpu + pytorch 1.13.1 + cuda 11.7:
    # pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    # gpu + pytorch 1.13.1 + cuda 11.6:
    # pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    # cpu + pytorch 2.0.1:
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # cpu + pytorch 1.13.1:
    # pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
    
    # use env_path/bin/pip to install pytorch
    try:
        pip_exec = env_path + "/bin/pip"
        if device == "[1] cpu":
            if pytorch_version == "[1] 2.0.1":
                result = subprocess.run([pip_exec, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
            elif pytorch_version == "[2] 1.13.1":
                result = subprocess.run([pip_exec, "install", "torch==1.13.1+cpu", "torchvision==0.14.1+cpu", "torchaudio==0.13.1", "--extra-index-url", "https://download.pytorch.org/whl/cpu"])
        elif device == "[2] gpu":
            if pytorch_version == "[1] 2.0.1":
                if cuda_version == "[1] 11.8":
                    result = subprocess.run([pip_exec, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
                elif cuda_version == "[2] 11.7":
                    result = subprocess.run([pip_exec, "install", "torch", "torchvision", "torchaudio"])
            elif pytorch_version == "[2] 1.13.1":
                if cuda_version == "[1] 11.7":
                    result = subprocess.run([pip_exec, "install", "torch==1.13.1+cu117", "torchvision==0.14.1+cu117", "torchaudio==0.13.1", "--extra-index-url", "https://download.pytorch.org/whl/cu117"])
                elif cuda_version == "[2] 11.6":
                    result = subprocess.run([pip_exec, "install", "torch==1.13.1+cu116", "torchvision==0.14.1+cu116", "torchaudio==0.13.1", "--extra-index-url", "https://download.pytorch.org/whl/cu116"])
        if result.returncode == 0:
            logging.info("install pytorch success")
        else:
            logging.error("install pytorch failed")
    except Exception as e:
        logging.error("install pytorch failed")
        logging.error(e)
        exit(1)

# install jax[tpu] if user want to use tpu
# @param env_path: the path of virtual env
def install_jax(env_path):
    try:
        # pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        pip_exec = env_path + "/bin/pip"
        result = subprocess.run([pip_exec, "install", "jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"])
        if result.returncode == 0:
            logging.info("install jax success")
        else:
            logging.error("install jax failed")
    except Exception as e:
        logging.error("install jax failed")
        logging.error(e)
        exit(1)

# if gpu, test cuda is available
# if tpu, test tpu is available
# @param env_path: the path of virtual env
# @param device: the device that user want to use
def test_device(env_path, device):
    python_exec = env_path + "/bin/python"
    if device == "[1] cpu":
        logging.info("use cpu, skip test device")
    elif device == "[2] gpu":
        # cuda check script
        cuda_check_script = """
import torch
if torch.cuda.is_available():
    print("True")
else:
    print("False")
"""
        try:
            result = subprocess.run([python_exec, "-c", cuda_check_script], text=True, capture_output=True, check=True)
            if result.stdout.strip() == "True":
                logging.info("test cuda success")
            else:
                logging.error("test cuda failed")
                exit(1)
        except Exception as e:
            logging.error("test cuda failed")
            logging.error(e)
            exit(1)
    elif device == "[3] tpu":
        # tpu check script
        tpu_check_script = """
import jax
if jax.devices("tpu"):
    print("True")
else:
    print("False")
"""
        try:
            result = subprocess.run([python_exec, "-c", tpu_check_script], text=True, capture_output=True, check=True)
            if result.stdout.strip() == "True":
                logging.info("test tpu success")
            else:
                logging.error("test tpu failed")
                exit(1)
        except Exception as e:
            logging.error("test tpu failed")
            logging.error(e)
            exit(1)


# setup env
# @param env_path: the path of virtual env
# @param device: the device that user want to use
# @param pytorch_version: the version of pytorch
# @param cuda_version: the version of cuda
def setup_env(env_path, device, pytorch_version, cuda_version):
    # install neccessary packages
    # use apt-get install screen, unzip, zip, git, wget, python3-pip, python3-venv
    try:
        subprocess.run(["apt-get", "update"])
        subprocess.run(["apt-get", "install", "-y", "screen", "unzip", "zip", "git", "wget", "python3-pip", "python3-venv"])
    except Exception as e:
        logging.error("install neccessary packages failed, this script only support ubuntu")
        logging.error(e)
        exit(1)

    create_env(env_path)
    if device == "[1] cpu" or device == "[2] gpu":
        install_pytorch(env_path, device, pytorch_version, cuda_version)
    elif device == "[3] tpu":
        install_jax(env_path)
    
    pip_exec = env_path + "/bin/pip"

    # no matter what device user want to use, install diffusers from source
    # pip install git+https://github.com/huggingface/diffusers
    try:
        result = subprocess.run([pip_exec, "install", "git+https://github.com/huggingface/diffusers"])
        if result.returncode == 0:
            logging.info("install diffusers success")
        else:
            logging.error("install diffusers failed")
    except Exception as e:
        logging.error("install diffusers failed")
        logging.error(e)
        exit(1)

    # install other packages
    if device == "[1] cpu" or device == "[2] gpu":
        # install these packages if user want to use cpu or gpu
        # accelerate>=0.16.0
        # torchvision
        # transformers>=4.25.1
        # ftfy
        # tensorboard
        # datasets
        result = subprocess.run([pip_exec, "install", "accelerate>=0.16.0", "torchvision", "transformers>=4.25.1", "ftfy", "tensorboard", "datasets"])
        if result.returncode == 0:
            logging.info("install common packages success for cpu or gpu")
        else:
            logging.error("install common packages failed for cpu or gpu")
    elif device == "[3] tpu":
        # install these packages if user want to use tpu
        # transformers>=4.25.1
        # datasets
        # flax
        # optax
        # torch
        # torchvision
        # ftfy
        # tensorboard
        # Jinja2
        result = subprocess.run([pip_exec, "install", "transformers>=4.25.1", "datasets", "flax", "optax", "torch", "torchvision", "ftfy", "tensorboard", "Jinja2"])
        if result.returncode == 0:
            logging.info("install common packages success for tpu")
        else:
            logging.error("install common packages failed for tpu")
    test_device(env_path, device)




if __name__ == "__main__":
    # ask what user wants to do
    # 1. setup train env
    # 2. start training
    # 3. convert models
    # that's all options, the option should start with [number]
    opration = questionary.select(
        "What do you want to do?",
        choices=[
            "[1] setup train env",
            "[2] start training",
            "[3] convert models",
        ],
    ).ask()

    if opration == "[1] setup train env":
        # if opration == "1. setup train env":
        # ask what device using, cpu, gpu, tpu
        device = questionary.select(
            "What device do you want to use?",
            choices=[
                "[1] cpu",
                "[2] gpu",
                "[3] tpu",
            ],
        ).ask()

        # if device == "[3] tpu":
        # set cuda and python version to None
        if device == "[3] tpu":
            pytorch_version = None
            cuda_version = None

        # if device == "[1] cpu" or device == "[2] gpu":
        # ask what pytorch version will be use
        if device == "[1] cpu" or device == "[2] gpu":
            pytorch_version = questionary.select(
                "What pytorch version do you want to use?",
                choices=[
                    "[1] 2.0.1",
                    "[2] 1.13.1",
                ],
            ).ask()
        
        # if device is GPU, ask cuda version
        # if pytorch version is 2.0.1, cuda version could be 11.8 or 11.7
        # if pytorch version is 1.13.1, cuda version could be 11.7 or 11.6
        if device == "[2] gpu":
            if pytorch_version == "[1] 2.0.1":
                cuda_version = questionary.select(
                    "What cuda version do you want to use?",
                    choices=[
                        "[1] 11.8",
                        "[2] 11.7",
                    ],
                ).ask()
            elif pytorch_version == "[2] 1.13.1":
                cuda_version = questionary.select(
                    "What cuda version do you want to use?",
                    choices=[
                        "[1] 11.7",
                        "[2] 11.6",
                    ],
                ).ask()

        # ask the vertrual env path, default is ~/train_env
        env_path = questionary.text(
            "What is the path of the virtual environment? (default: ~/train_env)",
            default="~/train_env",
        ).ask()

        # setup env
        setup_env(env_path, device, pytorch_version, cuda_version)


    elif opration == "[2] start training":
        pass

    elif opration == "[3] convert models":
        pass
