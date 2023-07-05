# wget https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/setup.py
import questionary
import subprocess
import logging
import os
import requests
import yaml

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
    # need sudo permission
    try:
        subprocess.run(["sudo", "apt-get", "update"])
        subprocess.run(["sudo","apt-get", "install", "-y", "screen", "unzip", "zip", "git", "wget", "python3-pip", "python3-venv"])
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

# load yaml from url
# @param url: the url of yaml
# @return: the yaml object
def load_yaml_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return yaml.safe_load(response.text)
    except Exception as e:
        logging.error("load template from github failed")
        logging.error(e)
        exit(1)

# create dreambooth_lora train config
# @param project_name: the name of project
# @param train_config_type: the type of train config
# @param train_config_path: the path of train config
def create_train_config_dreambooth_lora(project_name, train_config_type, train_config_path):
    # get usr home dir
    usr_home_dir = os.path.expanduser("~")

    # create train config
    # if train_config_type == "[1] [GPU] dreambooth-lora":
    # download the template from github then load it for modify
    if train_config_type == "[1] [GPU] dreambooth-lora":
        config = load_yaml_from_url("https://raw.githubusercontent.com/Eric-liucn/TPU_TRAIN_TOOL/main/train_config_templates/dreambooth_lora_gpu_template.yaml")
    
    # check if config is None
    if config is None:
        logging.error("load template from github failed")
        exit(1)

    # ask user to input the config
    # ask pretrained_model_name_or_path, default is runwayml/stable-diffusion-v1-5
    # validate if it in xxx/xxx format or a path
    # if not try again
    pretrained_model_name_or_path = questionary.text(
        "What is the pretrained model name or path?",
        default="runwayml/stable-diffusion-v1-5",
        validate=lambda text: True if "/" in text or os.path.exists(text) else "Please input a valid pretrained model name or path"
    ).ask()

    # set config's pretrained_model_name_or_path
    config["pretrained_model_name_or_path"] = pretrained_model_name_or_path

    # ask instance_data_dir, default is user home dir + project name + data + instance
    # then set config's instance_data_dir
    instance_data_dir = questionary.path(
        "What is the instance data dir?",
        default=usr_home_dir + "/" + project_name + "/data/instance"
    ).ask()
    config["instance_data_dir"] = instance_data_dir

    # ask the class_data_dir default is user home dir + project name + data + class
    class_data_dir = questionary.path(
        "What is the class data dir?",
        default=usr_home_dir + "/" + project_name + "/data/class"
    ).ask()
    config["class_data_dir"] = class_data_dir

    # ask the instance_prompt, default is ""
    instance_prompt = questionary.text(
        "What is the instance prompt?",
        default=""
    ).ask()
    config["instance_prompt"] = instance_prompt

    # ask the class_prompt, default is ""
    class_prompt = questionary.text(
        "What is the class prompt?",
        default=""
    ).ask()
    config["class_prompt"] = class_prompt

    # ask num_train_epochs
    num_train_epochs = questionary.text(
        "How many train epochs?",
        default="100"
    ).ask()
    config["num_train_epochs"] = num_train_epochs

    # ask learning_rate default is 1e-4
    learning_rate = questionary.text(
        "What is the learning rate?",
        default="1e-4"
    ).ask()
    config["learning_rate"] = learning_rate

    # ask lr_scheduler
    lr_scheduler = questionary.select(
        "What is the lr scheduler?",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup"
        ],
        default="constant_with_warmup"
    ).ask()
    config["lr_scheduler"] = lr_scheduler

    # ask if use_8bit_adam
    use_8bit_adam = questionary.confirm(
        "Do you want to use 8bit adam?",
        default=False
    ).ask()
    config["use_8bit_adam"] = use_8bit_adam

    # ask if enable_xformers_memory_efficient_attention
    enable_xformers_memory_efficient_attention = questionary.confirm(
        "Do you want to enable xformers memory efficient attention?",
        default=False
    ).ask()
    config["enable_xformers_memory_efficient_attention"] = enable_xformers_memory_efficient_attention

    # ask  mixed_precision
    mixed_precision = questionary.select(
        "What is the mixed precision?",
        choices=[
            "bf16",
            "fp16",
            "no"
        ],
        default="no"
    ).ask()
    config["mixed_precision"] = mixed_precision

    # ask if train_text_encoder
    train_text_encoder = questionary.confirm(
        "Do you want to train text encoder?",
        default=True
    ).ask()
    config["train_text_encoder"] = train_text_encoder

    # save config to train_config_path
    # and log it
    with open(train_config_path, "w") as f:
        yaml.dump(config, f)
    logging.info("create train config success")
    logging.info("train config path: " + train_config_path)



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
            "[4] create train config"
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

        # ask the vertrual env path, default is the user's home directory + train_env
        defautl_env_path = os.path.expanduser("~") + "/train_env"
        env_path = questionary.text(
            "What is the path of the virtual environment? (default: {})".format(defautl_env_path),
            default=defautl_env_path,
        ).ask()

        # setup env
        setup_env(env_path, device, pytorch_version, cuda_version)


    elif opration == "[2] start training":
        pass

    elif opration == "[3] convert models":
        pass

    elif opration == "[4] create train config":

        # user home dir path
        user_home_dir = os.path.expanduser("~")

        # ask user which kind if train config they want to create
        # LoRA, DreamBooth, Text-to-image
        train_config_type = questionary.select(
            "What kind of train config do you want to create?",
            choices=[
                "[1] [GPU] dreambooth-lora",
                "[2] [GPU] dreambooth",
                "[3] [GPT] text-to-image",
                "[4] [TPU] dreambooth-lora",
                "[5] [TPU] dreambooth",
                "[6] [TPU] text-to-image",
            ],
        ).ask()

        # ask the project name
        project_name = questionary.text(
            "What is the project name?",
            default="",
        ).ask()

        # no matter which kind of train config user want to create, ask the path to store the train config created
        # default path is user home dir + project_name_train_config.yaml
        train_config_path = questionary.path(
            "Where do you want to store the train config? (default: {}/train_config.yaml)".format(user_home_dir),
            default=user_home_dir + "/{}_train_config.yaml".format(project_name),
        ).ask()

        # if train_config_type == "[1] [GPU] dreambooth-lora":
        if train_config_type == "[1] [GPU] dreambooth-lora":
            create_train_config_dreambooth_lora(project_name, train_config_type, train_config_path)

