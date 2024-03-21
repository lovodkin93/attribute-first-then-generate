import sys
import os
import json
from pathlib import Path
import re
from json_minify import json_minify

def read_minified_config(config_file_path) -> dict:
    with open(config_file_path, "r") as f:
        minified_config = json_minify(f.read())
        config = json.loads(minified_config)

    return config


def prepare_config_for_hf() -> dict:
    """
    Write minified config so huggingface can read it even if it has comments, and handle env variables
    """

    # Read non-minified config with comments
    config_file_path = sys.argv[-1]

    config = read_minified_config(config_file_path)

    # Handle env variables templates: {env:ENV_VARIABLE}
    for key, value in config.items():
        if isinstance(value, str):
            results = re.findall(r'{env:(.*)}', value)
            for env_variable in results:
                value = value.replace(f"${{env:{env_variable}}}", os.environ[env_variable])
            
            if any(results):
                config[key] = value
                
    # for distributed training:
    config["local_rank"]=int(os.environ.get("LOCAL_RANK", -1))
    # if len(sys.argv)>2:
    #     arg_name, arg_value = sys.argv[1].replace("--", "").replace("-", "_").split("=") # will convert '--local-rank=<number>' into ["local_rank", "<number>"] 
    #     config[arg_name] = arg_value
    
    save_minified_config(config, config_file_path)
    
    return config, config_file_path

def save_minified_config(config, config_file_path):
    # Save minified config
    new_config_path = f"{os.environ['TMPDIR']}/minified_configs/{config_file_path}"
    Path(os.path.dirname(new_config_path)).mkdir(parents=True, exist_ok=True)
    with open(new_config_path, "w") as f:
        f.write(json.dumps(config))

    sys.argv[-1] = new_config_path