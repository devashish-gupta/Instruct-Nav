"""
Main script to parse bag files.
"""
import os
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
import yaml
from pyntcloud import PyntCloud

from musohu_parser import MuSoHuParser # add scand parser here if you want

# had to implement this since get_conf() was not available
class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    def __str__(self):
        return self._format_dict(self.__dict__)

    def _format_dict(self, d, indent=0):
        lines = []
        for key, value in d.items():
            if isinstance(value, ConfigObject):
                lines.append("  " * indent + f"{key}:")
                lines.append(value._format_dict(value.__dict__, indent + 1))
            else:
                lines.append("  " * indent + f"{key}: {value}")
        return "\n".join(lines)

def get_conf(config_file):
    """Reads the configuration from a YAML file and returns the configuration."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return ConfigObject(config)


def create_samples(input_path, language_dict, num_instruction = 2, obs_window: int = 6, pred_window: int = 8) -> dict:
    #print("LANGUAGE DICT: ", language_dict)
    """Create multiple samples from the parsed data folder

    input_path (PosixPath): directory of the parsed trajectory
    obs_window (int): observation window (history)
    pred_window (int): prediction window
    """
    with input_path.open("rb") as f:
        data = pickle.load(f)
    
    all_frames = sorted(list([x for x in (input_path.parent / "rgb").iterdir()]), key=lambda x: int(x.name.split(".")[0]))
    #print("ALL FRAMES LEN", len(all_frames))
    traj_len = len(data["position"])
    seq_len = obs_window + pred_window
    positions = []
    goal_positions = []
    yaws = []
    goal_yaws = []
    vws = []
    goal_vws = []
    past_frames = []
    goal_frames = []
    past_pc = []

    # Add instructions to the samples.pkl
    instruction_list = []
    num_frames = num_instruction * 10
    past_instructions_dict = {}
    frame_instructions = []

    # Add point-cloud data
    all_pc = sorted(list([x for x in (input_path.parent / "point_cloud").iterdir()]), key=lambda x: int(x.name.split(".")[0]))
    

    # need to return the most recent instruction, 

    for i in range(traj_len - seq_len):
        # past and future positions
        positions.append(data["position"][i : i + obs_window])
        goal_positions.append(data["position"][i + obs_window : i + seq_len])
        # past and future yaw
        yaws.append(data["yaw"][i : i + obs_window])
        goal_yaws.append(data["yaw"][i + obs_window : i + seq_len])
        # past and future vw
        vws.append(data["vw"][i : i + obs_window])
        goal_vws.append(data["vw"][i + obs_window : i + seq_len])
        # store image addresses
        past_frames.append(all_frames[i : i + obs_window])
        goal_frames.append(all_frames[i + obs_window : i + seq_len])
        #print("PAST FRAMES START AND END: ", past_frames[0][0], past_frames[-1][-1],"\n")
        #print("GOAL FRAMES START AND END: ", goal_frames[0][0], goal_frames[-1][-1], "\n")

        past_pc = all_pc[i:i+obs_window]

        # Add the corresponding language instructions for the entire frame window
        start_frame_index = i
        end_frame_index = i+obs_window

        #print("START FRAME: ", start_frame_index)

        frame_instructions = [value for value in past_instructions_dict.values()] # all past instructions
        latest_instruction = None
        for j in range(start_frame_index,end_frame_index):
            target_frame = os.path.normpath(all_frames[j])
            #extract number from the path
            # Split the path by '\\' to get the individual components
            path_components = target_frame.split('\\')

            # Get the filename component
            filename = path_components[-1]

            # Split the filename by '.' to separate the filename and extension
            filename_without_extension = filename.split('.')[0]

            # Extract the digits from the filename
            target_digits = ''.join(filter(str.isdigit, filename_without_extension))

            #print("TARGET FRAME: ", target_digits,".jpg")
            for dict in language_dict:
                images = dict.get('images', [])
                instructions = dict.get('instructions', [])
                #print("INSTRUCTIONS: ", instructions)
                for image, instruction in zip(images, instructions):
                    image = os.path.normpath(image)
                    path_components = image.split('\\')

                    # Get the filename component
                    filename = path_components[-1]

                    # Split the filename by '.' to separate the filename and extension
                    filename_without_extension = filename.split('.')[0]

                    # Extract the digits from the filename
                    current_digits = ''.join(filter(str.isdigit, filename_without_extension))

                    #print("CURRENT FRAME: ", current_digits,".jpg")

                    if current_digits == target_digits:
                        past_instructions_dict[current_digits] = instruction
                        #print(f"IMAGE {current_digits} MATCHES TARGET {target_digits}")
                        #print(f"IMAGE {current_digits} INSTRUCTION: ", instruction)
                        frame_instructions.append(instruction)
                        #print(frame_instructions)
                        #latest_instruction = instruction
                        break
                        
    if len(frame_instructions) > 0:      
        #print("FRAME INSTRUCTIONS: ", frame_instructions)  
        instruction_list = frame_instructions[-num_instruction:]  # Append the most recent instructions)
        #print("LEN FRAME INSTRUCTIONS: ", len(frame_instructions))
        # print("INSTRUCTION LIST: ", instruction_list)

        #print("MOST RECENT INSTRUCTIONS: ", instruction_list)
    # print(type(instruction_list))
    post_processed = {
        "past_positions": positions,
        "future_positions": goal_positions,
        "past_yaw": yaws,
        "future_yaw": goal_yaws,
        "past_vw": vws,
        "future_vw": goal_vws,
        "past_frames": past_frames,
        "future_frames": goal_frames,
        "past_pc": past_pc,
        "instructions": instruction_list
    }
    return post_processed


def merge(base_dict: dict, new_dict: dict):
    """Merges two dictionary together

    base_dict (dict): The base dictionary to be updated
    new_dict (dict): The new data to be added to the base dictionary
    """
    # assert base_dict is None, "Base dictionary cannot be None"
    assert (
        base_dict.keys() == new_dict.keys()
    ), "The two dictionaries must have the same keys"
    for key in base_dict.keys():
        base_dict[key].extend(new_dict[key])

    return base_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        default="musohu",
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "-c",
        "--conf",
        default="../conf/musohu_parser",
        type=str,
        help="Config file address.",
    )
    parser.add_argument(
        "-cs",
        "--create_samples",
        action="store_true",
        help="Create samples. Applicable only after parsing bags.",
    )
    parser.add_argument(
        "-lang",
        "--language",
        default="compiled_language_modality.pkl",
        type=str,
        help="Compiled language modality .pkl file",
    )
    args = parser.parse_args()
    cfg_dir = args.conf
    cfg = get_conf(cfg_dir)
    # dataset = "musohu" if "musohu" in cfg_dir.lower() else "scand"
    dataset = args.name
    if args.create_samples:
        # Creating samples
        parsed_path = Path(cfg.parsed_dir) / "samples.pkl"
        #print("PARSED PATH: ", parsed_path)
        save_path = Path(cfg.save_dir) / "samples.pkl"
        if (parsed_path).exists():
            parsed_path.rename(f"{parsed_path.stem + '_old' + save_path.suffix}")
        # List all the pickle files
        list_pickles = list(parsed_path.parent.glob("**/*traj_data.pkl"))
        #print("LIST PICKLES: ", list_pickles)
        # list_pickles = [x for x in Path(cfg.save_dir).iterdir() if x.suffix == '.pkl']
        # Base dictionary to store data
        base_dict = dict()
        # Language dictionary
        with open(args.language, 'rb') as f:
            language_dict = list(pickle.load(f).values())
        # Iterate over processed files and create samples from them
        bar = tqdm(list_pickles, desc="Creating samples: ")
        for file_name in bar:
            bar.set_postfix(Trajectory=f"{file_name}")
            print("FILE_NAME: ", file_name)
            post_processed = create_samples(
                file_name, language_dict=language_dict, num_instruction=100, obs_window=cfg.obs_len, pred_window=cfg.pred_len
            )
            if bool(base_dict):
                base_dict = merge(base_dict, post_processed)
            else:
                base_dict = post_processed

        # Saving the final file
        with save_path.open("wb") as f:
            pickle.dump(base_dict, f)
    else:
        if dataset == "musohu":
            # cfg.musohu.update({"sample_rate": cfg.sample_rate})
            # cfg.musohu.update({"save_dir": cfg.save_dir})
            cfg.musohu.sample_rate = cfg.sample_rate
            cfg.musohu.parsed_dir = cfg.parsed_dir
            parser = MuSoHuParser(cfg.musohu)
            bag_files = Path(cfg.musohu.bags_dir).resolve()
            bag_files = [str(x) for x in bag_files.iterdir() if x.suffix == ".bag"]
            # if there are ram limitations, reduce the number of max_workers
            print(f'bags: {bag_files}')
            process_map(parser.parse_bags, bag_files, max_workers=os.cpu_count() - 4)
        
        else:
            raise Exception("Invalid dataset!")