import os
import json
import io


def get_all_file_names_in_dir(folder_path, file_type):
    return [x for x in os.listdir(folder_path) if x.endswith(file_type)]


def get_json_from_json_file(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
