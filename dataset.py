import json


def load_groups(path):

    with open(path, "r", encoding="utf-8") as f:
        groups = json.load(f)

    return groups