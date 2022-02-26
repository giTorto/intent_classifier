import re
import yaml
import os
from os import listdir
from os.path import isfile, join

ent_regex = re.compile(r'\[(?P<entity_text>[^\]]+)'
                       r'\]\((?P<entity>[^:)]*?)'
                       r'(?:\:(?P<value>[^)]+))?\)')  # [entity_text](entity_type(:entity_synonym)?)


def build_entity(start_index, end_index, entity_value, entity_type):
    return {
        "start_index": start_index,
        "end_index": end_index,
        "entity": entity_type,
        "value": entity_value
    }


class Message(object):

    def __init__(self, text, intent, entities=None):
        self.text = text
        self.intent = intent
        self.entities = entities

    def set(self, property, value):
        if property == "entities":
            self.entities = value
        elif property == "intent":
            self.intent = value
        elif property == "text":
            self.text = value
        else:
            raise Exception("Not a valid property")


def _find_entities_in_training_example(example):
    """Extracts entities from a markdown intent example."""
    entities = []
    offset = 0
    for match in re.finditer(ent_regex, example):
        entity_text = match.groupdict()['entity_text']
        entity_type = match.groupdict()['entity']
        entity_value = match.groupdict()['value'] if match.groupdict()['value'] else entity_text

        start_index = match.start() - offset
        end_index = start_index + len(entity_text)
        offset += len(match.group(0)) - len(entity_text)

        entity = build_entity(start_index, end_index, entity_value, entity_type)
        entities.append(entity)

    return entities


def _parse_training_example(example, intent):
    """Extract entities and synonyms, and convert to plain text."""
    entities = _find_entities_in_training_example(example)
    plain_text = re.sub(ent_regex, lambda m: m.groupdict()['entity_text'], example)
    message = Message(plain_text, {'intent': intent})
    if len(entities) > 0:
        message.set('entities', entities)
    return message


def load_data(file_name):

    with open(file_name, 'r') as in_file:
        training_data = yaml.safe_load(in_file)

    data_point = []
    for intent_obj in training_data.get("nlu"):
        intent = intent_obj.get('intent')
        examples = intent_obj.get("examples").split("\n-")

        for example in examples:
            message = _parse_training_example(example.strip(), intent)
            data_point.append(message)

    return data_point


def load_training_data_folder(folder_name):
    file_names = [join(folder_name,f) for f in listdir(folder_name) if isfile(join(folder_name, f))]

    training_data = None
    test_data = None
    for file_name in file_names:
        if "train" in file_name:
            training_data = load_data(file_name)
        elif "test" in file_name:
            test_data = load_data(file_name)
        else:
            raise Exception(f"What the fuck is this? {file_name}")

    if training_data is None or test_data is None:
        raise Exception("Train or Test is None")
    return training_data, test_data


