## Emoji Creation Plugin

### Updates

This plugin is a Python plugin that allows you to create emojis from text prompts
and add them to the dataset... if they are unique enough!

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/emoji-maker
```

You will also need to install the plugin's requirements:

```shell
pip install -r requirements.txt
```

## Operators

### `create_emoji`

This operator creates an emoji from a text prompt and adds it to the dataset.

## To Do

1. Add moderation to the emoji creation process using OpenAI moderation API.
2. Prohibit names of people. Likely with LangChain function calling.
3. Only activate operator when dataset name is "Emojis"
