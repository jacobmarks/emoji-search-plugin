"""Emojis plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""



from bson import json_util
import json
import numpy as np
import pyperclip
from sentence_transformers.cross_encoder import CrossEncoder

import fiftyone as fo

import fiftyone.operators as foo
from fiftyone.operators import types

import fiftyone.zoo as foz
from fiftyone import ViewField as F


cross_encoder_name = "cross-encoder/stsb-distilroberta-base"
embedding_model_name = "clip-vit-base32-torch"

def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _get_basic_search_results(prompt, dataset):
    model = foz.load_zoo_model(embedding_model_name)        
    query_embedding = model.embed_prompt(f"A photo of {prompt}")
    basic_search_results = dataset.sort_by_similarity(query_embedding, k = 30)
    return basic_search_results


def _refine_search_results(prompt, dataset, subview):
    threshold = 0.1
    cross_encoder = CrossEncoder(cross_encoder_name)
    corpus = subview.values("description_gpt4")
    ids = subview.values("id")

    sentence_pairs = [[prompt, description.replace("A photo of", "")] for description in corpus]
    scores = cross_encoder.predict(sentence_pairs)
    sim_scores_argsort = reversed(np.argsort(scores))

    sorted_ids = [ids[i] for i in sim_scores_argsort if scores[i] > threshold]

    return dataset.select(sorted_ids, ordered=True) if sorted_ids else dataset.select(ids[:10], ordered=True)


def _get_emoji_from_sample(sample):
    unicode_str = sample.unicode
    # Split the string at spaces and convert each part
    emoji_parts = unicode_str.split()
    emoji_chars = [chr(int(part.replace('U+', ''), 16)) for part in emoji_parts]
    # Combine parts to form the emoji
    emoji = ''.join(emoji_chars)
    return emoji

class SearchEmojis(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="search_emojis",
            label="Emoji Search: find emojis similar to a prompt",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_placement(self, ctx):
        if "emoji" in ctx.dataset.name.lower():
            return types.Placement(
                types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
                types.Button(
                    label="Search Emojis",
                    icon="/assets/icon.svg",
                ),
            )
        else:
            return types.Placement()

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Emoji Search",
            description=("Search for emojis similar to a prompt"),
        )
        inputs.str(
            "prompt",
            label="Prompt",
            description="The prompt to search for emojis similar to",
        )
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset = ctx.dataset
        prompt = ctx.params.get("prompt", None)
        basic_view = _get_basic_search_results(prompt, dataset)
        view = _refine_search_results(prompt, dataset, basic_view)
        
        ctx.trigger(
            "set_view",
            params=dict(view=serialize_view(view)),
        )


class CopyEmojiToClipboard(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="copy_emoji_to_clipboard",
            label="Emoji Copy: copy current emojis to clipboard",
            dynamic=True,
            unlisted=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        sample_id = ctx.current_sample
        sample = ctx.dataset[sample_id]
        emoji = _get_emoji_from_sample(sample)

        form_view = types.View(
            label="Copy Emoji",
            description=(f"Copy emoji {emoji} to clipboard"),
        )
        return types.Property(inputs, view=form_view)

    def resolve_placement(self, ctx):
        if "emoji" in ctx.dataset.name.lower():
            return types.Placement(
                types.Places.SAMPLES_VIEWER_ACTIONS,
                types.Button(
                    label="Copy Emoji to Clipboard",
                    icon="/assets/copy_icon.svg",
                ),
            )
        else:
            return types.Placement()
        
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("emoji", label="Copied Emoji to Clipboard")
        return types.Property(outputs)

    def execute(self, ctx):
        sample_id = ctx.current_sample
        sample = ctx.dataset[sample_id]
        emoji = _get_emoji_from_sample(sample)
        pyperclip.copy(emoji)
        return {"emoji": emoji}

        

def register(plugin):
    plugin.register(SearchEmojis)
    plugin.register(CopyEmojiToClipboard)
