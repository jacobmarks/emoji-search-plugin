"""Emojis plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""


import numpy as np
import os
import pyperclip
from sentence_transformers.cross_encoder import CrossEncoder

import eta.core.web as etaw

import fiftyone as fo

import fiftyone.operators as foo
from fiftyone.operators import types

import fiftyone.zoo as foz
from fiftyone import ViewField as F


cross_encoder_name = "cross-encoder/stsb-distilroberta-base"
embedding_model_name = "clip-vit-base32-torch"


def _is_teams_deployment():
    val = os.environ.get("FIFTYONE_INTERNAL_SERVICE", "")
    return val.lower() in ("true", "1")


TEAMS_DEPLOYMENT = _is_teams_deployment()


def _get_basic_search_results(prompt, dataset):
    model = foz.load_zoo_model(embedding_model_name)
    query_embedding = model.embed_prompt(f"{prompt}")
    basic_search_results = dataset.sort_by_similarity(
        query_embedding, k=100, brain_key="img_sim", dist_field="img_dist"
    )
    basic_search_results = basic_search_results.match(F("img_dist") < 0.9)
    ordered_name_sim_ids = basic_search_results.sort_by_similarity(
        query_embedding, brain_key="name_sim"
    ).values("id")
    return basic_search_results, ordered_name_sim_ids


def _reciprocal_rank(rank):
    return 1.0 / rank if rank > 0 else 0


def _get_ranks(sids):
    return {sid: i + 1 for i, sid in enumerate(sids)}


def _fuse_reciprocal_ranks(rank_lists):
    all_rank_ids = set()
    for ranks in rank_lists:
        all_rank_ids.update(ranks.keys())

    max_rank = len(all_rank_ids) + 1
    fused_ranks = {rid: 0 for rid in all_rank_ids}

    for ranks in rank_lists:
        for rid in all_rank_ids:
            rank = ranks.get(rid, max_rank)
            fused_ranks[rid] += _reciprocal_rank(rank)

    return sorted(fused_ranks, key=fused_ranks.get, reverse=True)


def _refine_search_results(prompt, dataset, subview, ordered_name_sim_ids):
    threshold = 0.15

    cross_encoder = CrossEncoder(cross_encoder_name)
    ids = subview.values("id")

    desc_corpus = subview.values("description")
    name_corpus = subview.values("name")

    desc_sentence_pairs = [
        [prompt, description] for description in desc_corpus
    ]
    name_sentence_pairs = [[prompt, name] for name in name_corpus]

    desc_scores = cross_encoder.predict(desc_sentence_pairs)
    name_scores = cross_encoder.predict(name_sentence_pairs)

    desc_scores_argsort = reversed(np.argsort(desc_scores))
    name_scores_argsort = reversed(np.argsort(name_scores))

    desc_refined_ids = [
        ids[i] for i in desc_scores_argsort if desc_scores[i] > threshold
    ]

    name_refined_ids = [
        ids[i] for i in name_scores_argsort if name_scores[i] > threshold
    ]

    desc_ranks = _get_ranks(desc_refined_ids)
    name_ranks = _get_ranks(name_refined_ids)
    img_sim_ranks = _get_ranks(ids)
    name_sim_ranks = _get_ranks(ordered_name_sim_ids)

    ranks_list = [desc_ranks, name_ranks, img_sim_ranks, name_sim_ranks]

    fused_ranks = _fuse_reciprocal_ranks(ranks_list)[:20]

    return (
        dataset.select(fused_ranks, ordered=True)
        if fused_ranks
        else dataset.select(desc_refined_ids[:10], ordered=True)
    )


def _get_emoji_from_sample(sample):
    unicode_str = sample.unicode
    # Split the string at spaces and convert each part
    emoji_parts = unicode_str.split()
    emoji_chars = [
        chr(int(part.replace("U+", ""), 16)) for part in emoji_parts
    ]
    # Combine parts to form the emoji
    emoji = "".join(emoji_chars)
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
            return types.Placement()  # pylint: disable=no-value-for-parameter

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
        basic_view, ordered_name_sim_ids = _get_basic_search_results(
            prompt, dataset
        )
        view = _refine_search_results(
            prompt, dataset, basic_view, ordered_name_sim_ids
        )

        ctx.ops.set_view(view=view)


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
        if "emoji" in ctx.dataset.name.lower() and not TEAMS_DEPLOYMENT:
            return types.Placement(
                types.Places.SAMPLES_VIEWER_ACTIONS,
                types.Button(
                    label="Copy Emoji to Clipboard",
                    icon="/assets/copy_icon.svg",
                ),
            )
        else:
            return types.Placement()  # pylint: disable=no-value-for-parameter

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("emoji", label="Copied Emoji to Clipboard")
        return types.Property(outputs)

    def execute(self, ctx):
        sample_id = ctx.current_sample
        sample = ctx.dataset[sample_id]
        emoji = _get_emoji_from_sample(sample)
        try:
            pyperclip.copy(emoji)
        except:
            pass
        return {"emoji": emoji}


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _download_emoji_images():
    fo_dir = fo.config.default_dataset_dir
    dataset_dir = os.path.join(fo_dir, "emojis")
    images_dir = os.path.join(dataset_dir, "images")
    images_zip_path = os.path.join(dataset_dir, "images.zip")
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
        print(f"Downloading images to {dataset_dir}...")

        # Download the file
        etaw.download_google_drive_file(
            "1oODj7JMADEzMmco_oP8mdIChXusG1VKG", path=images_zip_path
        )
        print("Download complete.")

    if not os.path.exists(images_dir):
        print(f"Extracting images to {dataset_dir}...")
        import zipfile

        with zipfile.ZipFile(images_zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")

    if os.path.isfile(images_zip_path):
        # Delete the zip file
        os.remove(images_zip_path)

    return images_dir


def _download_emoji_data():
    fo_dir = fo.config.default_dataset_dir
    dataset_dir = os.path.join(fo_dir, "emojis")
    data_dir = os.path.join(dataset_dir, "emojis")
    dataset_data_path = os.path.join(dataset_dir, "emojis.zip")

    if not os.path.exists(data_dir):
        # Download the file
        etaw.download_google_drive_file(
            "1x-QTsE3Hpxtdh-wSITluR40_7S25UmW-", path=dataset_data_path
        )
        print("Download complete.")

    if os.path.isfile(dataset_data_path):
        print(f"Extracting descriptions to {dataset_dir}...")
        import zipfile

        with zipfile.ZipFile(dataset_data_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")

        # Delete the zip file
        os.remove(dataset_data_path)

    return data_dir


class CreateEmojiDataset(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="create_emoji_dataset",
            label="Emoji Dataset: create a dataset of emojis",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Create Emoji Dataset",
            description=(
                "Create a dataset of emojis. This may take a minute or two."
            ),
        )
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        if "emojis" in fo.list_datasets():
            return

        images_dir = _download_emoji_images()
        data_dir = _download_emoji_data()

        emoji_dataset = fo.Dataset.from_dir(
            data_dir,
            dataset_type=fo.types.FiftyOneDataset,
            name="emojis",
            persistent=True,
        )
        for sample in emoji_dataset.iter_samples(autosave=True):
            sample.filepath = os.path.join(images_dir, sample.filename)

        ctx.ops.reload_dataset()


def register(plugin):
    plugin.register(SearchEmojis)
    plugin.register(CopyEmojiToClipboard)
    plugin.register(CreateEmojiDataset)
