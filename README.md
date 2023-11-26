## Emoji Search Plugin

This plugin allows you to search for emojis based on the text you input. The
operators will only appear in menus in the FiftyOne App if `emoji` is in the
dataset's name (case-insensitive).

With `pyperclip`, you can also copy an emoji to your clipboard! This is useful
if you want to paste the emoji into a text message. The option should appear in
the sample viewer menu when you enter the sample modal.

## Under the Hood

The plugin uses a three-step search process. We utilize text descriptions
of the emojis, which are generated by GPT-4, and are of the form "A photo of ...",
the names of the emojis, as listed in the Kaggle dataset, and high-resolution
images of the emojis, which are 10x upscaled using ESRGAN.

Step 1 performs a relatively cheap pre-filtering search using precomputed CLIP
embeddings of the emoji images, and an on-the-fly CLIP embedding of the
input text. This step returns 100 emojis, which are then passed to step 2.

Step 2 performs a more expensive re-ranking using `distilroberta-base``
cross-encoder model from the [sentence-transformers](https://www.sbert.net/)
library, which computes a similarity score between two text inputs, rather than
relying on separate embeddings.

We do this twice: once with the description text of the emojis as the first input,
and the user prompt as the second input, and once with the emoji names as the
first input, and the user prompt as the second input. Additionally, we compute
the similarity score between the user prompt and the names of the emojis, using
CLIP embeddings.

Step 3 performs a reciprocal rank fusion of the four orderings we have generated:
CLIP image similarity, CLIP name similarity, cross-encoder description similarity,
and cross-encoder name similarity. We then take the top (at most) 20 emojis from
this fusion, above a certain threshold.

## Installation

This plugin requires `fiftyone>=0.23.0`. Upgrade your FiftyOne installation
before installing this plugin:

```shell
pip install fiftyone --upgrade
```

You can then install the plugin itself:

```shell
fiftyone plugins download https://github.com/jacobmarks/emoji-search-plugin
```

You will also need to install the plugin's requirements:

```shell
fiftyone plugins requirements @jacobmarks/emoji_search --install
```

## Operators

### `download_emoji_dataset`

This operator will download the emoji dataset images and metadata, and create a
FiftyOne `Dataset` from them. You can then use the other operators on this dataset.
This dataset will be called `emoji`, and will be saved in your default dataset directory,
which you can check with `fiftyone.config.default_dataset_dir`.

### `search_emojis`

This operator will semantically search for emojis based on the input text. It
will return a subset of the emojis in the dataset that are most semantically
similar to the input text, above a certain threshold.

### `copy_emoji_to_clipboard`

This operator will copy the emoji to the clipboard!

## Dataset Preperation

This plugin is designed to work with the Emoji Dataset (link to try.fiftyone.ai when available).
The dataset was constructed as follows:

1. Base64 encoded images of emojis and associated data were downloaded from the
   [Kaggle Emoji Dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset).

2. ESRGAN was used to 10x upscale the images.

3. Captions of the form "A photo of ..." were generated for each emoji using GPT-4 and post-processed.

4. CLIP embeddings were computed for each emoji, name, description, and image, and the embeddings were
   used to create vector similarity indexes.

5. For fun, image attributes such as contrast and saturation were computed for each emoji.
