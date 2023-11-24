# Quick recipe to add some datasets from OpenSLR to the Hugging Face plateform (WIP)

[OpenSLR](https://www.openslr.org/resources.php) is a gold mine for good-quality TTS datasets. It hosts for example many datasets from [Google Language Resources](https://github.com/google/language-resources) that are great resources for under-resourced languages.

In this guide, I'll provide some general guidelines and advices to quickly add datasets.

## 1. Download data locally
Use wget

## 2. Unzip data

If tar.gz, use python and tarfile.

If zip, unzip it using shell commands (like unzip).

**EXAMPLE - 2 FIRST STEPS**

```sh
#!/bin/bash

mkdir /home/yoach/datasets/google-tamil
cd /home/yoach/datasets/google-tamil
wget https://www.openslr.org/resources/65/about.html
wget https://www.openslr.org/resources/65/LICENSE
wget https://www.openslr.org/resources/65/line_index_female.tsv
wget https://www.openslr.org/resources/65/ta_in_female.zip
wget https://www.openslr.org/resources/65/line_index_male.tsv
wget https://www.openslr.org/resources/65/ta_in_male.zip


unzip ta_in_female.zip -d female/
unzip ta_in_male.zip -d male/
```

## 3. Decide how you'll structure the dataset

Usually, there are 3 layers of structure:
1. the dataset name -> easy to figure it out ;)
2. the different configurations, i.e the different parts of the datasets. It's how you divide the dataset in relevant subparts. For example, CML-TTS has been divided by languages. This one by dialect and speaker. If no configurations, you don't have to use one.
3. the different split. Usually train, test, dev. If no split, do train.

## 4. Locate where are the audio path and other features

Usually, for each configuration (remember -> in most cases languages), you have to locate where are listed the file path and other features.

## 5. Rest TODO
