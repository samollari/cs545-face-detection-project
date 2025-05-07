# cs545-face-detection-project

## Preprocessing

### Set Up

- [ ] Install [`uv` package manager](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
- [ ] Clone this repo
- [ ] Install dependencies with `uv sync`

### Running
Run with `uv run main.py` and specify a path to a zipped dataset.
```
$ uv run main.py
usage: main.py [-h] [--dry-run] zip_path [regex]
main.py: error: the following arguments are required: zip_path
```

The dataset to process should be structured as followed:
```
archive.zip
├── Class_A/
│   ├── img1.png
│   └── img2.png
├── Class_B/
│   ├── img1.png
```

We used [DigiFace-1M](https://github.com/microsoft/DigiFace1M#downloading-the-dataset) which comes properly structured.

The program will create and output to a directory structured like:
```
archive/
├── edges/
│   ├── Class_A/
│   │   ├── img1.png
│   │   └── img2.png
│   ├── Class_B/
│   │   ├── img1.png
└── processed/
    ├── Class_A/
    │   ├── img1.png
    │   └── img2.png
    ├── Class_B/
    │   ├── img1.png
```

You may optionally provide a Python-compatible regular expression to only process a subset of files. For example, to only process the first 5 images in the first 3 classes in the example above, you may specify `"^Class_[A-C]/img[1-5]\.png$"`

You may also enable the `--dry-run` flag to list all files to process while not performing any work.
