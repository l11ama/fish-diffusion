Search.setIndex({"docnames": ["index", "pages/config", "pages/faq", "pages/guide", "pages/quality"], "filenames": ["index.md", "pages/config.md", "pages/faq.md", "pages/guide.md", "pages/quality.md"], "titles": ["Fish Diffusion", "Configuration", "FAQ", "Quick FishSVC Guide", "Enhancing Generation Quality"], "terms": {"fishsvc": 0, "prepar": 0, "environ": 0, "dataset": [0, 2], "singl": 0, "speaker": [0, 1], "multi": [0, 1], "train": [0, 1, 4], "time": [0, 1], "infer": [0, 4], "diff": [0, 1], "svc": [0, 1], "convers": 0, "TO": 0, "inferenc": 0, "diffsvc": [0, 2], "convert": [0, 2], "model": [0, 1, 2, 4], "preprocess": [0, 2, 3], "data": [0, 3, 4], "augment": [0, 3], "appendix": 0, "pitch": 0, "extractor": 0, "enhanc": 0, "faq": 0, "why": 0, "i": [0, 1, 3, 4], "so": [0, 3], "slow": 0, "my": [0, 3], "devic": 0, "audio": [0, 3], "blurri": 0, "weird": 0, "see": [0, 1, 3], "keyerror": 0, "pytorch": [0, 3], "lightning_vers": 0, "some": [0, 1], "error": [0, 3], "about": 0, "miss": 0, "kei": [0, 3], "when": 0, "resum": 0, "sinc": 1, "fish": [1, 4], "diffus": [1, 3, 4], "support": [1, 2], "variou": [1, 4], "modul": 1, "write": 1, "good": [1, 3], "config": [1, 2, 3, 4], "file": [1, 3], "essenti": [1, 4], "There": 1, "ar": [1, 2, 3], "mani": [1, 3], "exampl": [1, 3, 4], "folder": [1, 3], "To": [1, 2, 3, 4], "begin": 1, "you": [1, 2, 3, 4], "should": [1, 3], "creat": [1, 3], "exp_xxxxxx": 1, "py": [1, 3, 4], "follow": [1, 3, 4], "code": [1, 3], "_base_": [1, 3], "arch": 1, "diff_svc_v2": 1, "us": [1, 2, 3, 4], "v2": 1, "architectur": [1, 4], "trainer": [1, 3], "base": [1, 3], "default": [1, 3], "ddp": 1, "fp16": [1, 2], "schedul": 1, "warmup_cosin": 1, "cosin": 1, "learn": 1, "rate": 1, "scheulder": 1, "audio_fold": 1, "dataload": [1, 3], "current": [1, 2], "includ": [1, 3], "two": 1, "compon": 1, "text_features_extractor": 1, "pitch_extractor": [1, 4], "stabl": 1, "featur": 1, "hubertsoft": 1, "chinesehubertsoft": 1, "contentvec": 1, "although": 1, "chines": [1, 3], "second": [1, 3], "one": [1, 3], "work": [1, 3], "multilingu": 1, "outperform": 1, "vanilla": 1, "extract": 1, "can": [1, 2, 3, 4], "choos": 1, "parselmouthpitchextractor": 1, "crepepitchextractor": 1, "parselmouth": 1, "enough": 1, "most": 1, "case": [1, 3], "": [1, 3], "liter": 1, "100x": 1, "faster": 1, "than": [1, 3], "crepe": 1, "howev": [1, 4], "edg": 1, "more": [1, 2, 3], "robust": 1, "For": [1, 3, 4], "result": [1, 4], "dict": [1, 3], "type": [1, 3], "pretrain": [1, 2], "true": [1, 3], "lengyu": [1, 3], "hubert": [1, 4], "soft": 1, "gate_s": 1, "25": 1, "control": 1, "how": [1, 3], "much": 1, "inform": 1, "kept": 1, "too": 1, "larg": [1, 4], "lead": 1, "leak": 1, "note": [1, 3], "need": [1, 2, 3], "rerun": 1, "command": [1, 3, 4], "after": [1, 3], "chang": [1, 2, 3], "Be": 1, "naive_svc": [1, 3], "try": [1, 3], "load": [1, 2], "from": [1, 2, 3, 4], "valid": [1, 3], "want": [1, 2, 3, 4], "refer": [1, 3], "svc_hubert_soft_multi_speak": 1, "_delete_": 1, "delet": 1, "concatdataset": 1, "contact": 1, "multipl": 1, "naivesvcdataset": 1, "path": [1, 3], "speaker_0": 1, "first": [1, 3], "speaker_id": 1, "0": [1, 3, 4], "speaker_1": 1, "1": [1, 3], "ani": 1, "other": 1, "wai": 1, "do": [1, 3], "thi": [1, 2, 3, 4], "collate_fn": 1, "onli": [1, 3, 4], "speaker_encod": 1, "input_s": 1, "2": [1, 3, 4], "python": [1, 3, 4], "script": 1, "like": [1, 3], "But": 1, "what": 1, "have": [1, 2, 3], "100": 1, "t": [1, 3], "setup": 1, "manual": [1, 3], "don": [1, 3], "know": 1, "worri": 1, "structur": [1, 2, 3], "speaker0": 1, "xxx1": [1, 3], "wav": [1, 3, 4], "speaker1": 1, "test": [1, 3], "pick": 1, "fish_diffus": 1, "util": 1, "import": [1, 4], "get_speaker_map_from_subfold": 1, "get_datasets_from_subfold": 1, "speaker_map": 1, "updat": [1, 3], "subfold": 1, "train_dataset": 1, "build": 1, "valid_dataset": 1, "len": 1, "enabl": 1, "fixedpitchshift": 1, "key_shift": 1, "5": [1, 4], "probabl": [1, 2], "75": 1, "randompitchshift": 1, "randomtimestretch": 1, "factor": 1, "8": 1, "edit": 1, "same": [1, 3], "yet": 1, "2023": [1, 3], "03": [1, 3], "repo": 1, "harvest": 1, "dio": 1, "we": [1, 2, 4], "recommend": [1, 3, 4], "If": [1, 2, 3, 4], "your": [1, 2, 3, 4], "small": [1, 4], "otherwis": 1, "here": [1, 3], "comparison": 1, "pleas": 2, "make": [2, 3], "sure": [2, 3], "latest": [2, 3, 4], "version": [2, 4], "librari": 2, "cpu": [2, 3], "laptop": 2, "mai": [2, 3, 4], "fp32": [2, 3], "instead": [2, 3], "sound": 2, "noisi": 2, "wait": [2, 3], "step": [2, 3, 4], "indic": 2, "which": [2, 3, 4], "lightn": 2, "onnx": 2, "export": 2, "solv": 2, "problem": 2, "march": 3, "made": 3, "kangarroar": 3, "feb": 3, "01": 3, "17": 3, "ooppeenn": 3, "instal": 3, "power": 3, "gpu": 3, "googl": 3, "colab": 3, "option": 3, "get": 3, "start": 3, "conda": 3, "pc": 3, "miniconda": 3, "eat": 3, "lot": 3, "disk": 3, "space": 3, "The": [3, 4], "link": 3, "http": 3, "doc": 3, "io": 3, "en": 3, "html": 3, "user": 3, "through": 3, "mirror": 3, "sourc": 3, "accord": 3, "station": 3, "tuna": 3, "tsinghua": 3, "edu": 3, "cn": [3, 4], "anaconda": 3, "help": [3, 4], "open": 3, "name": 3, "3": [3, 4], "10": 3, "onc": 3, "done": 3, "call": 3, "access": 3, "activ": 3, "next": 3, "virtual": 3, "pdm": 3, "manag": 3, "project": 3, "depend": 3, "window": 3, "curl": 3, "ssl": 3, "raw": 3, "githubusercont": 3, "com": 3, "main": 3, "python3": 3, "linux": 3, "pypi": 3, "finish": 3, "set": [3, 4], "up": 3, "proce": 3, "download": 3, "github": 3, "either": 3, "click": 3, "zip": 3, "decompress": 3, "wherev": 3, "Or": 3, "clone": 3, "repositori": 3, "git": 3, "fishaudio": 3, "In": 3, "point": 3, "where": 3, "all": 3, "just": 3, "explor": 3, "bar": 3, "copi": 3, "full": 3, "run": [3, 4], "cd": 3, "c": 3, "document": 3, "difuss": 3, "sync": 3, "requir": 3, "nsf": 3, "hifigan": 3, "vocod": 3, "gener": 3, "an": 3, "automat": 3, "tool": [3, 4], "download_nsf_hifigan": 3, "It": 3, "put": 3, "checkpoint": [3, 4], "until": 3, "directori": [3, 4], "lxx": 3, "0xx8": 3, "xx2": 3, "0xxx2": 3, "xxx7": 3, "xxx007": 3, "few": 3, "20": 3, "check": 3, "qualiti": 3, "strongli": 3, "FOR": 3, "local": 3, "THAT": 3, "THE": 3, "aren": 3, "higher": 3, "long": 3, "IF": 3, "4gb": 3, "OF": 3, "vram": 3, "process": 3, "extract_featur": 3, "svc_content_vec": 3, "clean": [3, 4], "without": 3, "new": 3, "mess": 3, "origin": 3, "configur": 3, "batchsiz": 3, "accordingli": 3, "resourc": 3, "found": 3, "batch_siz": 3, "shuffl": 3, "num_work": 3, "persistent_work": 3, "fals": 3, "valu": 3, "batch": 3, "size": 3, "6": 3, "core": 3, "gtx": 3, "1650": 3, "number": 3, "worker": 3, "12": 3, "vari": 3, "better": 3, "baselin": 3, "abl": 3, "cuda": 3, "out": 3, "memori": 3, "lower": 3, "everi": 3, "save": 3, "ckpt": [3, 4], "By": 3, "000": 3, "log_every_n_step": 3, "val_check_interv": 3, "5000": 3, "check_val_every_n_epoch": 3, "none": 3, "max_step": 3, "300000": 3, "warn": 3, "fs2": 3, "nan": 3, "bf16": 3, "precis": 3, "16": 3, "callback": 3, "modelcheckpoint": 3, "filenam": 3, "epoch": 3, "valid_loss": 3, "2f": 3, "every_n_train_step": 3, "10000": 3, "save_top_k": 3, "learningratemonitor": 3, "logging_interv": 3, "line": 3, "1000": 3, "also": 3, "take": 3, "hard": 3, "drive": 3, "ask": 3, "wandb": 3, "account": 3, "look": 3, "graph": 3, "similar": 3, "tensorboard": 3, "termin": 3, "show": 3, "w": 3, "b": 3, "fulli": 3, "right": 3, "now": 3, "drop": 3, "input": [3, 4], "output": [3, 4], "everyth": 3, "add": 3, "pitch_adjust": 3, "mean": 3, "log": 3, "render": 3, "end": 3, "svc_hubert_soft": 3, "8liv3": 3, "pretti": 3, "fast": 3, "actual": 3, "less": 3, "minut": 3, "web": 3, "ui": 3, "gradio": 3, "straightforward": 3, "allow": 3, "immedi": 3, "specif": 3, "diff_svc_convert": 3, "svc_hubert_soft_diff_svc": 3, "model_train_400000": 3, "mismatch": 3, "residu": 3, "channel": 3, "normal": 3, "svc_hybert_soft_diff_svc": 3, "And": 3, "paramount": 4, "u": 4, "strive": 4, "deliv": 4, "highest": 4, "possibl": 4, "high": 4, "reverber": 4, "challeng": 4, "signific": 4, "artifact": 4, "address": 4, "ve": 4, "tradition": 4, "hifising": 4, "re": 4, "excit": 4, "introduc": 4, "our": 4, "reduct": 4, "technologi": 4, "shallow": 4, "denois": 4, "ensur": 4, "abov": 4, "properli": 4, "place": 4, "v1": 4, "improv": 4, "denoiser_cn_hubert": 4, "sampler_interv": 4, "skip_step": 4, "970": 4, "paramet": 4, "dictat": 4, "behavior": 4, "perform": 4, "30": 4, "preserv": 4, "accent": 4, "As": 4, "experi": 4, "etc": 4, "achiev": 4, "differ": 4, "outcom": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"fish": [0, 3], "diffus": 0, "content": 0, "quick": [0, 3], "guid": [0, 3], "configur": [0, 1], "gener": [0, 2, 4], "qualiti": [0, 4], "common": 0, "question": 0, "preprocess": 1, "dataset": [1, 3], "data": 1, "augment": 1, "appendix": 1, "pitch": 1, "extractor": 1, "faq": 2, "why": 2, "train": [2, 3], "i": 2, "so": 2, "slow": 2, "my": 2, "devic": 2, "audio": 2, "blurri": 2, "weird": 2, "see": 2, "keyerror": 2, "pytorch": 2, "lightning_vers": 2, "some": 2, "error": 2, "about": 2, "miss": 2, "kei": 2, "when": 2, "resum": 2, "fishsvc": 3, "prepar": 3, "environ": 3, "singl": 3, "speaker": 3, "multi": 3, "time": 3, "infer": 3, "diff": 3, "svc": 3, "convers": 3, "TO": 3, "inferenc": 3, "diffsvc": 3, "convert": 3, "model": 3, "enhanc": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.todo": 2, "sphinx": 57}, "alltitles": {"Fish Diffusion": [[0, "fish-diffusion"]], "Contents": [[0, "contents"]], "Quick Guide": [[0, "quick-guide"]], "Configuration": [[0, "configuration"], [1, "configuration"]], "Generation Quality": [[0, "generation-quality"]], "Common Questions": [[0, "common-questions"]], "Preprocessing": [[1, "preprocessing"]], "Dataset": [[1, "dataset"]], "Data Augmentation": [[1, "data-augmentation"]], "Appendix: Pitch Extractors": [[1, "appendix-pitch-extractors"]], "FAQ": [[2, "faq"]], "Why training is so slow on my device?": [[2, "why-training-is-so-slow-on-my-device"]], "Why the generated audio is blurry or weird?": [[2, "why-the-generated-audio-is-blurry-or-weird"]], "Why I see KeyError \u2018pytorch-lightning_version\u2019?": [[2, "why-i-see-keyerror-pytorch-lightning-version"]], "Why I see some error about missing keys when resuming training?": [[2, "why-i-see-some-error-about-missing-keys-when-resuming-training"]], "Quick FishSVC Guide": [[3, "quick-fishsvc-guide"]], "Preparing the environment": [[3, "preparing-the-environment"]], "Dataset Preparation (Single Speaker)": [[3, "dataset-preparation-single-speaker"]], "Dataset Preparation (Multi Speaker)": [[3, "dataset-preparation-multi-speaker"]], "Training time!": [[3, "training-time"]], "Inference": [[3, "inference"]], "DIFF SVC CONVERSION TO FISH SVC": [[3, "diff-svc-conversion-to-fish-svc"]], "Inferencing with a DiffSVC converted model": [[3, "inferencing-with-a-diffsvc-converted-model"]], "Enhancing Generation Quality": [[4, "enhancing-generation-quality"]]}, "indexentries": {}})