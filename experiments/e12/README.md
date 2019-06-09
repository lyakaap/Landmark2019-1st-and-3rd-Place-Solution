```
# train19, 1..32

$ mkdir -p data/input/delf_190505/train19
$ PYTHONPATH=lib:trunk/models_190505/research python code/exp12/delf_extract_features.py \
  --config_path code/exp12/params/delf_config.pbtxt \
  --list_images_path data/working/train19_exists.lst_blk1.lst \
  --output_dir data/input/delf_190505/train19/

# query

$ mkdir -p data/input/delf_190505/test
$ PYTHONPATH=lib:trunk/models_190505/research python code/exp12/delf_extract_features.py \
  --config_path code/exp12/params/delf_config.pbtxt \
  --list_images_path data/working/exp12/test_filepath.lst \
  --output_dir data/input/delf_190505/test/
```
