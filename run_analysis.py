import os
import sys

sys.path.insert(0, os.getcwd())

model_name = 'gpt2'

print("starting import")
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display

curdir = os.getcwd()

os.makedirs("content/tasks/configs/", exist_ok=True)
os.makedirs("content/tasks/data/semeval", exist_ok=True)

py_io.write_json({
  "task": "semeval",
  "paths": {
    "train": f"{curdir}/content/tasks/data/semeval/train.all.json",
    "val": f"{curdir}/content/tasks/data/semeval/test.json",
  },
  "name": "semeval"
}, f"{curdir}/content/tasks/configs/semeval_config.json")

print("donwloading model", flush=True)

export_model.export_model(
    hf_pretrained_model_name_or_path=f"{model_name}",
    output_base_path=f"{curdir}/content/current/models/{model_name}",
)

task_name = "semeval"

print("Starting tokenizer", flush=True)

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"{curdir}/content/tasks/configs/{task_name}_config.json",
    hf_pretrained_model_name_or_path=f"{model_name}",
    output_dir=f"{curdir}/content/cache/{task_name}",
    phases=["train", "val"],
))

row = caching.ChunkedFilesDataCache(f"{curdir}/content/cache/semeval/train").load_chunk(0)[0]["data_row"]
print(row.input_ids)
print(row.tokens)
print(row.tokens[row.spans[0][0]: row.spans[0][1]+1])
print(row.tokens[row.spans[1][0]: row.spans[1][1]+1])

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"{curdir}/content/tasks/configs",
    task_cache_base_path=f"{curdir}/content/cache",
    train_task_name_list=["semeval"],
    val_task_name_list=["semeval"],
    train_batch_size=8,
    eval_batch_size=16,
    epochs=1,
    num_gpus=0,
).create_config()
os.makedirs(f"{curdir}/content/run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, f"{curdir}/content/run_configs/semeval_run_config.json")
display.show_json(jiant_run_config)

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path=f"{curdir}/content/run_configs/semeval_run_config.json",
    output_dir=f"{curdir}/content/runs/semeval",
    hf_pretrained_model_name_or_path=f"{model_name}",
    model_path=f"{curdir}/content/current/models/{model_name}/model/model.p",
    model_config_path=f"{curdir}/content/current/models/{model_name}/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=500,
    do_train=True,
    do_val=True,
    do_save=True,
    force_overwrite=True,
)

main_runscript.run_loop(run_args)
