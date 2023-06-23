# TokenDrop + BucketSampler: Towards Efficient Padding-free Fine-tuning of Large Language Models

## Dependencies
Huggingface transformers 4.25.0.dev0, and all of its associated dependencies

PyTorch 1.12.1

Tested on RTX 2080 Ti GPU with CUDA Version 11.4

## Commands
+ To fine-tune with RandomSampler, navigate to /transformers/examples/pytorch/text-classification and use `python run_glue.py --model_name_or_path <google/electra-base-discriminator or roberta-base> --task_name <name of GLUE task> --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate <3e-5, 4e-5 or 5e-5> --num_train_epochs 3 --output_dir <path to output dir> `.
+ To fine-tune with TokenDrop + BucketSampler, use `python tokendrop+bucketsampler.py --model_name_or_path <google/electra-base-discriminator or roberta-base> --task_name <name of GLUE task> --per_device_eval_batch_size 64 --learning_rate <3e-5, 4e-5 or 5e-5> --num_train_epochs 3  --output_dir <path to output dir> --bucket_sampler True --tokendrop True`. 

