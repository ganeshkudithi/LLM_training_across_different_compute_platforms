# FINETUNING_TORCHTUNE

Torchtune is a PyTorch native library designed to simplify the process of authoring, fine-tuning, and experimenting with Large Language Models (LLMs).
Torchtune has used the LoRA for the fine-tuning the LLMs.


create the environment

        conda create --name torchtune python=3.10 --yes

activate the environment

        conda activate torchtune
Install the required torchtune libraries

         pip install torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/cu121
         pip install --pre torchtune --extra-index-url https://download.pytorch.org/whl/cu121
 use : 

         tune ls
 to see the listed RECIPE and CONFIG files of torchtune

                 RECIPE                                   CONFIG                                  
        full_finetune_single_device              llama2/7B_full_low_memory               
                                                 code_llama2/7B_full_low_memory          
                                                 llama3/8B_full_single_device            
                                                 llama3_1/8B_full_single_device          
                                                 mistral/7B_full_low_memory              
                                                 phi3/mini_full_low_memory               
        full_finetune_distributed                llama2/7B_full                          
                                                 llama2/13B_full                         
                                                 llama3/8B_full                          
                                                 llama3_1/8B_full                        
                                                 llama3/70B_full                         
                                                 llama3_1/70B_full                       
                                                 mistral/7B_full                         
                                                 gemma/2B_full                           
                                                 gemma/7B_full                           
                                                 phi3/mini_full                          
        lora_finetune_single_device              llama2/7B_lora_single_device            
                                                 llama2/7B_qlora_single_device           
                                                 code_llama2/7B_lora_single_device       
                                                 code_llama2/7B_qlora_single_device      
                                                 llama3/8B_lora_single_device            
                                                 llama3_1/8B_lora_single_device          
                                                 llama3/8B_qlora_single_device           
                                                 llama3_1/8B_qlora_single_device         
                                                 llama2/13B_qlora_single_device          
                                                 mistral/7B_lora_single_device           
                                                 mistral/7B_qlora_single_device          
                                                 gemma/2B_lora_single_device             
                                                 gemma/2B_qlora_single_device            
                                                 gemma/7B_lora_single_device             
                                                 gemma/7B_qlora_single_device            
                                                 phi3/mini_lora_single_device            
                                                 phi3/mini_qlora_single_device           
        lora_dpo_single_device                   llama2/7B_lora_dpo_single_device        
        lora_dpo_distributed                     llama2/7B_lora_dpo                      
        lora_finetune_distributed                llama2/7B_lora                          
                                                 llama2/13B_lora                         
                                                 llama2/70B_lora                         
                                                 llama3/70B_lora                         
                                                 llama3_1/70B_lora                       
                                                 llama3/8B_lora                          
                                                 llama3_1/8B_lora                        
                                                 mistral/7B_lora                         
                                                 gemma/2B_lora                           
                                                 gemma/7B_lora                           
                                                 phi3/mini_lora                          
        lora_finetune_fsdp2                      llama2/7B_lora                          
                                                 llama2/13B_lora                         
                                                 llama2/70B_lora                         
                                                 llama2/7B_qlora                         
                                                 llama2/70B_qlora                        
        generate                                 generation                              
        eleuther_eval                            eleuther_evaluation                     
        quantize                                 quantization                            
        qat_distributed                          llama2/7B_qat_full                      
                                                 llama3/8B_qat_full

Download the desired configuration using

        tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --hf-token 
Torchtune primarily uses YAML configuration files to specify the full set of parameters needed for fine-tuning. To pass custom parameter values, we can make a copy of the configuration file for the Llama-2-7b-hf model using the tune cp command:

        tune cp llama2/7B_full_low_memory my_llama3_1_custom_config.yaml
This my_llama3_1_custom_config.yaml file looks like:

               # Config for single device full finetuning in full_finetune_single_device.py
        # using a Llama2 7B model
        #
        # This config assumes that you've run the following command before launching
        # this run:
        #   tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --ignore-patterns "*.safetensors" --hf-token <HF_TOKEN>
        #
        # The default config uses an optimizer from bitsandbytes. If you do not have it installed,
        # you can install it with
        #   pip install bitsandbytes
        #
        # To launch on a single device, run the following command from root:
        #   tune run full_finetune_single_device --config llama2/7B_full_low_memory
        #
        # You can add specific overrides through the command line. For example
        # to override the checkpointer directory while launching training
        # you can run:
        #   tune run full_finetune_single_device --config llama2/7B_full_low_memory checkpointer.checkpoint_dir=<YOUR_CHECKPOINT_DIR>
        #
        # This config works only for training on single device.
        
        
        output_dir: /tmp/torchtune/llama2_7B/full_low_memory # /tmp may be deleted by your system. Change it to your preference.
        
        # Tokenizer
        tokenizer:
          _component_: torchtune.models.llama2.llama2_tokenizer
          path: /tmp/Llama-2-7b-hf/tokenizer.model
          max_seq_len: null
        
        # Dataset
        dataset:
          _component_: torchtune.datasets.alpaca_dataset
          packed: False  # True increases speed
        seed: null
        shuffle: True
        
        # Model Arguments
        model:
          _component_: torchtune.models.llama2.llama2_7b
        
        checkpointer:
          _component_: torchtune.training.FullModelHFCheckpointer
          checkpoint_dir: /tmp/Llama-2-7b-hf
          checkpoint_files: [
            pytorch_model-00001-of-00002.bin,
            pytorch_model-00002-of-00002.bin
          ]
          recipe_checkpoint: null
          output_dir: ${output_dir}
          model_type: LLAMA2
        resume_from_checkpoint: False
        
        # Fine-tuning arguments
        batch_size: 2
        epochs: 1
        optimizer:
          _component_: bitsandbytes.optim.PagedAdamW
          lr: 1e-5
        lr_scheduler:
          _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
          num_warmup_steps: 100
        optimizer_in_bwd: True  # True saves memory. Requires gradient_accumulation_steps=1
        loss:
          _component_: torchtune.modules.loss.CEWithChunkedOutputLoss
        max_steps_per_epoch: null
        gradient_accumulation_steps: 1  # Use to increase effective batch size
        clip_grad_norm: null
        compile: False  # torch.compile the model + loss, True increases speed + decreases memory
        
        # Training environment
        device: cuda
        
        # Memory management
        enable_activation_checkpointing: True  # True reduces memory
        enable_activation_offloading: True  # True reduces memory
        
        # Reduced precision
        dtype: bf16
        
        # Logging
        metric_logger:
          _component_: torchtune.training.metric_logging.DiskLogger
          log_dir: ${output_dir}/logs
        log_every_n_steps: 1
        log_peak_memory_stats: True
        
        
        # Profiler (disabled)
        profiler:
          _component_: torchtune.training.setup_torch_profiler
          enabled: False
        
          #Output directory of trace artifacts
          output_dir: ${output_dir}/profiling_outputs
        
          #`torch.profiler.ProfilerActivity` types to trace
          cpu: True
          cuda: True
        
          #trace options passed to `torch.profiler.profile`
          profile_memory: False
          with_stack: False
          record_shapes: True
          with_flops: False
        
          # `torch.profiler.schedule` options:
          # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
          wait_steps: 5
          warmup_steps: 3
          active_steps: 2
          num_cycles: 1

To customize with our own dataset, we will use one python script


        from torch.utils.data import Dataset
        
        class PlainTextDataset(Dataset):
            def __init__(self, tokenizer, file_path, max_length=2048):
                self.tokenizer = tokenizer
                self.max_length = max_length
        
                with open(file_path, "r", encoding="utf-8") as f:
                    self.lines = [line.strip() for line in f if len(line.strip()) > 0]
        
            def __len__(self):
                return len(self.lines)
        
            def __getitem__(self, idx):
                text = self.lines[idx]
        
                # Torchtune tokenizer
                ids = self.tokenizer.encode(text)
        
                # Truncate
                ids = ids[:self.max_length]
        
                # Pad
                if len(ids) < self.max_length:
                    ids = ids + [self.tokenizer.pad_id] * (self.max_length - len(ids))
        
                # Torchtune expects: tokens and labels
                return {
                    "tokens": ids,      # <--- required by torchtune padded_collate_sft
                    "labels": ids       # training is causal LM, so labels = tokens
                }

Now we can change the configuration file to use the above python script

         dataset:
              _component_: file_name.PlainTextDataset
              file_path : /home/path/to/the/dataset.txt
              max_length: 2048
              #packed: False  # True increases speed

To train the model with our dataset, run the config file

         tune run full_finetune_single_device --config my_llama3_1_custom_config.yaml
         
The model has been trained on our custom dataset; now we will perform inference on the fine-tuned LLM. 

         tune cp generation ./my_llama3_1_custom_generation_config.yaml

The generation file looks like


        # Config for running the InferenceRecipe in generate.py to generate output
        # from Llama2 7B model
        #
        # This config assumes that you've run the following command before launching
        # this run:
        #   tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --ignore-patterns "*.safetensors" --hf-token <HF_TOKEN>
        #
        # To launch, run the following command from root torchtune directory:
        #    tune run generate --config generation
        
        output_dir: ./ # Not needed
        
        # Model arguments
        model:
          _component_: torchtune.models.llama2.llama2_7b
        
        checkpointer:
          _component_: torchtune.training.FullModelHFCheckpointer
          checkpoint_dir: /tmp/Llama-2-7b-hf/
          checkpoint_files: [
            pytorch_model-00001-of-00002.bin,
            pytorch_model-00002-of-00002.bin,
          ]
          output_dir: ${output_dir}
          model_type: LLAMA2
        
        device: cuda
        dtype: bf16
        
        seed: 1234
        
        # Tokenizer arguments
        tokenizer:
          _component_: torchtune.models.llama2.llama2_tokenizer
          path: /tmp/Llama-2-7b-hf/tokenizer.model
          max_seq_len: null
          prompt_template: null
        
        # Generation arguments; defaults taken from gpt-fast
        prompt: 
          system: null
          user: "what is the meaning of domain knowledge?"
        max_new_tokens: 300
        temperature: 0.6 # 0.8 and 0.6 are popular values to try
        top_k: 300
        
        enable_kv_cache: True
        
        quantizer: null

we can perform the inference by using command

        tune run generate --config ./my_llama3_1_custom_generation_config.yaml
