"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
import platform
# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set environment variables for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable asynchronous CUDA operations

import torch
# Configure PyTorch memory management
torch.backends.cuda.max_split_size_mb = 128
torch.cuda.empty_cache()

# Configure CUDA settings for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable memory efficient attention if available
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(True)

# Enable flash attention if available
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)

# Enable memory profiling
if torch.cuda.is_available():
    torch.cuda.memory._record_memory_history()
    torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Reserve 95% of GPU memory

import argparse
import json
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from adversarial_attack import evaluate_model_with_attack

warnings.filterwarnings("ignore", category=FutureWarning)

def print_gpu_info():
    """Print detailed GPU information"""
    print("\nGPU Information:")
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    
    # Get memory information
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\nGPU Capabilities:")
    print(f"Total GPU Memory: {gpu_props.total_memory/1024**2:.1f}MB")
    print(f"CUDA Capability Major/Minor version: {gpu_props.major}.{gpu_props.minor}")
    print(f"Multi-Processor Count: {gpu_props.multi_processor_count}")
    
    # Current memory usage
    print(f"\nCurrent Memory Status:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

def setup_gpu():
    """Setup GPU and memory management"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    # Clear any existing cache
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Set device
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    
    return device

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # Setup GPU first
    device = setup_gpu()
    print_gpu_info()
    
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    if "fgsm_epsilon" not in config:
        config["fgsm_epsilon"] = 0.01  # Default FGSM attack strength

    # Configure batch size based on available GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # More conservative memory estimation - account for model size and gradients
    memory_per_sample = 4 * 1024 * 1024  # Estimate 4MB per sample to be safe
    
    # Set very conservative batch sizes
    if args.eval:
        # For evaluation, use a very small batch size
        config["batch_size"] = 2  # Fixed small batch size for evaluation
        print(f"\nForced evaluation batch size: 2 (for GTX 1650 Ti memory constraints)")
    else:
        # For training, use auto-configured size
        suggested_batch_size = max(1, min(16, int((total_memory * 0.6) / memory_per_sample)))
        config["batch_size"] = min(suggested_batch_size, config.get("batch_size", 16))
        print(f"\nUsing training batch size: {config['batch_size']} (suggested: {suggested_batch_size})")
    
    # Enable memory efficient options
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Platform-specific memory management
    is_windows = platform.system() == 'Windows'
    
    # More aggressive memory management
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        # Set memory fraction to 85% to leave room for system
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # Configure PyTorch memory allocator
        if is_windows:
            # Windows-specific memory settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
            # Disable memory profiling on Windows
            os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        else:
            # Linux/Unix memory settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'
            # Enable memory stats tracking only on Linux
            if hasattr(torch.cuda.memory, '_record_memory_history'):
                torch.cuda.memory._record_memory_history(max_entries=10000)
    
    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create data loaders with optimized settings
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path=Path(config["database_path"]),
        seed=args.seed,
        config=config  # Remove is_eval since we've already set the batch size
    )
    
    # Create model and move to GPU
    model = get_model(model_config, device)
    model = model.to(device, memory_format=torch.channels_last)  # Use channels_last memory format
    
    # Configure optimizer settings
    optim_config["steps_per_epoch"] = len(trn_loader)
    optim_config["epochs"] = config["num_epochs"]  # Ensure epochs is set
    
    # Set scheduler-specific defaults if not present
    if optim_config["scheduler"] == "cosine" and "lr_min" not in optim_config:
        optim_config["lr_min"] = optim_config["base_lr"] * 0.01  # Default to 1% of base_lr
    
    if optim_config["scheduler"] == "sgdr" and "T0" not in optim_config:
        optim_config["T0"] = 10  # Default restart period
        optim_config["Tmult"] = 2  # Default period multiplier
    
    # Create optimizer with gradient clipping
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer) if config.get("use_swa", False) else None

    if args.eval:
        # Determine model path from args or config
        if args.eval_model_weights:
            model_load_path = Path(args.eval_model_weights)
        elif "model_path" in config:
            model_load_path = Path(config["model_path"])
        else:
            raise ValueError("No model path provided. Use --eval_model_weights or specify model_path in config.")
            
        model.load_state_dict(torch.load(model_load_path))
        print("Model loaded from {}".format(model_load_path))
        print("Start evaluation...")
        
        # Check if attack type is specified via argument or config
        attack_type = args.attack or config.get("attack_type", None)
        
        if attack_type:
            print(f"\n{'='*50}")
            print(f"Running evaluation with {attack_type.upper()} attack")
            print(f"{'='*50}")
            
            # Prepare attack parameters
            attack_params = {}
            
            # Create attack-specific output path to avoid overwriting normal eval results
            attack_output_dir = model_tag / f"{attack_type}_attack_results"
            os.makedirs(attack_output_dir, exist_ok=True)
            
            # Create attack-specific score path
            attack_score_path = attack_output_dir / f"{attack_type}_scores.txt"
            config["attack_score_path"] = str(attack_score_path)
            
            # Use epsilon from command line if provided, otherwise from config or default
            if attack_type == "fgsm":
                if args.epsilon is not None:
                    attack_params["epsilon"] = args.epsilon
                    print(f"Using command line epsilon: {args.epsilon}")
                elif "fgsm_epsilon" in config:
                    attack_params["epsilon"] = config["fgsm_epsilon"]
                    print(f"Using config epsilon: {config['fgsm_epsilon']}")
                else:
                    print("Using default epsilon: 0.05")
            
            elif attack_type == "pgd":
                if args.epsilon is not None:
                    attack_params["epsilon"] = args.epsilon
                    print(f"Using command line epsilon: {args.epsilon}")
                elif "pgd_epsilon" in config:
                    attack_params["epsilon"] = config["pgd_epsilon"]
                    print(f"Using config epsilon: {config['pgd_epsilon']}")
                else:
                    print("Using default epsilon: 0.05")
                
                # Use other PGD parameters from config if available
                if "pgd_alpha" in config:
                    attack_params["alpha"] = config["pgd_alpha"]
                    print(f"Using config alpha: {config['pgd_alpha']}")
                if "pgd_num_iter" in config:
                    attack_params["num_iter"] = config["pgd_num_iter"]
                    print(f"Using config num_iter: {config['pgd_num_iter']}")
            
            elif attack_type == "deepfool":
                if "deepfool_max_iter" in config:
                    attack_params["max_iter"] = config["deepfool_max_iter"]
                    print(f"Using config max_iter: {config['deepfool_max_iter']}")
                else:
                    print("Using default max_iter: 10")
            
            print(f"{'='*50}\n")
            
            # Run evaluation with attack
            eval_eer, eval_tdcf = evaluate_model_with_attack(
                model=model,
                eval_loader=eval_loader,
                device=device,
                database_path=Path(config["database_path"]),
                config=config,
                attack_type=attack_type,
                attack_params=attack_params
            )
            
            # Save attack-specific results
            result_output = attack_output_dir / f"{attack_type}_metrics.txt"
            with open(result_output, "w") as f:
                f.write(f"Attack Type: {attack_type.upper()}\n")
                f.write(f"Parameters:\n")
                for param, value in attack_params.items():
                    f.write(f"- {param}: {value}\n")
                f.write(f"\nResults:\n")
                f.write(f"EER: {eval_eer:.3f}%\n")
                f.write(f"min t-DCF: {eval_tdcf:.5f}\n")
            
            # Print a clear summary of the attack results
            print("\n" + "="*50)
            print(f"ADVERSARIAL ATTACK RESULTS: {attack_type.upper()}")
            print("="*50)
            print(f"EER: {eval_eer:.3f}%")
            print(f"min t-DCF: {eval_tdcf:.5f}")
            print("="*50)
            
            print(f"\nAttack results saved to: {attack_output_dir}")
        else:
            # Get max_batches from config or use default
            max_eval_batches = config.get("max_eval_batches", 1000)
            print(f"\nRunning standard evaluation (no attack)...")
            print(f"Using max evaluation batches: {max_eval_batches}")
            
            produce_evaluation_file(
                eval_loader, 
                model, 
                device,
                eval_score_path, 
                eval_trial_path,
                max_batches=max_eval_batches
            )
            
            # Calculate EER and t-DCF with output file
            result_output = model_tag / "t-DCF_EER_eval.txt"
            eval_eer, eval_tdcf = calculate_tDCF_EER(
                cm_scores_file=eval_score_path,
                asv_score_file=database_path / config["asv_score_path"],
                output_file=result_output
            )
            print("EVAL - EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
        return
    
    # Training loop
    print("Start training...")
    num_epochs = config["num_epochs"]
    best_dev_eer = 1.
    best_dev_tdcf = 1.
    best_eval_eer = 1.
    best_eval_tdcf = 1.
    n_swa_update = 0
    
    for epoch in range(num_epochs):
        print("\nEpoch {}/{}:".format(epoch + 1, num_epochs))
        
        # Training phase
        model.train()
        running_loss = 0.
        num_total = 0.
        num_correct = 0.
        
        # Use tqdm for progress tracking
        for batch_x, batch_y in tqdm(trn_loader, desc=f"Training Epoch {epoch+1}"):
            batch_size = batch_x.size(0)
            
            # Move data to GPU asynchronously
            batch_x = batch_x.to(device, non_blocking=True, memory_format=torch.channels_last)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Clear gradients and cache
            optimizer.zero_grad(set_to_none=True)
            
            try:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    batch_out, _ = model(batch_x)
                    loss = criterion(batch_out, batch_y)
                    
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * batch_size
                num_correct += (batch_out.max(1)[1] == batch_y).sum().item()
                num_total += batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM error in batch. Current memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            finally:
                # Clean up memory
                del batch_x, batch_y, batch_out
                if batch_idx % 10 == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
            
        scheduler.step()
        epoch_loss = running_loss / num_total
        epoch_acc = num_correct / num_total
        print("Training Loss: {:.4f}, Accuracy: {:.2f}%".format(
            epoch_loss, epoch_acc * 100))
        
        # Memory status after training
        print(f"GPU Memory after epoch: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("train_acc", epoch_acc, epoch)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            produce_evaluation_file(dev_loader, model, device,
                                   dev_score_path, dev_trial_path)
            dev_eer, dev_tdcf = calculate_tDCF_EER(
                cm_scores_file=dev_score_path,
                asv_score_file=database_path / config["asv_score_path"])
            print("DEV - EER: {:.3f}, min t-DCF: {:.5f}".format(dev_eer, dev_tdcf))
            
            # Update best scores
            if dev_eer < best_dev_eer:
                print("Best DEV EER: {:.3f} -> {:.3f}".format(best_dev_eer, dev_eer))
                best_dev_eer = dev_eer
                torch.save(model.state_dict(),
                           model_save_path / "best_eer.pth")
            
            if dev_tdcf < best_dev_tdcf:
                print("Best DEV min t-DCF: {:.5f} -> {:.5f}".format(
                    best_dev_tdcf, dev_tdcf))
                best_dev_tdcf = dev_tdcf
                torch.save(model.state_dict(),
                           model_save_path / "best_tdcf.pth")
            
            # SWA update if configured
            if optimizer_swa is not None and epoch >= config.get("swa_start", 0):
                optimizer_swa.update_swa()
                n_swa_update += 1
            
            writer.add_scalar("dev_eer", dev_eer, epoch)
            writer.add_scalar("dev_tdcf", dev_tdcf, epoch)
            
            # Memory cleanup
            torch.cuda.empty_cache()
    
    print("\nTraining completed. Starting final evaluation...")
    
    # Final SWA update if used
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        produce_evaluation_file(eval_loader, model, device,
                               eval_score_path, eval_trial_path)
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"])
        
        print("\nFinal Results:")
        print("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
        
        with open(model_tag / "final_metrics.txt", "w") as f:
            f.write("EER: {:.3f}, min t-DCF: {:.5f}\n".format(eval_eer, eval_tdcf))
            f.write("Best DEV EER: {:.3f}, Best DEV min t-DCF: {:.5f}\n".format(
                best_dev_eer, best_dev_tdcf))
    
    writer.close()
    print("\nExperiment completed successfully!")


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # Configure number of workers based on CPU cores
    num_workers = min(4, os.cpu_count())
    print(f"\nUsing {num_workers} workers for data loading")
    
    # Use different batch sizes for training and evaluation
    batch_size = config["batch_size"]
    
    trn_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=num_workers,
                            worker_init_fn=seed_worker,
                            generator=gen,
                            persistent_workers=True,
                            prefetch_factor=2)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            persistent_workers=True,
                            prefetch_factor=2)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             persistent_workers=True,
                             prefetch_factor=2)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    max_batches: int = 1000) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    
    # Ensure model is on GPU
    model = model.to(device)
    
    # Platform-specific memory monitoring
    is_windows = platform.system() == 'Windows'
    if is_windows:
        print("\nEvaluation started - Windows platform detected")
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    else:
        print("\nEvaluation started - Unix/Linux platform detected")
        print(f"Initial GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Initial GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
    
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    
    # Track progress
    total_batches = min(len(data_loader), max_batches)
    print(f"\nProcessing {total_batches} batches (limited from {len(data_loader)} total batches)")
    
    for batch_idx, (batch_x, utt_id) in enumerate(data_loader, 1):
        if batch_idx > max_batches:
            print(f"\nReached max batches limit ({max_batches})")
            break
            
        # Move data to GPU explicitly
        batch_x = batch_x.to(device, non_blocking=True)
        
        # Print progress and memory stats
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{total_batches} ({(batch_idx/total_batches)*100:.1f}%)")
            print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        try:
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).cpu().numpy().ravel()
            
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM error in batch {batch_idx}. Attempting recovery...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        finally:
            # Clean up memory
            del batch_x, batch_out
            if batch_idx % 50 == 0:  # Periodic cleanup
                torch.cuda.empty_cache()

    # Final memory status
    if is_windows:
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    else:
        print(f"\nFinal GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Final GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
    
    print(f"\nProcessed {len(fname_list)} samples in {batch_idx} batches")
    
    # Write all available scores
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--attack",
                        type=str,
                        default=None,
                        choices=["fgsm", "pgd", "deepfool"],
                        help="adversarial attack type to use for evaluation (fgsm, pgd, or deepfool)")
    parser.add_argument("--epsilon",
                        type=float,
                        default=None,
                        help="epsilon value for adversarial attacks (attack strength)")
    main(parser.parse_args())
