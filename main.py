"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import os
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
    memory_per_sample = 2 * 1024 * 1024  # Estimate 2MB per sample
    max_batch_size = min(int(total_memory * 0.8 / memory_per_sample), config["batch_size"])
    config["batch_size"] = max_batch_size
    print(f"\nAuto-configured batch size: {max_batch_size}")

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
        config=config
    )
    
    # Create model and move to GPU
    model = get_model(model_config, device)
    model = model.to(device, memory_format=torch.channels_last)  # Use channels_last memory format
    
    # Create optimizer with gradient clipping
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer) if config.get("use_swa", False) else None

    if args.eval:
        model_load_path = Path(args.model_path)
        model.load_state_dict(torch.load(model_load_path))
        print("Model loaded from {}".format(model_load_path))
        print("Start evaluation...")
        
        # Evaluate with specified attack if configured
        if config.get("use_adversarial", False):
            evaluate_model_with_attack(
                model=model,
                eval_loader=eval_loader,
                device=device,
                database_path=Path(config["database_path"]),
                config=config
            )
            return
        
        produce_evaluation_file(eval_loader, model, device,
                               eval_score_path, eval_trial_path)
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"])
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
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
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
            optimizer.zero_grad(set_to_none=True)  # More efficient than .zero_grad()
            
            running_loss += loss.item() * batch_size
            num_correct += (batch_out.max(1)[1] == batch_y).sum().item()
            num_total += batch_size
            
            # Free up memory
            del batch_x, batch_y, batch_out
            
        scheduler.step()
        epoch_loss = running_loss / num_total
        epoch_acc = num_correct / num_total
        print("Training Loss: {:.4f}, Accuracy: {:.2f}%".format(
            epoch_loss, epoch_acc * 100))
        
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
    
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
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
                            batch_size=config["batch_size"],
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
                             batch_size=config["batch_size"],
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
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    
    # Ensure model is on GPU
    model = model.to(device)
    print(f"\nEvaluation GPU Memory before starting: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    
    # Track progress
    total_batches = len(data_loader)
    
    for batch_idx, (batch_x, utt_id) in enumerate(data_loader, 1):
        # Move data to GPU explicitly
        batch_x = batch_x.to(device, non_blocking=True)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{total_batches} ({(batch_idx/total_batches)*100:.1f}%)")
            print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).cpu().numpy().ravel()
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        # Periodic GPU memory clear
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    assert len(trial_lines) == len(fname_list) == len(score_list)
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
    main(parser.parse_args())
