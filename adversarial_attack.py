import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from evaluation import calculate_tDCF_EER
from data_utils import genSpoof_list
import os
import gc
import time
from torch.amp import autocast, GradScaler
from torch.cuda.amp import autocast as cuda_autocast

def verify_gpu_memory():
    """Verify GPU memory availability and configuration"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    total_memory = gpu_properties.total_memory / (1024**3)  # Convert to GB
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Clear cache
    torch.cuda.empty_cache()
    
    print("\nGPU Memory Configuration:")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Current Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"Current Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print(f"CUDA Capability Major/Minor version: {gpu_properties.major}.{gpu_properties.minor}")
    
    # Try to allocate a test tensor to verify memory access
    try:
        # Try to allocate 10% of total memory
        test_size = int(0.1 * total_memory * (1024**3) / 4)  # Convert GB to float32 elements
        test_tensor = torch.empty(test_size, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        print("Successfully verified GPU memory allocation")
        
        # Enable memory stats tracking
        torch.cuda.memory.set_per_process_memory_fraction(0.9)  # Reserve 90% of GPU memory
        torch.cuda.memory._record_memory_history(max_entries=10000)
        
    except RuntimeError as e:
        print(f"WARNING: Could not allocate test tensor: {str(e)}")
        raise

@autocast(device_type='cuda')
def fgsm_attack(model, x, epsilon, device='cuda'):
    """
    Fast Gradient Sign Method attack.
    """
    x = x.clone().detach().to(device, dtype=torch.float16)
    x = torch.nn.Parameter(x, requires_grad=True)
    
    with autocast(device_type='cuda'):
        _, outputs = model(x)
        # Target is opposite of prediction
        _, predicted = torch.max(outputs.data, 1)
        target = 1 - predicted  # Flip between 0 and 1
        loss = F.cross_entropy(outputs, target)
    
    loss.backward()
    
    with torch.no_grad():
        perturbed_x = x + epsilon * x.grad.sign()
        perturbed_x = torch.clamp(perturbed_x, x.min(), x.max())
    
    return perturbed_x.detach()

@autocast(device_type='cuda')
def pgd_attack(model, x, epsilon, alpha=0.01, num_iter=10, device='cuda'):
    """
    Projected Gradient Descent attack.
    """
    x = x.clone().detach().to(device, dtype=torch.float16)
    x_orig = x.clone()
    
    # Initialize with random perturbation
    delta = torch.zeros_like(x, device=device, dtype=torch.float16)
    delta = torch.nn.Parameter(delta, requires_grad=True)
    
    for i in range(num_iter):
        # Forward pass
        with autocast(device_type='cuda'):
            _, outputs = model(x_orig + delta)
            # Target is opposite of prediction
            _, predicted = torch.max(outputs.data, 1)
            target = 1 - predicted  # Flip between 0 and 1
            loss = F.cross_entropy(outputs, target)
        
        # Backward pass
        loss.backward()
        
        with torch.no_grad():
            grad = delta.grad.detach()
            delta.data = delta.data + alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(x_orig + delta.data, x_orig.min(), x_orig.max()) - x_orig
        
        delta.grad.zero_()
    
    return x_orig + delta.detach()

@autocast(device_type='cuda')
def deepfool_attack(model, x, num_classes=2, max_iter=10, device='cuda'):
    """
    DeepFool attack (lightweight version).
    """
    x = x.clone().detach().to(device, dtype=torch.float16)
    perturbed_x = x.clone()
    perturbed_x = torch.nn.Parameter(perturbed_x, requires_grad=True)
    batch_size = x.size(0)
    
    # Track success rate properly
    success_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with autocast(device_type='cuda'):
        _, outputs = model(x)
        f_orig = outputs.detach()
        f_orig_label = f_orig.max(1)[1]
        
        for i in range(max_iter):
            _, f = model(perturbed_x)
            
            # Check which samples were successfully attacked
            current_preds = f.argmax(1)
            newly_successful = (current_preds != f_orig_label) & (~success_mask)
            success_mask = success_mask | newly_successful
            
            if success_mask.all():
                print(f"All samples successfully perturbed at iteration {i}")
                break
            
            # Initialize perturbation
            w_total = torch.zeros_like(perturbed_x)
            f_total = torch.zeros(batch_size, dtype=torch.float16, device=device)
            
            # Get gradients for each sample in the batch
            for b in range(batch_size):
                if success_mask[b]:
                    continue  # Skip already successful samples
                
                try:
                    # Get gradient of original class
                    grad_orig = torch.autograd.grad(f[b, f_orig_label[b]], perturbed_x, retain_graph=True)[0][b]
                    
                    # Get gradient of target class (using the other class in binary classification)
                    target_label = 1 - f_orig_label[b]  # Flip between 0 and 1
                    grad_target = torch.autograd.grad(f[b, target_label], perturbed_x, retain_graph=True)[0][b]
                    
                    # Calculate perturbation for this sample
                    w = grad_target - grad_orig
                    f_diff = (f[b, target_label] - f[b, f_orig_label[b]]).abs()
                    
                    # Normalize perturbation
                    w_norm = w.view(-1).norm(p=2)
                    if w_norm > 1e-6:  # Avoid division by zero
                        r_i = (f_diff / (w_norm + 1e-8))
                        w_total[b] = r_i * w.sign()
                        f_total[b] = f_diff
                except RuntimeError as e:
                    print(f"Error processing sample {b} in batch: {str(e)}")
                    continue
            
            # Apply perturbation
            with torch.no_grad():
                perturbed_x.data = torch.clamp(perturbed_x.data + w_total, x.min(), x.max())
            
            if i == max_iter - 1:
                print(f"Reached max iterations. Success rate: {success_mask.float().mean()*100:.2f}%")
                # If max iterations reached without success, return original for unsuccessful samples
                perturbed_x.data[~success_mask] = x[~success_mask]
    
    success_rate = success_mask.float().mean().item()
    print(f"Attack complete. Success rate: {success_rate*100:.2f}%")
    return perturbed_x.detach()

def process_batch_parallel(model, batch_x, attack_fn, attack_params, device, chunk_size=8):
    """Process a batch in parallel chunks with specified attack"""
    num_chunks = (batch_x.size(0) + chunk_size - 1) // chunk_size
    chunks = torch.chunk(batch_x, num_chunks)
    
    with autocast(device_type='cuda'):
        results = [attack_fn(model, chunk.to(device), **attack_params) for chunk in chunks]
    
    return torch.cat(results)

def evaluate_model_with_attack(model, eval_loader, device, database_path, config, attack_type=None, attack_params=None):
    """
    Evaluate model performance with specified adversarial attack.
    """
    # Get attack type from config if not specified
    if attack_type is None:
        attack_type = config.get("attack_type", "fgsm")  # Default to FGSM for backward compatibility
    
    # Get max batches from config
    max_batches = config.get("max_eval_batches", 500)  # Default to 500 batches
    
    # Define attack functions and their default parameters
    attack_functions = {
        'fgsm': (fgsm_attack, {'epsilon': config.get("fgsm_epsilon", 0.01)}),
        'pgd': (pgd_attack, {
            'epsilon': config.get("pgd_epsilon", 0.01),
            'alpha': config.get("pgd_alpha", 0.001),
            'num_iter': config.get("pgd_num_iter", 5)
        }),
        'deepfool': (deepfool_attack, {
            'num_classes': 2,
            'max_iter': config.get("deepfool_max_iter", 5)
        })
    }
    
    if attack_type not in attack_functions:
        raise ValueError(f"Attack type '{attack_type}' not supported. Choose from: {list(attack_functions.keys())}")
    
    attack_fn, default_params = attack_functions[attack_type]
    if attack_params:
        default_params.update(attack_params)
    
    print("\n" + "="*50)
    print(f"Starting {attack_type.upper()} Attack")
    print("="*50)
    print(f"Attack Parameters:")
    for param, value in default_params.items():
        print(f"- {param}: {value}")
    print(f"- Device: {device}")
    print(f"- Batch Size: {eval_loader.batch_size}")
    print(f"- Mixed Precision: Enabled (cuda)")
    print(f"- Max Batches: {max_batches}")
    
    # Move model to GPU and set to eval mode
    model = model.to(device)
    if hasattr(model, 'half'):
        model = model.half()  # Use FP16 if supported
    model.eval()
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    # Verify GPU memory and configuration
    verify_gpu_memory()
    
    # Create temporary directories for scores
    tmp_dir = Path("tmp_scores")
    os.makedirs(tmp_dir, exist_ok=True)
    adv_score_path = tmp_dir / f"{attack_type}_scores.txt"
    
    # Prepare trial file path
    track = config["track"]
    prefix_2019 = f"ASVspoof2019.{track}"
    trial_path = (database_path / 
                 f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.eval.trl.txt")
    
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    
    adv_scores = []
    fname_list = []
    total_batches = min(len(eval_loader), max_batches)
    start_time = time.time()
    last_update_time = start_time
    
    # Initialize CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print("\nStarting adversarial example generation...")
    print(f"Total batches to process: {total_batches}")
    
    try:
        # Pre-allocate GPU memory for tensors
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_event.record()
        
        for batch_idx, (batch_x, utt_id) in enumerate(eval_loader):
            if batch_idx >= max_batches:
                print(f"\nReached max batches limit ({max_batches})")
                break
            
            current_time = time.time()
            if batch_idx % 10 == 0 or (current_time - last_update_time) >= 30:
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_time = current_time - start_time
                progress = (batch_idx + 1) / total_batches
                eta = elapsed_time / (progress) - elapsed_time if progress > 0 else 0
                
                # Get current GPU memory usage
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                
                print("\nProgress Update:")
                print(f"Batch: {batch_idx}/{total_batches} ({progress*100:.1f}%)")
                print(f"GPU Memory Allocated: {allocated:.2f} GB")
                print(f"GPU Memory Reserved: {reserved:.2f} GB")
                print(f"Time Elapsed: {elapsed_time/60:.1f} minutes")
                print(f"ETA: {eta/60:.1f} minutes")
                print(f"Samples Processed: {len(adv_scores)}")
                print(f"Processing Speed: {len(adv_scores)/(elapsed_time/60):.1f} samples/minute")
                
                last_update_time = current_time
                
                # Periodic memory cleanup
                if batch_idx > 0 and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            try:
                with autocast(device_type='cuda'):
                    # Move data to GPU
                    batch_x = batch_x.to(device, non_blocking=True)
                    
                    # Generate adversarial examples
                    adv_x = process_batch_parallel(model, batch_x, attack_fn, default_params, device)
                    
                    # Compute scores
                    with torch.no_grad():
                        _, adv_output = model(adv_x)
                        adv_score = adv_output[:, 1].cpu().numpy()
                
                adv_scores.extend(adv_score.tolist())
                fname_list.extend(utt_id)
                
                del adv_x, adv_output
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"\nError processing batch {batch_idx}: {str(e)}")
                continue
    
    except RuntimeError as e:
        print(f"\nError during processing: {str(e)}")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        raise
    
    end_event.record()
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    samples_per_second = len(adv_scores) / total_time
    
    print(f"\n{attack_type.upper()} Attack completed!")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Average time per batch: {total_time/total_batches:.2f} seconds")
    print(f"Processing speed: {samples_per_second:.1f} samples/second")
    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    print("\nWriting scores to file...")
    total_samples = len(fname_list)
    with open(adv_score_path, "w") as fh:
        if total_samples == 0:
            print("WARNING: No scores to write - attack may have failed completely")
            print("This might be due to:")
            print("1. All attacks failed to generate valid perturbations")
            print("2. Memory issues during attack execution")
            print("3. Numerical instabilities in gradient computation")
            # Write a dummy score to prevent empty file errors
            fh.write("dummy_utt - spoof -1.0\n")
        else:
            print(f"Processing {total_samples} samples...")
            score_stats = {
                "min": float('inf'),
                "max": float('-inf'),
                "total": 0.0,
                "count": 0
            }
            
            for fn, sco in zip(fname_list, adv_scores):
                if not np.isnan(sco) and not np.isinf(sco):
                    score_stats["min"] = min(score_stats["min"], sco)
                    score_stats["max"] = max(score_stats["max"], sco)
                    score_stats["total"] += sco
                    score_stats["count"] += 1
                    # Format: <utt_id> <spoof/bonafide> <score>
                    fh.write(f"{fn} spoof {sco:.6f}\n")
            
            if score_stats["count"] > 0:
                print("\nScore Statistics:")
                print(f"Min score: {score_stats['min']:.6f}")
                print(f"Max score: {score_stats['max']:.6f}")
                print(f"Mean score: {score_stats['total']/score_stats['count']:.6f}")
                print(f"Valid scores: {score_stats['count']}/{total_samples}")
            else:
                print("WARNING: No valid scores were generated!")
    
    print(f"Wrote {total_samples} scores to {adv_score_path}")
    
    print("\nCalculating EER and t-DCF...")
    try:
        if len(fname_list) == 0:
            print("WARNING: No valid scores available - attack failed completely")
            return 0.0, 0.0, 1.0, 1.0  # Return worst case metrics
            
        adv_eer, adv_tdcf = calculate_tDCF_EER(
            cm_scores_file=str(adv_score_path),
            asv_score_file=str(database_path / config["asv_score_path"]),
            output_file=str(tmp_dir / f"{attack_type}_result.txt"),
            printout=False
        )
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        print("Checking score file format...")
        with open(adv_score_path, "r") as f:
            print("First few lines of score file:")
            print("".join(f.readlines()[:5]))
        # Return worst case metrics on error
        return 0.0, 0.0, 1.0, 1.0
    
    print(f"\n{attack_type.upper()} Attack Results:")
    print(f"EER: {adv_eer*100:.2f}%")
    print(f"t-DCF: {adv_tdcf:.4f}")
    
    return 0.0, 0.0, adv_eer, adv_tdcf

# For backward compatibility
def evaluate_model_with_fgsm(model, eval_loader, device, database_path, config, epsilon=0.01):
    """Wrapper for backward compatibility"""
    return evaluate_model_with_attack(
        model, eval_loader, device, database_path, config,
        attack_type='fgsm',
        attack_params={'epsilon': epsilon}
    )

def test_model_robustness(model, test_loader, epsilon, device='cuda'):
    """
    Test the model's robustness against FGSM attacks.
    
    Args:
        model: The model to test
        test_loader: DataLoader containing test data
        epsilon: Attack strength
        device: Device to run the test on
        
    Returns:
        float: Accuracy on adversarial examples
    """
    correct = 0
    total = 0
    model.eval()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        perturbed_images = fgsm_attack(model, images, epsilon, device)
        
        # Test on adversarial examples
        with torch.no_grad():
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def visualize_adversarial_examples(original_image, perturbed_image, epsilon):
    """
    Visualize original image, perturbation, and adversarial example.
    
    Args:
        original_image: Original input image
        perturbed_image: Adversarially perturbed image
        epsilon: Attack strength used
    """
    import matplotlib.pyplot as plt
    
    # Convert tensors to numpy arrays
    original_image = original_image.cpu().detach().numpy()
    perturbed_image = perturbed_image.cpu().detach().numpy()
    
    # Calculate perturbation
    perturbation = perturbed_image - original_image
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(np.transpose(original_image[0], (1, 2, 0)))
    plt.axis('off')
    
    # Plot perturbation
    plt.subplot(132)
    plt.title(f'Perturbation (ε={epsilon})')
    plt.imshow(np.transpose(perturbation[0], (1, 2, 0)))
    plt.axis('off')
    
    # Plot adversarial image
    plt.subplot(133)
    plt.title('Adversarial Example')
    plt.imshow(np.transpose(perturbed_image[0], (1, 2, 0)))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def write_score_file(save_path, keys, scores):
    """Helper function to write scores to file in the required format."""
    with open(save_path, 'w') as f:
        for key, score in zip(keys, scores):
            f.write(f'ASVspoof2019.LA.cm.eval.flac {key} - {score:.6f}\n')

# Example usage:
if __name__ == "__main__":
    # Example parameters
    epsilon = 0.01  # Attack strength
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your model here
    # model = YourModel().to(device)
    # model.load_state_dict(torch.load('path_to_model_weights'))
    
    # Test robustness
    # test_loader = torch.utils.data.DataLoader(...)
    # accuracy = test_model_robustness(model, test_loader, epsilon, device)
    # print(f"Accuracy under FGSM attack (ε={epsilon}): {accuracy:.2f}%") 