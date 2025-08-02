import copy
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# Modern Dirichlet-based imports
from sampling_dir import (cifar10_iid, cifar10_dirichlet, cifar100_iid, cifar100_dirichlet,
                     mnist_iid, mnist_dirichlet, analyze_distribution, plot_distribution)

# Legacy imports (commented out - use modern Dirichlet methods instead)
# from sampling import mnist_noniid, mnist_noniid_unequal, cifar_noniid


def get_dataset(args):
    """ 
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    
    Now uses modern Dirichlet distribution for non-IID data instead of legacy shard methods.
    """

    if args.dataset == 'cifar' or args.dataset == 'cifar100':
        if args.dataset == 'cifar':
            dataset_class = datasets.CIFAR10
            data_dir = './data/cifar/'  # Updated path (removed '../')
            args.num_classes = 10  # Auto-update for CIFAR-10
        else:
            dataset_class = datasets.CIFAR100
            data_dir = './data/cifar100/'  # Updated path (removed '../')
            args.num_classes = 100  # Auto-update for CIFAR-100

        args.num_channels = 3  # Auto-update for CIFAR datasets

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = dataset_class(data_dir, train=True, download=True,
                                     transform=apply_transform)
        test_dataset = dataset_class(data_dir, train=False, download=True,
                                    transform=apply_transform)

        # Sample training data amongst users using modern Dirichlet approach
        if args.iid:
            print(f"Creating IID data distribution for {args.dataset.upper()}...")
            if args.dataset == 'cifar':
                user_groups = cifar10_iid(train_dataset, args.num_users)
            else:
                user_groups = cifar100_iid(train_dataset, args.num_users)
        else:
            # Modern Dirichlet-based Non-IID distribution
            alpha = getattr(args, 'alpha', 0.5)  # Default alpha if not specified
            min_samples = getattr(args, 'min_samples', 50)  # Default min samples
            
            print(f"Creating Non-IID data distribution for {args.dataset.upper()} with alpha={alpha}...")
            
            if args.dataset == 'cifar':
                user_groups = cifar10_dirichlet(train_dataset, args.num_users, 
                                              alpha=alpha, min_samples_per_user=min_samples)
            else:
                user_groups = cifar100_dirichlet(train_dataset, args.num_users, 
                                               alpha=alpha, min_samples_per_user=min_samples)
            


    elif args.dataset == 'mnist' or args.dataset == 'fmnist':  # Fixed comparison
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'  # Updated path (removed '../')
            dataset_class = datasets.MNIST
        else:
            data_dir = './data/fmnist/'  # Updated path (removed '../')
            dataset_class = datasets.FashionMNIST

        args.num_classes = 10  # Auto-update for MNIST/FashionMNIST
        args.num_channels = 1  # Auto-update for grayscale datasets

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = dataset_class(data_dir, train=True, download=True,
                                     transform=apply_transform)
        test_dataset = dataset_class(data_dir, train=False, download=True,
                                    transform=apply_transform)

        # Sample training data amongst users using modern Dirichlet approach
        if args.iid:
            print(f"Creating IID data distribution for {args.dataset.upper()}...")
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Modern Dirichlet-based Non-IID distribution
            alpha = getattr(args, 'alpha', 0.5)  # Default alpha if not specified
            min_samples = getattr(args, 'min_samples', 50)  # Default min samples
            
            print(f"Creating Non-IID data distribution for {args.dataset.upper()} with alpha={alpha}...")
            user_groups = mnist_dirichlet(train_dataset, args.num_users, 
                                        alpha=alpha, min_samples_per_user=min_samples)
            


    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported. Choose from: 'mnist', 'fmnist', 'cifar', 'cifar100'")

    # Optional: Analyze data distribution if requested
    if hasattr(args, 'analyze_data') and args.analyze_data:
        dataset_name = args.dataset.upper()
        iid_status = "IID" if args.iid else f"Non-IID (α={getattr(args, 'alpha', 0.5)})"
        analyze_distribution(user_groups, train_dataset, f"{dataset_name} {iid_status}")

    # Optional: Plot data distribution if requested
    if hasattr(args, 'plot_data') and args.plot_data:
        dataset_name = args.dataset.upper()
        iid_status = "IID" if args.iid else f"Non-IID (α={getattr(args, 'alpha', 0.5)})"
        save_path = getattr(args, 'save_plot', None)
        plot_distribution(user_groups, train_dataset, f"{dataset_name} {iid_status}", save_path)

    return train_dataset, test_dataset, user_groups

def evaluate_global_model_on_training_data(model, train_dataset, user_groups, device, batch_size=64):
    """Evaluate global model on all training data."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # Combine all client data
    all_indices = []
    for client_indices in user_groups.values():
        all_indices.extend(client_indices)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, all_indices),
        batch_size=batch_size, shuffle=False
    )
    
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    return 100.0 * total_correct / total_samples


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    """
    Print experiment details with modern Dirichlet distribution information
    """
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    Dataset   : {args.dataset.upper()}')
    print(f'    Classes   : {args.num_classes}')
    print(f'    Channels  : {args.num_channels}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    Distribution: IID')
    else:
        print('    Distribution: Non-IID (Dirichlet)')
        alpha = getattr(args, 'alpha', 0.5)
        print(f'    Dirichlet Alpha: {alpha}')
        
        # Provide interpretation of alpha value
        if alpha >= 10.0:
            print('    Heterogeneity: Nearly IID')
        elif alpha >= 1.0:
            print('    Heterogeneity: Moderate')
        elif alpha >= 0.5:
            print('    Heterogeneity: High')
        elif alpha >= 0.1:
            print('    Heterogeneity: Extreme')
        else:
            print('    Heterogeneity: Very Extreme')
            
        min_samples = getattr(args, 'min_samples', 10)
        print(f'    Min samples per user: {min_samples}')
    
    print(f'    Total users: {args.num_users}')
    print(f'    Fraction of users: {args.frac}')
    print(f'    Local Batch size: {args.local_bs}')
    print(f'    Local Epochs: {args.local_ep}\n')
    
    
    return

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two parameter vectors."""
    if isinstance(vec1, torch.Tensor) and isinstance(vec2, torch.Tensor):
        # PyTorch implementation
        vec1_flat = vec1.flatten()
        vec2_flat = vec2.flatten()
        
        return torch.nn.functional.cosine_similarity(
            vec1_flat.unsqueeze(0), 
            vec2_flat.unsqueeze(0)
        ).item()
    else:
        # NumPy implementation
        vec1_flat = np.array(vec1).flatten()
        vec2_flat = np.array(vec2).flatten()
        
        norm1 = np.linalg.norm(vec1_flat)
        norm2 = np.linalg.norm(vec2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1_flat, vec2_flat) / (norm1 * norm2)

def calculate_statistics(values):
    """Calculate comprehensive statistics for a sequence of values."""
    if not values:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0,
            'count': 0,
            'trend': 0
        }
    
    values_array = np.array(values)
    
    # Calculate trend (simple linear regression slope)
    x = np.arange(len(values_array))
    if len(values_array) > 1:
        trend = np.polyfit(x, values_array, 1)[0]
    else:
        trend = 0
    
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'count': len(values),
        'trend': float(trend)
    }

def extract_parameter_statistics(model_params):
    """Extract comprehensive statistics from model parameters."""
    if isinstance(model_params, dict):
        # State dict
        all_values = torch.cat([p.flatten() for p in model_params.values()])
    elif isinstance(model_params, list):
        # List of tensors
        all_values = torch.cat([p.flatten() for p in model_params])
    elif isinstance(model_params, torch.Tensor):
        # Single tensor
        all_values = model_params.flatten()
    else:
        # Unsupported type
        raise TypeError("Unsupported parameter type")
    
    # Convert to numpy
    all_values = all_values.detach().cpu().numpy()
    
    # Calculate statistics
    return {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'median': float(np.median(all_values)),
        'l1_norm': float(np.linalg.norm(all_values, 1)),
        'l2_norm': float(np.linalg.norm(all_values, 2)),
        'non_zeros': float((np.abs(all_values) > 1e-6).sum() / all_values.size)
    }
def plot_data_distribution(user_groups, dataset, save_path=None, title="Client Data Distribution"):
    labels = np.array(dataset.targets)
    num_clients = len(user_groups)
    num_classes = len(np.unique(labels))
    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
    for user, idxs in user_groups.items():
        user_labels = labels[list(idxs)]
        for c in range(num_classes):
            dist_matrix[user, c] = np.sum(user_labels == c)
    plt.figure(figsize=(12, 6))
    plt.imshow(dist_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Samples per class')
    plt.xlabel('Class')
    plt.ylabel('Client')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()