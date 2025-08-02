import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
import logging
matplotlib.use('Agg')

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

from federated_PUMB import PUMBFederatedServer
from utils_dir import get_dataset, exp_details, plot_data_distribution

# Create directories for saving results
os.makedirs('../save/objects', exist_ok=True)
os.makedirs('../save/images', exist_ok=True)
os.makedirs('../save/logs', exist_ok=True)

def setup_pumb_logging(args):
    """Setup comprehensive logging for PUMB diagnostics"""
    log_filename = f'../save/logs/pumb_diagnostics_{args.dataset}_{args.model}_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}].log'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Create specific loggers
    main_logger = logging.getLogger('PUMB_Main')
    main_logger.info(f"=== PUMB FEDERATED LEARNING EXPERIMENT START ===")
    main_logger.info(f"Dataset: {args.dataset}, Model: {args.model}")
    main_logger.info(f"IID: {args.iid}, Alpha: {getattr(args, 'alpha', 'NA')}")
    main_logger.info(f"Epochs: {args.epochs}, Fraction: {args.frac}")
    main_logger.info(f"Local epochs: {args.local_ep}, Local batch size: {args.local_bs}")
    
    return main_logger

def log_training_stability(round_num, train_losses, train_accuracies, selected_clients, server_stats=None):
    """Log overall training stability metrics"""
    
    logger = logging.getLogger('PUMB_Stability')
    
    if round_num < 5:
        return
    
    # Loss stability analysis
    recent_losses = train_losses[-5:]
    loss_variance = np.var(recent_losses)
    loss_trend = np.polyfit(range(5), recent_losses, 1)[0]
    
    # Accuracy trend
    recent_accuracies = train_accuracies[-5:]
    acc_trend = np.polyfit(range(5), recent_accuracies, 1)[0]
    
    logger.info(f"=== ROUND {round_num} STABILITY ANALYSIS ===")
    logger.info(f"Loss variance (last 5): {loss_variance:.6f}")
    logger.info(f"Loss trend: {loss_trend:.6f} ({'improving' if loss_trend < 0 else 'worsening'})")
    logger.info(f"Accuracy trend: {acc_trend:.6f} ({'improving' if acc_trend > 0 else 'declining'})")
    
    # Log server statistics if available
    if server_stats:
        logger.info(f"Memory bank size: {server_stats.get('memory_bank_size', 'N/A')}")
        logger.info(f"Has global direction: {server_stats.get('has_global_direction', 'N/A')}")
    
    # Detect problematic patterns
    if loss_variance > 0.5:
        logger.warning(f"HIGH LOSS VARIANCE detected: {loss_variance:.4f}")
    
    if abs(loss_trend) < 0.01 and loss_variance > 0.1:
        logger.warning("OSCILLATORY BEHAVIOR: High variance with no clear trend")
    
    if loss_trend > 0.1:
        logger.warning("DIVERGENCE WARNING: Loss increasing significantly")
    
    # Check for convergence
    if abs(loss_trend) < 0.001 and loss_variance < 0.01:
        logger.info("CONVERGENCE INDICATOR: Loss appears to be stabilizing")

def log_round_summary(round_num, selected_clients, local_losses, data_sizes, train_acc, test_acc=None):
    """Log summary of each training round"""
    logger = logging.getLogger('PUMB_Round')
    
    # Calculate client statistics
    loss_improvements = []
    for client_id, (loss_before, loss_after) in local_losses.items():
        improvement = loss_before - loss_after
        loss_improvements.append(improvement)
    
    avg_improvement = np.mean(loss_improvements)
    std_improvement = np.std(loss_improvements)
    
    # Data size statistics
    total_data = sum(data_sizes.values())
    avg_data_size = np.mean(list(data_sizes.values()))
    
    logger.info(f"=== ROUND {round_num} SUMMARY ===")
    logger.info(f"Selected clients: {selected_clients}")
    logger.info(f"Total training samples: {total_data}")
    logger.info(f"Avg loss improvement: {avg_improvement:.4f} ± {std_improvement:.4f}")
    logger.info(f"Training accuracy: {train_acc:.4f}")
    if test_acc is not None:
        logger.info(f"Test accuracy: {test_acc:.4f}")

def log_experiment_summary(args, train_loss, train_accuracy, test_acc, total_time):
    """Log final experiment summary"""
    logger = logging.getLogger('PUMB_Summary')
    
    logger.info(f"=== EXPERIMENT COMPLETED ===")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Final training loss: {train_loss[-1]:.4f}")
    logger.info(f"Final training accuracy: {train_accuracy[-1]:.4f}")
    logger.info(f"Final test accuracy: {test_acc:.4f}")
    
    # Calculate convergence metrics
    if len(train_loss) >= 10:
        early_loss = np.mean(train_loss[:5])
        late_loss = np.mean(train_loss[-5:])
        loss_reduction = (early_loss - late_loss) / early_loss
        
        early_acc = np.mean(train_accuracy[:5])
        late_acc = np.mean(train_accuracy[-5:])
        acc_improvement = late_acc - early_acc
        
        logger.info(f"Loss reduction: {loss_reduction:.2%}")
        logger.info(f"Accuracy improvement: {acc_improvement:.4f}")

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    # Setup comprehensive logging
    main_logger = setup_pumb_logging(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Plot client data distribution (IID/non-IID)
    plot_data_distribution(
        user_groups, train_dataset,
        save_path='../save/images/data_distribution_{}_iid[{}]_alpha[{}].png'.format(
            args.dataset, args.iid, getattr(args, 'alpha', 'NA')
        ),
        title="Client Data Distribution (IID={})".format(args.iid)
    )
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()
    main_logger.info(f"Model architecture: {global_model}")

    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize PUMB server
    server = PUMBFederatedServer(global_model, optimizer, loss_fn, args, embedding_dim=512)
    main_logger.info("PUMB Federated Server initialized")

    train_loss, train_accuracy = [], []
    test_accuracy_history = []
    print_every = 2

    # for epoch in tqdm(range(args.epochs)):
    #     main_logger.info(f'Starting Global Training Round: {epoch+1}')
    #     global_model.train()
    #     m = max(int(args.frac * args.num_users), 1)
    #     available_clients = list(range(args.num_users))
    #     selected_clients = server.select_clients(available_clients, m)

    #     local_updates = {}
    #     local_losses = {}
    #     data_sizes = {}

    #     for idx in selected_clients:
    #         local_model = LocalUpdate(args=args, dataset=train_dataset,
    #                                   idxs=user_groups[idx], logger=None)
    #         # Evaluate loss before local training
    #         loss_before = local_model.inference(model=global_model)[1]
    #         # Local training
    #         w, loss = local_model.update_weights(
    #             model=copy.deepcopy(global_model), global_round=epoch)
                
    #         # Create model with updated weights for evaluation
    #         updated_model = copy.deepcopy(global_model)
    #         updated_model.load_state_dict(w)
    #         loss_after = local_model.inference(model=updated_model)[1]
            
    #         # Calculate parameter update
    #         param_update = {name: w[name] - global_model.state_dict()[name]
    #                         for name in w}
    #         local_updates[idx] = param_update
    #         local_losses[idx] = (loss_before, loss_after)
    #         data_sizes[idx] = len(user_groups[idx])

    #     # Update global model using PUMB
    #     server.update_global_model(local_updates, local_losses, data_sizes)

    #     # Logging
    #     loss_avg = np.mean([loss_after for _, loss_after in local_losses.values()])
    #     train_loss.append(loss_avg)

    #     # Calculate avg training accuracy over all users at every epoch
    #     list_acc = []
    #     global_model.eval()
    #     for c in range(args.num_users):
    #         local_model = LocalUpdate(args=args, dataset=train_dataset,
    #                                   idxs=user_groups[c], logger=None)
    #         acc, _ = local_model.inference(model=global_model)
    #         list_acc.append(acc)
    #     train_accuracy.append(np.mean(list_acc))

    #     # Periodic test evaluation for stability analysis
    #     test_acc = None
    #     if (epoch + 1) % 5 == 0:
    #         test_acc, _ = test_inference(args, global_model, test_dataset)
    #         test_accuracy_history.append(test_acc)

    #     # Log round summary
    #     log_round_summary(epoch + 1, selected_clients, local_losses, 
    #                      data_sizes, train_accuracy[-1], test_acc)

    #     # Log training stability
    #     log_training_stability(epoch + 1, train_loss, train_accuracy, 
    #                          selected_clients, server.get_server_stats())

    #     # Periodic hyperparameter logging
    #     if (epoch + 1) % 10 == 0:
    #         main_logger.info(f"Hyperparameter check - LR: {args.lr}, "
    #                        f"Local epochs: {args.local_ep}, "
    #                        f"Batch size: {args.local_bs}")

    #     if (epoch+1) % print_every == 0:
    #         print(f' \nAvg Training Stats after {epoch+1} global rounds:')
    #         print(f'Training Loss : {np.mean(np.array(train_loss))}')
    #         print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    for epoch in tqdm(range(args.epochs)):
        main_logger.info(f'Starting Global Training Round: {epoch+1}')
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # Intelligent Client Selection
        m = max(int(args.frac * args.num_users), 1)
        available_clients = list(range(args.num_users))
        selected_clients = server.select_clients(available_clients, m)

        client_models = {} # Store full client model states
        client_losses = {} # Store (before, after) loss tuples
        data_sizes = {} # Store client data sizes
        param_updates = {}# Store parameter updates for quality calculation

        # Local updates and quality computation
        all_loss_improvements = []

        for idx in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=None)
            # Download global model (θ_i^{t-1} ← θ^{t-1})                        
            initial_state = copy.deepcopy(global_model.state_dict())

            # Local update (θ_i^t, ΔL_i^t ← LocalUpdate)
            loss_before = local_model.inference(model=global_model)[1]
            updated_weights, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            # Evaluate loss after local update
            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(updated_weights)
            loss_after = local_model.inference(model=temp_model)[1]

            # Calculate parameter update (Δθ_i^t ← θ_i^t - θ_i^{t-1})
            param_update = {name: updated_weights[name] - initial_state[name]
                            for name in updated_weights}

            # Store data for aggregation
            client_models[idx] = updated_weights
            client_losses[idx] = (loss_before, loss_after)
            data_sizes[idx] = len(user_groups[idx])
            param_updates[idx] = param_update

            # Collect loss improvements for normalization
            loss_improvement = max(0, loss_before - loss_after)
            all_loss_improvements.append(loss_improvement)

        # Quality computation and memory bank update
        for idx in selected_clients:
            loss_before, loss_after = client_losses[idx]

            # Compute quality with proper normalization
            quality = server.quality_calc.calculate_quality(
                loss_before, loss_after, data_sizes, param_updates[idx],
                epoch, idx, all_loss_improvements
            )

            # generate embedding
            embedding = server.embedding_gen.generate_embedding(param_updates[idx])

            # Add to memory bank
            server.memory_bank.add_update(idx, embedding, quality, epoch)

        # Compute aggregation weights
        aggregation_weights = server.client_selector.get_aggregation_weights(
            selected_clients, client_models, data_sizes,
            server.global_direction, server.embedding_gen,
            server.quality_calc, epoch
        )

        # Quality-weighted aggregation (θ^t ← Σ w_i^t θ_i^t)
        #server._aggregate_updates(client_models, aggregation_weights)
        server.update_global_model(client_models, client_losses, data_sizes)
        # Update global direction for next round
        # current_state = server._get_model_state_copy()
        # if server.prev_model_state is not None:
        #     server.global_direction = {
        #         name: current_state[name] - server.prev_model_state[name]
        #         for name in current_state
        #     }
        # server.prev_model_state = current_state
        # server.current_round += 1

        # Logging and evaluation
        loss_avg = np.mean([loss_after for _, loss_after in client_losses.values()])
        train_loss.append(loss_avg)

        # Compute training accuracy
        list_acc = []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=None)
            acc, _ = local_model.inference(model=global_model)
            list_acc.append(acc)
        train_accuracy.append(np.mean(list_acc))

        if (epoch+1) % print_every == 0:
            print(f'Avg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {loss_avg:.6f}')
            print(f'Train Accuracy: {100*train_accuracy[-1]:.2f}%')
            print(f'Aggregation weights std: {np.std(list(aggregation_weights.values())):.4f}')

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Log experiment summary
    total_time = time.time() - start_time
    log_experiment_summary(args, train_loss, train_accuracy, test_acc, total_time)

    # Save train_loss and train_accuracy
    file_name = '../save/objects/PUMB_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(total_time))


    # Plot Loss and Accuracy
    plt.figure(figsize=(12, 8))
    plt.title(f'PUMB Results - Test Accuracy: {test_acc*100:.2f}%\nTraining Loss and Accuracy vs Communication Rounds')
    plt.xlabel('Communication Rounds')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(range(len(train_loss)), train_loss, color='r', label='Training Loss', linewidth=2)
    ax2.plot(range(len(train_accuracy)), train_accuracy, color='k', label='Training Accuracy', linewidth=2)
    ax1.set_ylabel('Training Loss', color='r')
    ax2.set_ylabel('Training Accuracy', color='k')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='k')

    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    # Add timestamp to filename

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Add param names to filename for clarity
    plot_filename = (
        '../save/images/PUMB_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_explr[{}_exploration]_lr[{}_lr]_initR[{}_initR]_loss_acc_{}.png'
        .format(
            args.dataset,
            args.model,
            args.epochs,
            args.frac,
            args.iid,
            args.local_ep,
            args.local_bs,
            getattr(args, 'pumb_exploration_ratio', 'NA'),
            getattr(args, 'lr', 'NA'),
            getattr(args, 'pumb_initial_rounds', 'NA'),
            timestamp
        )
    )
    # Clean up param names in filename
    plot_filename = plot_filename.replace('_exploration', '').replace('_lr', '').replace('_initR', '')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    main_logger.info("=== PUMB FEDERATED LEARNING EXPERIMENT COMPLETED ===")
