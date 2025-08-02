import numpy as np
from collections import Counter, defaultdict  # ADDED: Missing imports
from embedding_generator import EmbeddingGenerator
from quality_metric import QualityMetric
from memory_bank import MemoryBank
import logging
import torch


class IntelligentSelector:
    def __init__(self, memory_bank, initial_rounds=3, exploration_ratio=0.4):
        """Initialize the intelligent client selector.
        
        Args:
            memory_bank: Reference to the memory bank object
            initial_rounds: Number of initial rounds for cold start
            exploration_ratio: Ratio of random client selection (exploration)
        """
        self.memory_bank = memory_bank
        self.initial_rounds = initial_rounds
        self.exploration_ratio = exploration_ratio
        
        # Enhanced logging attributes
        self.selection_history = []
        self.client_selection_count = Counter()
        self.quality_score_history = defaultdict(list)
        self.selection_reasons = []
        self.logger = logging.getLogger('PUMB_Selection')
        self.round_count = 0  # Track rounds for logging

    def select_clients(self, available_clients, num_to_select, round_num, 
                       global_direction=None):
        """Select clients for the current round based on memory bank data.
        
        Args:
            available_clients: List of available client IDs
            num_to_select: Number of clients to select
            round_num: Current federated learning round
            global_direction: Current global update direction (optional)
            
        Returns:
            selected_clients: List of selected client IDs
        """
        self.round_count = round_num  # Update round count for logging

        # Cold start: Random selection for initial rounds
        if round_num < self.initial_rounds:
            selected = self._random_selection(available_clients, num_to_select)
            self.logger.info(f"Round {round_num}: Cold start - randomly selected {selected}")
            return selected
        
        # Hybrid selection strategy
        num_exploit = int(num_to_select * (1 - self.exploration_ratio))
        num_explore = num_to_select - num_exploit
        
        # Ensure at least one client for each strategy
        num_exploit = max(1, min(num_exploit, len(available_clients) - 1))
        num_explore = num_to_select - num_exploit
        
        # Exploitation: Select clients with best reliability scores
        exploit_candidates = []
        for client_id in available_clients:
            reliability = self.memory_bank.get_client_reliability(client_id)
            exploit_candidates.append((client_id, reliability))
        
        # Sort by reliability (descending)
        exploit_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top reliable clients
        exploit_clients = [client_id for client_id, _ in exploit_candidates[:num_exploit]]
        
        # Remove selected clients from the pool for exploration
        remaining_clients = [c for c in available_clients if c not in exploit_clients]
        
        # Exploration: Random selection from remaining clients
        explore_clients = self._random_selection(remaining_clients, num_explore)
        
        # Combine selections
        selected_clients = exploit_clients + explore_clients

        # Enhanced logging
        self._log_selection_details(selected_clients, available_clients, num_to_select)

        return selected_clients
    
    def _random_selection(self, client_pool, num_to_select):
        """Randomly select clients from the pool."""
        if not client_pool:
            return []
            
        if num_to_select >= len(client_pool):
            return client_pool.copy()
            
        return np.random.choice(client_pool, num_to_select, replace=False).tolist()
    
    def _log_selection_details(self, selected_clients, available_clients, num_clients):
        # Update counters
        self.selection_history.append(selected_clients.copy())
        for client in selected_clients:
            self.client_selection_count[client] += 1

        # Get comprehensive client information
        all_scores = {}
        all_reliability = {}
        all_trends = {}
        
        for client_id in available_clients:
            client_stats = self.memory_bank.get_client_statistics(client_id)
            if client_stats is not None:
                score = client_stats['recent_quality']
                reliability = client_stats['reliability_score']
                trend = client_stats['quality_trend']
            else:
                score = 0.0
                reliability = 0.0
                trend = 0.0
            
            all_scores[client_id] = score
            all_reliability[client_id] = reliability
            all_trends[client_id] = trend
            self.quality_score_history[client_id].append(score)

        # Log selection details
        selected_scores = [all_scores[cid] for cid in selected_clients]
        selected_reliability = [all_reliability[cid] for cid in selected_clients]
        unselected_clients = [cid for cid in available_clients if cid not in selected_clients]
        unselected_scores = [all_scores[cid] for cid in unselected_clients] if unselected_clients else []

        self.logger.info(f"=== ROUND {self.round_count} CLIENT SELECTION ===")
        self.logger.info(f"Exploration ratio: {self.exploration_ratio:.3f}")
        self.logger.info(f"Available clients: {len(available_clients)}, Selecting: {num_clients}")
        self.logger.info(f"Selected clients: {selected_clients}")
        self.logger.info(f"Selected scores: {[f'{s:.3f}' for s in selected_scores]}")
        self.logger.info(f"Selected reliability: {[f'{r:.3f}' for r in selected_reliability]}")
        
        if selected_scores:
            self.logger.info(f"Selected score stats: mean={np.mean(selected_scores):.3f}, "
                            f"std={np.std(selected_scores):.3f}, "
                            f"min={np.min(selected_scores):.3f}, "
                            f"max={np.max(selected_scores):.3f}")

        if unselected_scores:
            self.logger.info(f"Unselected score stats: mean={np.mean(unselected_scores):.3f}, "
                            f"std={np.std(unselected_scores):.3f}")

        # Show top reliable clients
        top_reliable = self.memory_bank.get_top_reliable_clients(5)
        self.logger.info(f"Top 5 reliable clients: {top_reliable}")

        # Rest of the logging code remains the same...
        if len(self.selection_history) >= 5:
            recent_selections = self.selection_history[-5:]
            frequent_clients = Counter()
            for round_clients in recent_selections:
                for client in round_clients:
                    frequent_clients[client] += 1
            self.logger.info(f"Most frequent clients (last 5 rounds): {frequent_clients.most_common(5)}")

        if self.round_count > 3:
            trending_up = []
            trending_down = []
            for client_id in selected_clients:
                trend = all_trends[client_id]
                if trend > 0.05:
                    trending_up.append((client_id, trend))
                elif trend < -0.05:
                    trending_down.append((client_id, trend))
            if trending_up:
                self.logger.info(f"Clients trending UP: {trending_up}")
            if trending_down:
                self.logger.info(f"Clients trending DOWN: {trending_down}")


    # Fix 3: Implement Correct Weight Calculation from Paper
    def get_aggregation_weights(self, selected_clients, client_models, data_sizes, 
                            global_direction=None, embedding_gen=None, 
                            quality_calc=None, current_round=0):
        """
        THEORY-ALIGNED: Implement multiplicative weight combination from paper
        w_i^t = (reliability_i^t · similarity_i^t · quality_i^t) / Σ_j(...)
        """
        if not selected_clients:
            return {}
        
        weights = {}
        
        for client_id in selected_clients:
            # Reliability from memory bank
            reliability = max(0.1, self.memory_bank.get_client_reliability(client_id))
            
            # Similarity computation (if enough history exists)
            if self.memory_bank.round_count > 3 and embedding_gen:
                # Generate embedding for current update
                param_update = {name: client_models[client_id][name] - 
                            self.memory_bank.get_last_global_state().get(name, 
                            torch.zeros_like(client_models[client_id][name]))
                            for name in client_models[client_id]}
                
                current_embedding = embedding_gen.generate_embedding(param_update)
                similarity = self.memory_bank.compute_similarity(client_id, current_embedding)
            else:
                similarity = 1.0
            
            # Current quality (will be computed in main training loop)
            # For weight calculation, use recent average quality
            recent_qualities = self.memory_bank.get_recent_qualities(client_id, window=3)
            quality = np.mean(recent_qualities) if recent_qualities else 0.5
            
            # Multiplicative combination as per theory
            weights[client_id] = reliability * similarity * quality
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {client_id: w / total_weight for client_id, w in weights.items()}
        else:
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(selected_clients)
            weights = {client_id: uniform_weight for client_id in selected_clients}
        
        return weights

    # def get_aggregation_weights(self, selected_clients, client_updates, data_sizes, 
    #                         global_direction=None, embedding_gen=None, quality_calc=None, pumb_alpha=0.4):
    #     """
    #     Calculate aggregation weights based on client update quality.

    #     Args:
    #         selected_clients: List of selected client IDs
    #         client_updates: Dictionary of client parameter updates
    #         data_sizes: Dictionary of client data sizes
    #         global_direction: Current global update direction (optional)
    #         embedding_gen: Optional embedding generator from the server
    #         quality_calc: Optional quality calculator from the server
    #         pumb_alpha: Weighting factor for reliability vs. data size (passed, not hard-coded)
        
    #     Returns:
    #         weights: Dictionary mapping client_id to aggregation weight
    #     """
    #     embedding_gen = embedding_gen or EmbeddingGenerator()
    #     quality_calc = quality_calc or QualityMetric()
        
    #     if not selected_clients:
    #         return {}
            
    #     # Calculate basic weights based on data sizes
    #     total_data = sum(data_sizes.values())
    #     if total_data == 0:
    #         base_weights = {client_id: 1.0 / len(selected_clients) for client_id in selected_clients}
    #     else:
    #         base_weights = {client_id: data_sizes[client_id] / total_data for client_id in selected_clients}
        
    #     # If not enough history, use data-weighted average
    #     if self.memory_bank.round_count < 3:
    #         self.logger.info(f"Using data-weighted aggregation (early rounds)")
    #         return base_weights
            
    #     # Calculate reliability scores
    #     reliability_scores = {}
    #     for client_id in selected_clients:
    #         reliability = self.memory_bank.get_client_reliability(client_id)
    #         reliability_scores[client_id] = max(0.1, reliability)
        
    #     total_reliability = sum(reliability_scores.values())
    #     if total_reliability > 0:
    #         reliability_weights = {
    #             client_id: score / total_reliability 
    #             for client_id, score in reliability_scores.items()
    #         }
    #     else:
    #         reliability_weights = {client_id: 1.0 / len(selected_clients) for client_id in selected_clients}
        
    #     # Use passed pumb_alpha instead of hard-coded value
    #     alpha = pumb_alpha
    #     weights = {
    #         client_id: (1-alpha) * base_weights[client_id] + alpha * reliability_weights[client_id]
    #         for client_id in selected_clients
    #     }
        
    #     # Normalize final weights
    #     total_weight = sum(weights.values())
    #     if total_weight > 0:
    #         weights = {client_id: w / total_weight for client_id, w in weights.items()}
        
    #     self.logger.info(f"Aggregation weights: {[(cid, f'{w:.3f}') for cid, w in weights.items()]}")
        
    #     return weights
