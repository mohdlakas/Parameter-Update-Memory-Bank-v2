# test_pumb_components.py
import torch
import numpy as np
import sys
import traceback
from memory_bank import MemoryBank
from embedding_generator import EmbeddingGenerator
from quality_metric import QualityMetric
from intelligent_selector import IntelligentSelector

def test_embedding_generator():
    """Test the embedding generator with various input types."""
    print("\n🧪 Testing Embedding Generator...")
    
    try:
        eg = EmbeddingGenerator()
        print(f"✓ EmbeddingGenerator initialized with dim: {eg.embedding_dim}")
        
        # Test 1: Dictionary of tensors (typical FL scenario)
        dummy_updates = {
            'layer1.weight': torch.randn(10, 5), 
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(5, 3),
            'layer2.bias': torch.randn(3)
        }
        
        embedding = eg.generate_embedding(dummy_updates)
        print(f"✓ Dict input - Embedding shape: {embedding.shape}")
        print(f"✓ Embedding stats: mean={embedding.mean():.4f}, std={embedding.std():.4f}")
        
        # Test 2: Single tensor
        single_tensor = torch.randn(100)
        embedding2 = eg.generate_embedding(single_tensor)
        print(f"✓ Single tensor - Embedding shape: {embedding2.shape}")
        
        # Test 3: Empty/edge case
        empty_dict = {}
        embedding3 = eg.generate_embedding(empty_dict)
        print(f"✓ Empty dict - Embedding shape: {embedding3.shape}")
        
        # Test 4: Very small updates
        tiny_updates = {'weight': torch.randn(2, 2) * 1e-8}
        embedding4 = eg.generate_embedding(tiny_updates)
        print(f"✓ Tiny updates - Embedding shape: {embedding4.shape}")
        
        return True, embedding
        
    except Exception as e:
        print(f"❌ EmbeddingGenerator failed: {e}")
        traceback.print_exc()
        return False, None

def test_quality_metric():
    """Test the quality metric computation."""
    print("\n📊 Testing Quality Metric...")
    
    try:
        qm = QualityMetric()
        print("✓ QualityMetric initialized")
        
        # Test 1: Good improvement scenario
        dummy_updates = {
            'layer1.weight': torch.randn(10, 5) * 0.1, 
            'layer1.bias': torch.randn(10) * 0.1
        }
        loss_improvement = 0.15  # Good improvement
        data_size = 100
        
        quality1 = qm.compute_quality(loss_improvement, dummy_updates, data_size)
        print(f"✓ Good scenario - Quality: {quality1:.4f}")
        
        # Test 2: Poor improvement scenario
        loss_improvement = -0.05  # Negative improvement
        quality2 = qm.compute_quality(loss_improvement, dummy_updates, data_size)
        print(f"✓ Poor scenario - Quality: {quality2:.4f}")
        
        # Test 3: Zero improvement
        quality3 = qm.compute_quality(0.0, dummy_updates, data_size)
        print(f"✓ Zero improvement - Quality: {quality3:.4f}")
        
        # Test 4: Large inconsistent updates
        noisy_updates = {
            'layer1.weight': torch.randn(10, 5) * 10,  # Very large updates
            'layer1.bias': torch.randn(10) * 0.001     # Very small updates
        }
        quality4 = qm.compute_quality(0.1, noisy_updates, data_size)
        print(f"✓ Noisy updates - Quality: {quality4:.4f}")
        
        # Test 5: Edge case - empty updates
        quality5 = qm.compute_quality(0.1, {}, data_size)
        print(f"✓ Empty updates - Quality: {quality5:.4f}")
        
        return True, quality1
        
    except Exception as e:
        print(f"❌ QualityMetric failed: {e}")
        traceback.print_exc()
        return False, None

def test_memory_bank():
    """Test the memory bank operations."""
    print("\n🏦 Testing Memory Bank...")
    
    try:
        mb = MemoryBank(embedding_dim=512, max_memories=100)
        print("✓ MemoryBank initialized")
        
        # Add some dummy memories
        for i in range(5):
            embedding = np.random.randn(512)
            quality = np.random.random()
            mb.add_memory(
                client_id=i, 
                embedding=embedding, 
                quality=quality, 
                round_num=1,
                loss_improvement=np.random.random() * 0.2
            )
        
        print(f"✓ Added 5 memories, bank size: {len(mb.memories)}")
        
        # Test client stats
        stats = mb.get_client_stats(client_id=0)
        print(f"✓ Client 0 stats: {stats}")
        
        # Test similarity search
        query_embedding = np.random.randn(512)
        similar_clients, similarities = mb.find_similar_patterns(query_embedding, k=3)
        print(f"✓ Found {len(similar_clients)} similar patterns")
        print(f"✓ Similarities: {similarities}")
        
        # Test aggregation weights
        client_ids = [0, 1, 2]
        embeddings = {i: np.random.randn(512) for i in client_ids}
        qualities = {i: np.random.random() for i in client_ids}
        
        weights = mb.compute_aggregation_weights(client_ids, embeddings, qualities)
        print(f"✓ Aggregation weights: {weights}")
        print(f"✓ Weights sum to: {sum(weights.values()):.4f}")
        
        # Test memory overflow
        print("Testing memory overflow...")
        for i in range(150):  # Exceed max_memories
            embedding = np.random.randn(512)
            mb.add_memory(i+10, embedding, 0.5, 2)
        
        print(f"✓ After overflow, bank size: {len(mb.memories)} (should be <= 100)")
        
        return True, mb
        
    except Exception as e:
        print(f"❌ MemoryBank failed: {e}")
        traceback.print_exc()
        return False, None

def test_intelligent_selector():
    """Test the intelligent client selector."""
    print("\n🧠 Testing Intelligent Selector...")
    
    try:
        # First create a memory bank with some history
        mb = MemoryBank()
        
        # Add some client history
        clients_performance = {
            0: [0.8, 0.7, 0.9],  # Good client
            1: [0.3, 0.2, 0.4],  # Poor client  
            2: [0.6, 0.5, 0.7],  # Average client
            3: [0.9, 0.8, 0.85], # Excellent client
            4: [0.1, 0.15, 0.2]  # Very poor client
        }
        
        for client_id, qualities in clients_performance.items():
            for round_num, quality in enumerate(qualities):
                embedding = np.random.randn(512)
                mb.add_memory(client_id, embedding, quality, round_num)
        
        # Initialize selector
        selector = IntelligentSelector(mb, initial_rounds=2, exploration_ratio=0.3)
        print("✓ IntelligentSelector initialized")
        
        # Test cold start (early rounds)
        available_clients = list(range(10))
        selected = selector.select_clients(available_clients, num_clients=3, current_round=1)
        print(f"✓ Cold start selection (round 1): {selected}")
        
        # Test intelligent selection (later rounds)
        selected = selector.select_clients(available_clients, num_clients=3, current_round=10)
        print(f"✓ Intelligent selection (round 10): {selected}")
        
        # Test with different numbers of clients
        selected_large = selector.select_clients(available_clients, num_clients=7, current_round=10)
        print(f"✓ Large selection (7 clients): {selected_large}")
        
        # Test edge case - more clients requested than available
        few_clients = [0, 1, 2]
        selected_few = selector.select_clients(few_clients, num_clients=5, current_round=10)
        print(f"✓ Edge case selection: {selected_few}")
        
        return True, selector
        
    except Exception as e:
        print(f"❌ IntelligentSelector failed: {e}")
        traceback.print_exc()
        return False, None

def test_integration():
    """Test integration between all components."""
    print("\n🔗 Testing Component Integration...")
    
    try:
        # Initialize all components
        eg = EmbeddingGenerator()
        qm = QualityMetric()
        mb = MemoryBank()
        selector = IntelligentSelector(mb)
        
        print("✓ All components initialized")
        
        # Simulate a federated learning round
        num_clients = 10
        selected_clients = list(range(5))  # Simulate 5 selected clients
        
        client_updates = {}
        client_embeddings = {}
        client_qualities = {}
        
        for client_id in selected_clients:
            # Simulate parameter updates
            updates = {
                'fc1.weight': torch.randn(64, 784) * 0.01,
                'fc1.bias': torch.randn(64) * 0.01,
                'fc2.weight': torch.randn(10, 64) * 0.01,
                'fc2.bias': torch.randn(10) * 0.01
            }
            
            # Generate embedding
            embedding = eg.generate_embedding(updates)
            
            # Compute quality (simulate loss improvement)
            loss_improvement = np.random.normal(0.05, 0.02)  # Small positive improvement
            data_size = np.random.randint(50, 200)
            quality = qm.compute_quality(loss_improvement, updates, data_size)
            
            # Store everything
            client_updates[client_id] = updates
            client_embeddings[client_id] = embedding
            client_qualities[client_id] = quality
            
            # Add to memory bank
            mb.add_memory(client_id, embedding, quality, round_num=1, 
                         loss_improvement=loss_improvement)
        
        print(f"✓ Processed {len(selected_clients)} clients")
        print(f"✓ Quality range: {min(client_qualities.values()):.3f} - {max(client_qualities.values()):.3f}")
        
        # Test aggregation weights
        weights = mb.compute_aggregation_weights(selected_clients, client_embeddings, client_qualities)
        print(f"✓ Aggregation weights computed: {weights}")
        
        # Test next round selection
        next_selected = selector.select_clients(list(range(num_clients)), 5, current_round=2)
        print(f"✓ Next round selection: {next_selected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_test():
    """Test performance with realistic sizes."""
    print("\n⚡ Performance Testing...")
    
    try:
        import time
        
        # Test with realistic federated learning sizes
        eg = EmbeddingGenerator(embedding_dim=512)
        mb = MemoryBank(embedding_dim=512, max_memories=1000)
        
        # Simulate ResNet-like parameter sizes
        large_updates = {
            'conv1.weight': torch.randn(64, 3, 7, 7),
            'bn1.weight': torch.randn(64),
            'bn1.bias': torch.randn(64),
            'layer1.0.conv1.weight': torch.randn(64, 64, 3, 3),
            'layer1.0.bn1.weight': torch.randn(64),
            'fc.weight': torch.randn(1000, 512),
            'fc.bias': torch.randn(1000)
        }
        
        # Time embedding generation
        start_time = time.time()
        for _ in range(100):
            embedding = eg.generate_embedding(large_updates)
        embedding_time = time.time() - start_time
        print(f"✓ 100 embeddings generated in {embedding_time:.3f}s ({embedding_time*10:.1f}ms each)")
        
        # Time memory operations
        start_time = time.time()
        for i in range(100):
            embedding = np.random.randn(512)
            mb.add_memory(i, embedding, np.random.random(), 1)
        memory_time = time.time() - start_time
        print(f"✓ 100 memory additions in {memory_time:.3f}s ({memory_time*10:.1f}ms each)")
        
        # Time similarity search
        start_time = time.time()
        query = np.random.randn(512)
        for _ in range(100):
            similar_clients, similarities = mb.find_similar_patterns(query, k=5)
        search_time = time.time() - start_time
        print(f"✓ 100 similarity searches in {search_time:.3f}s ({search_time*10:.1f}ms each)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 PUMB Components Testing Suite")
    print("=" * 50)
    
    results = {}
    
    # Run individual component tests
    results['embedding'] = test_embedding_generator()[0]
    results['quality'] = test_quality_metric()[0]
    results['memory'] = test_memory_bank()[0]
    results['selector'] = test_intelligent_selector()[0]
    
    # Run integration test
    results['integration'] = test_integration()
    
    # Run performance test
    results['performance'] = run_performance_test()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize():12} : {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! Your PUMB implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Common fixes:")
        print("   - Check that all required packages are installed (faiss-cpu, torch, numpy)")
        print("   - Verify that your component files are in the same directory")
        print("   - Look for any import errors or missing methods")

if __name__ == "__main__":
    main()
