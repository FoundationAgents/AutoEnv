import yaml
import os
from datetime import datetime
import json

def load_level(filename):
    """Load a level YAML file"""
    with open(f"levels/{filename}", 'r') as f:
        return yaml.safe_load(f)

def load_config():
    """Load the environment configuration"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward(level_data, config):
    """Calculate the theoretical maximum reward for a level"""
    
    # Extract reward multipliers from config
    harvest_reward = config['reward']['harvest_reward']  # 0.1
    product_reward = config['reward']['product_reward']  # 0.2
    relationship_reward_multiplier = config['reward']['relationship_reward_multiplier']  # 0.5
    market_reward_multiplier = config['reward']['market_reward_multiplier']  # 0.1
    
    # Extract initial resources
    initial_seeds = level_data['agent']['inventory']['seeds']  # 5
    initial_water = level_data['agent']['inventory']['water']  # 5
    initial_gifts = level_data['agent']['inventory']['gifts']  # 3
    initial_feed = level_data['agent']['inventory']['animal_feed']  # 3
    
    # Count available structures
    num_fields = len(level_data['objects']['crop_fields'])  # 4
    num_barns = len(level_data['objects']['barns'])  # 3
    num_villagers = len(level_data['objects']['villagers'])  # 3
    
    max_steps = 50  # From config
    
    # Calculate maximum possible rewards
    
    # 1. HARVEST REWARDS
    # We have 5 seeds and 5 water initially. Each crop needs 1 seed + 2 waterings to mature
    # So we can grow at most min(5 seeds, 5 water / 2) = min(5, 2.5) = 2 crops with initial resources
    # But we have 4 fields available, so limited by water
    max_crops_from_initial = min(initial_seeds, initial_water // 2)
    max_harvest_reward = max_crops_from_initial * harvest_reward
    
    # 2. RELATIONSHIP REWARDS  
    # We can give 3 gifts, each gives +5 relationship points
    max_relationship_gains = initial_gifts * 5  # 3 * 5 = 15 points
    max_relationship_reward = max_relationship_gains * relationship_reward_multiplier
    
    # 3. ANIMAL PRODUCT REWARDS
    # Each animal can be fed once (we have 3 feed), then produces 1 product every 5 steps
    # After feeding, animal produces for 10 steps, so 2 products per feeding cycle
    # With 50 total steps and 3 animals, if we feed optimally:
    # Feed all 3 animals at step 1, they produce at steps 6 and 11
    # We could potentially get more products if we had more feed, but limited to 3 feed initially
    max_products_per_animal = 2  # Products every 5 steps for 10 steps after feeding
    max_total_products = min(num_barns, initial_feed) * max_products_per_animal
    max_product_reward = max_total_products * product_reward
    
    # 4. MARKET SALE REWARDS
    # Sales depend on what we can produce and relationship multipliers
    # Base prices: crops = 1 coin, products = 2 coins each
    # Relationship multiplier: 1.0 + (avg_relationship / 100)
    # With 3 gifts giving +5 each to different villagers, avg relationship = 5
    # Initial relationships vary per level, let's calculate actual
    initial_relationships = [v['relationship'] for v in level_data['objects']['villagers']]
    total_initial_rel = sum(initial_relationships)
    
    # After giving gifts (assume we give to different villagers)
    total_relationship_after_gifts = total_initial_rel + (initial_gifts * 5)
    avg_relationship = total_relationship_after_gifts / num_villagers
    relationship_multiplier = 1.0 + (avg_relationship / 100)
    
    # Calculate potential sales
    potential_crops_to_sell = max_crops_from_initial
    potential_products_to_sell = max_total_products
    
    base_sales_value = (potential_crops_to_sell * 1) + (potential_products_to_sell * 2)
    total_coins_earned = base_sales_value * relationship_multiplier
    max_market_reward = total_coins_earned * market_reward_multiplier
    
    # TOTAL MAXIMUM REWARD
    total_max_reward = (max_harvest_reward + max_relationship_reward + 
                       max_product_reward + max_market_reward)
    
    return {
        'max_reward': round(total_max_reward, 2),
        'breakdown': {
            'harvest_reward': round(max_harvest_reward, 2),
            'relationship_reward': round(max_relationship_reward, 2),
            'product_reward': round(max_product_reward, 2),
            'market_reward': round(max_market_reward, 2)
        },
        'assumptions': {
            'max_crops': max_crops_from_initial,
            'max_products': max_total_products,
            'relationship_multiplier': round(relationship_multiplier, 2),
            'total_coins_from_sales': round(total_coins_earned, 2)
        }
    }

def main():
    config = load_config()
    results = {
        'environment_id': '20250918_152710_env_89_agricultural_life_simulation',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': {},
        'summary': {}
    }
    
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    all_max_rewards = []
    
    for level_file in level_files:
        print(f"Analyzing {level_file}...")
        level_data = load_level(level_file)
        reward_analysis = calculate_max_reward(level_data, config)
        
        results['levels'][level_file] = {
            'max_reward': reward_analysis['max_reward'],
            'calculation_method': 'analytical_optimization',
            'notes': 'Conservative estimate assuming optimal play with resource constraints',
            'breakdown': reward_analysis['breakdown'],
            'assumptions': reward_analysis['assumptions']
        }
        
        all_max_rewards.append(reward_analysis['max_reward'])
    
    # Calculate summary statistics
    results['summary'] = {
        'total_levels': len(all_max_rewards),
        'average_max_reward': round(sum(all_max_rewards) / len(all_max_rewards), 2),
        'min_max_reward': min(all_max_rewards),
        'max_max_reward': max(all_max_rewards)
    }
    
    # Save results
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to level_max_rewards.json")
    print(f"Analyzed {len(all_max_rewards)} levels")
    print(f"Average max reward: {results['summary']['average_max_reward']}")
    print(f"Range: {results['summary']['min_max_reward']} - {results['summary']['max_max_reward']}")

if __name__ == "__main__":
    main()