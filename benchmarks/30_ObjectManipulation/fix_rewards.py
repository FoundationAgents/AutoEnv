import json

# Read the existing file
with open('level_max_rewards.json', 'r') as f:
    data = json.load(f)

# Fix floating point precision issues
for level_name, level_data in data['levels'].items():
    level_data['max_reward'] = round(level_data['max_reward'], 1)
    level_data['breakdown']['chore_completions'] = round(level_data['breakdown']['chore_completions'], 1)

# Fix summary statistics
data['summary']['average_max_reward'] = round(data['summary']['average_max_reward'], 1)
data['summary']['min_max_reward'] = round(data['summary']['min_max_reward'], 1)
data['summary']['max_max_reward'] = round(data['summary']['max_max_reward'], 1)

# Write back to file
with open('level_max_rewards.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Fixed floating point precision issues")
print(f"All levels have max reward: {data['summary']['max_max_reward']}")