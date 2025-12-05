import yaml
import sys
sys.path.append(".")
from env_main import BioluminescentEnv
from env_generate import ProtocolEvaluator

def test_level_max_reward(level_file):
    """
    Verify that the theoretical maximum reward of 3.0 is achievable
    by simulating optimal play for a specific level.
    """
    print(f"Testing level: {level_file}")
    
    # Load the environment
    env = BioluminescentEnv("test_env")
    
    # Reset with the specific level
    level_id = level_file.replace('.yaml', '')
    obs = env.reset(mode="load", world_id=level_id)
    
    print(f"Initial state: {obs['handshakes_completed']}/3 handshakes completed")
    print(f"Protocol: {env._state['session']['active_protocol']}")
    print(f"Protocol params: {env._state['session']['protocol_params']}")
    
    total_reward = 0.0
    step_count = 0
    
    # Simulate optimal play for up to 3 handshakes
    while not env.done() and step_count < 40:
        step_count += 1
        
        # Get current incoming pattern
        incoming_pattern = env._state['session']['current_incoming_pattern']
        protocol = env._state['session']['active_protocol']
        params = env._state['session']['protocol_params']
        
        print(f"\nStep {step_count}: Incoming pattern length: {len(incoming_pattern)}")
        
        # Generate optimal response based on protocol
        response_pattern = generate_optimal_response(incoming_pattern, protocol, params)
        
        # Execute action
        action = {
            'action': 'RESPOND_PATTERN',
            'params': {
                'pattern_length': len(response_pattern),
                'pulse_colors': [p['color'] for p in response_pattern],
                'pulse_durations': [p['duration'] for p in response_pattern], 
                'pulse_intensities': [p['intensity'] for p in response_pattern]
            }
        }
        
        # Step environment
        env._t += 1
        state = env.transition(action)
        reward, events, info = env.reward(action)
        
        total_reward += reward
        print(f"Reward this step: {reward}, Total reward: {total_reward}")
        print(f"Handshakes completed: {state['session']['handshakes_completed']}")
        
        if not state['session']['session_active']:
            print("Session ended")
            break
            
    print(f"\nFinal total reward: {total_reward}")
    print(f"Theoretical maximum: 3.0")
    print(f"Achievement: {total_reward/3.0*100:.1f}%")
    
    return total_reward

def generate_optimal_response(incoming_pattern, protocol, params):
    """Generate the optimal response for a given protocol and parameters."""
    
    if protocol == "color_mirroring":
        colors = ["blue", "green", "purple", "white"]
        offset = params[0] % len(colors)
        
        response = []
        for pulse in incoming_pattern:
            new_color_idx = (colors.index(pulse['color']) + offset) % len(colors)
            new_pulse = pulse.copy()
            new_pulse['color'] = colors[new_color_idx]
            response.append(new_pulse)
        return response
        
    elif protocol == "duration_inversion":
        invert = params[1] % 2 == 1
        
        response = []
        for pulse in incoming_pattern:
            new_pulse = pulse.copy()
            if invert:
                new_pulse['duration'] = "long" if pulse['duration'] == "short" else "short"
            response.append(new_pulse)
        return response
        
    elif protocol == "intensity_parity":
        # For intensity_parity, we need to match the parity of high intensity count
        incoming_high_count = sum(1 if p['intensity'] == 'high' else 0 for p in incoming_pattern)
        modulo = max(1, params[0])
        target_parity = incoming_high_count % modulo
        
        response = []
        current_high_count = 0
        
        for i, pulse in enumerate(incoming_pattern):
            new_pulse = pulse.copy()
            
            # For the last pulse, set intensity to match target parity
            if i == len(incoming_pattern) - 1:
                needed_high = (target_parity - current_high_count) % modulo
                if needed_high > 0:
                    new_pulse['intensity'] = 'high'
                else:
                    new_pulse['intensity'] = 'low'
            else:
                # For other pulses, can use any intensity (let's use original)
                if new_pulse['intensity'] == 'high':
                    current_high_count += 1
                    
            response.append(new_pulse)
        return response
        
    elif protocol == "sequence_fibonacci":
        fib_a, fib_b = params[0], params[1]
        expected_length = (fib_a + fib_b) % 5 + 2
        
        # Create response with correct length
        response = []
        for i in range(expected_length):
            if i < len(incoming_pattern):
                response.append(incoming_pattern[i].copy())
            else:
                # Fill with default values if we need more pulses
                response.append({
                    'color': 'blue',
                    'duration': 'short', 
                    'intensity': 'low'
                })
        return response
        
    return incoming_pattern  # Fallback

if __name__ == "__main__":
    # Test with level_01
    test_level_max_reward("level_01.yaml")