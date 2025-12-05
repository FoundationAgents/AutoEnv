# ==================== environment crafting prompts ====================
CRAFT_ENV_DESIGN_PROMPT = """

You are tasked with creating a detailed environment design document based on the user's requirements.

Your job: **given the requirements, produce**
1. A detailed environment design document (wrapped in <env_design></env_design>)

Content Rules, Make sure the environment design document contains the following information:
------------- 
1. Background: The setting and context of the environment.
2. Objective: What the agent needs to accomplish or achieve.
3. State Setup: How the environment state is initialized and structured.
4. Actions: What actions are available to the agent and their parameters.
5. State Transition Rule: How actions change the environment state.
6. Rewards: How the agent receives feedback for different behaviors. 
7. Observation: What information player can observe and how it supports learning. Design observations that balance challenge with learnability - provide sufficient cues for agents to identify patterns and make progress, while maintaining appropriate difficulty. Consider information hierarchy (what to show/hide) and ensure observations contain actionable signals that guide strategic decision-making.
8. Termination: When and how the environment episode ends.
9. Special Features: Any unique mechanics or constraints in this environment.

Important Reminder
-------------
1. **Environment Should be Learnable**, When designing, you should make sure the reward and transition are learnable, it means in this environment's different levels, each item's basic functionality should remain consistent, otherwise the environment cannot be used for learning.

2. **Observation Should be Clear**, When designing, you should ensure the agent has sufficient information to understand available actions and their parameters. Additionally, clearly specify what feedback and state changes the agent will observe after taking actions to support effective learning.

3. **Difficulty should stem from complexity instead of randomness**, When designing, the reward or transition patterns should not be completely random, but rather should maintain similar, learnable logic across different levels.

4. **Reward should be Clear**, When designing, you should specify whether this game uses binary reward or cumulative reward. For the former, you only need to design one reward event. For the latter, you can design multiple reward events.

5. **Termination should be Clear**, When designing, you should specify when the game ends. Default termination conditions include step limits and resource depletion. However, there are exceptions: for instance, when a specific action's resource is exhausted, it should not directly cause termination but rather disable that action. Another point is that for binary reward (0/1) environments, the game typically ends after success is achieved, while for cumulative reward environments, some environments have no clear success condition and rewards continue to accumulate.

6. **Do not construct the description in list format, but rather in structured paragraph format.**

Requirements
-------------
{requirements}

"""

CRAFT_ENV_YAML_PROMPT = """
You are an **Environment Designer LLM**.

Your job: **given the requirements, produce**
1. A **valid YAML** environment config (wrapped in <env_config></env_config>)
2. Concise implementation guidance for downstream agents (wrapped in <env_implement_help></env_implement_help>)

Important output rules
----------------------
- Inside <env_config> **ONLY** YAML. No markdown fences, no comments, no extra text.
- The YAML **MUST** parse with `yaml.safe_load` (Python). Use two-space indent, no tabs.
- Top-level keys (all required, names must match abstraction):  
  `meta`, `state_template`, `generator`, `transition`, `reward`,  
  `observation`, `termination`, `skin`
- Inside <env_implement_help>: give file / class names, and **what each abstract
  method should do**, but keep it under **120 lines**.
- Outside the two tag blocks, you may reason / explain, **but NEVER output YAML** again.
- The generator is used to generate usable levels, so make sure the generator design is reasonable, and explain its logic clearly in implementation_help.


Before writing, **think step-by-step (do not show your chain of thought)**.
Check that:
✓ all mandatory keys present  
✓ field names align with abstractions  
✓ values meet the requirements


Important checklist
----------------------
1. make sure the max_step is only set in the termination section, and add the explanation that when reading max_step, if the level has changed max_steps, it should override the environment's self.configs["termination"]["max_steps"].


Below is the information provided:

## Environment Description
{env_desc}

## Environment-config example
{config_yaml_example}

## Environment abstraction (Python)
{environment_abstraction}

## Observation & level_generator abstraction
{observation_abstraction}

{generator_abstraction}

==== REPEAT YOUR TASK ====
Your job: **given the requirements, produce**
1. A **valid YAML** environment config (wrapped in <env_config></env_config>)
2. Concise implementation guidance for downstream agents (wrapped in <env_implement_help></env_implement_help>)

====  YOUR RESPONSE STARTS  ====
"""

CRAFT_ENV_CODE_PROMPT = """
You are an **Environment Code Engineer**.

Your job: **given the requirements, environment config, and implementation help, produce**
1. A **valid Python** environment code (wrapped in <env_main_code></env_main_code>)
2. A **valid Python** observation code (wrapped in <env_obs_code></env_obs_code>)
3. A **valid Python** level_generator code (wrapped in <env_generate_code></env_generate_code>)
4. A valid script to use the above code to generate concrete levels, which can be run directly from the command line with Python (wrapped in <env_main_code_use></env_main_code_use>)

Your code must be:
- **Consistent and fully integrated**: The three code parts must work together seamlessly. The environment code must correctly import and use the observation and level_generator code you provide.
- **Directly runnable**: The code should be ready to use in a Python project, with all necessary imports and class/method definitions.
- **Able to generate levels**: The environment must be able to initialize, step, and generate levels using the provided generator and observation logic.
- **Parameter Format Consistency**: The environment's transition method must expect action parameters in dictionary format (e.g., params.get('param_name')) to match the action space specification.

Important output rules
----------------------
- Inside <env_main_code>, <env_obs_code>, <env_generate_code>, and <env_main_code_use> blocks: **ONLY** Python. No markdown fences, no comments, no extra text.
- All imports must be valid and consistent across the code blocks.
- The environment code must instantiate and use the observation and generator classes you define, not placeholders.
- If you define a class in <env_obs_code> or <env_generate_code>, import and use it in <env_main_code> as needed.
- All class and method names must match those described in the implementation help and config.
- Do not use any undefined variables, classes, or methods.
- The code you provide inside <env_obs_code> will be saved as env_obs.py, and the code inside <env_generate_code> will be saved as env_generate.py, the code inside <env_main_code> will be saved as env_main.py for import and use in the environment.
- When importing the abstraction class, you should use the following code:
  from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
  from base.env.base_observation import ObservationPolicy
  from base.env.base_generator import WorldGenerator
- The environment folder is {env_folder_path}, so when write the dsl_config, you can load just with f"./config.yaml"
- All level files must be loaded from and saved to the directory: ./levels/ (relative to the environment folder)
Before writing, **think step-by-step for design this code (do not show your chain of thought)**.



Below is the information provided:

## Environment Description
{env_desc}

## Environment-config
{config_yaml}

## Environment abstraction (Python)
{environment_abstraction}

## Environment Implement Help
{env_implement_help}

## Observation & level_generator abstraction
{observation_abstraction}

{generator_abstraction}


==== REPEAT YOUR TASK ====
Your job: **given the requirements, environment config, and implementation help, produce**
1. A **valid Python** environment code (wrapped in <env_main_code></env_main_code>)
2. A **valid Python** observation code (wrapped in <env_obs_code></env_obs_code>)
3. A **valid Python** level_generator code (wrapped in <env_generate_code></env_generate_code>)
4. A **valid Python** script to generate levels using the above code, wrapped in <env_main_code_use></env_main_code_use>

The four code parts must be consistent and directly usable together.

====  YOUR RESPONSE STARTS  ====
"""


CRAFT_ENV_CODE_AND_INSTRUCTION_PROMPT = """
You are an **Environment Engineer**.

Your job: **given the requirements, environment config, and implementation help, produce**
1. A **valid Python** environment code (wrapped in <env_main_code></env_main_code>)
2. A **valid Python** observation code (wrapped in <env_obs_code></env_obs_code>)
3. A **valid Python** level_generator code (wrapped in <env_generate_code></env_generate_code>)
4. A valid script to use the above code to generate concrete levels, which can be run directly from the command line with Python (wrapped in <env_main_code_use></env_main_code_use>)
5. An agent instruction for agent understanding (wrapped in <agent_instruction></agent_instruction>)
6. A standard action space description for agent use (wrapped in <action_space></action_space>)

Your code must be:
- **Consistent and fully integrated**: The three code parts must work together seamlessly. The environment code must correctly import and use the observation and level_generator code you provide.
- **Directly runnable**: The code should be ready to use in a Python project, with all necessary imports and class/method definitions.
- **Able to generate levels**: The environment must be able to initialize, step, and generate levels using the provided generator and observation logic.
- **Parameter Format Consistency**: The environment's transition method must expect action parameters in dictionary format (e.g., params.get('param_name')) to match the action space specification.



### Rules for Environment Code
----------------------
- Inside <env_main_code>, <env_obs_code>, <env_generate_code>, and <env_main_code_use> blocks: **ONLY** Python. No markdown fences, no comments, no extra text.
- All imports must be valid and consistent across the code blocks.
- The environment code must instantiate and use the observation and generator classes you define, not placeholders.
- If you define a class in <env_obs_code> or <env_generate_code>, import and use it in <env_main_code> as needed.
- All class and method names must match those described in the implementation help and config.
- Do not use any undefined variables, classes, or methods.
- You must design observations carefully to ensure they provide sufficient information for agents to make informed decisions. If an action requires specific parameters or has context-dependent validity, the observation should include the necessary information to determine valid parameter values (e.g., available item IDs, valid coordinates, resource counts). Consider whether this information belongs in the observation state or should be clarified in the action space description.
- The code you provide inside <env_obs_code> will be saved as env_obs.py, and the code inside <env_generate_code> will be saved as env_generate.py, the code inside <env_main_code> will be saved as env_main.py for import and use in the environment.
- When importing the abstraction class, you should use the following code:
  from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
  from base.env.base_observation import ObservationPolicy
  from base.env.base_generator import WorldGenerator
- The environment folder is {env_folder_path}, so when write the dsl_config, you can load just with f"./config.yaml"
- All level files must be loaded from and saved to the directory: ./levels/ (relative to the environment folder)


### Agent Instruction
- Provide a **clear, concise instruction** of the environment that helps an agent to act in the environment:
  - The background of the environment
  - What the agent's goal or objective is
  - Key elements or constraints the agent should be aware of
- Keep it simple and focused - avoid technical implementation details
- Example format: "You are in a maze environment. Your task is to navigate from the starting position to the exit while avoiding obstacles."

### Action Space Description:
- Output a **JSON list** describing the action space, to be saved in a `.txt` file.
- For each action:
  - The `"name"` field **must exactly match** the parameter accepted by the environment's `transition` method.
  - The `"description"` should clearly explain what the action does, but **do not reveal** which state variables it changes or what rewards it may yield.
  - The `"parameters"` field should list all parameters required for the action, with a brief description for each.
- **CRITICAL**: After the JSON list, provide this exact formatting instruction:
  "Actions should be formatted as dictionaries with an 'action' key specifying the action name and a 'params' key containing the required parameters as a dictionary. For example: {{\"action\": \"ACTION_NAME\", \"params\": {{\"param1\": value1, \"param2\": value2}}}}"

### Formatting rules
----------------
- Do **not** include markdown fences, comments, or extra text—**output only the content**.
- The output must be directly usable by downstream agents.
- Agent instruction should be wrapped in <agent_instruction></agent_instruction>
- Action space should be wrapped in <action_space></action_space>

Before writing, **think step-by-step for design this code (do not show your chain of thought)**.
Please Make Sure:
1. All action names and parameters match those in the environment code  
2. Action descriptions are clear and concise  
3. No information about state changes or rewards is leaked

Below is an example for action space:

[
  {{
    "name": "move",
    "description": "Move the agent to the specified location.",
    "parameters": {{
      "dx": "horizontal displacement for movement",
      "dy": "vertical displacement for movement"
    }}
  }},
  ... more actions
]


Below is the information provided:

## Environment Description
{env_desc}

## Environment-config
{config_yaml}

## Environment abstraction (Python)
{environment_abstraction}

## Environment Implement Help
{env_implement_help}

## Observation & level_generator abstraction
{observation_abstraction}

{generator_abstraction}

==== REPEAT YOUR TASK ====
Your job: **given the requirements, environment config, and implementation help, produce**
1. A **valid Python** environment code (wrapped in <env_main_code></env_main_code>)
2. A **valid Python** observation code (wrapped in <env_obs_code></env_obs_code>)
3. A **valid Python** level_generator code (wrapped in <env_generate_code></env_generate_code>)
4. A valid script to use the above code to generate concrete levels, which can be run directly from the command line with Python (wrapped in <env_main_code_use></env_main_code_use>)
5. An agent instruction for agent understanding (wrapped in <agent_instruction></agent_instruction>)
6. A standard action space description for agent use (wrapped in <action_space></action_space>)
""" 


CRAFT_ENV_VALIDATOR_PROMPT = """
You are an **Environment Generator and Validator**.

Your job: **given the requirements, environments code, generator code produce**
1. Generate a validator code to validate generated levels(wrapped in <env_validator_code></env_validator_code>).

### The checklist's responsibility
{validator_checklist}

Make Sure the Validator code can fill this check.

Below is the information provided:

## Environment Description
{env_desc}

## Environment-config
{config_yaml}

## Environment code
{env_code}

## Observation 
{observation_code}

## Generator
{generator_code} 


=== REPEAT YOUR TASK ===
Your job: **given the requirements, environments code, generator code produce**
1. Generate a validator code to validate generated levels(it must be wrapped in <env_validator_code></env_validator_code>).

==== YOUR RESPONSE STARTS ====
"""


CRAFT_AGENT_INSTRUCTION_PROMPT = """
You are an **Environment Designer LLM**.

Your job: **Given the requirements and the environment code, produce**
1. An agent instruction for agent understanding (wrapped in <agent_instruction></agent_instruction>)
2. A standard action space description for agent use (wrapped in <action_space></action_space>)

Output requirements
-------------------

### Agent Instruction
- Provide a **clear, concise instruction** of the environment that helps an agent to act in the environment:
  - The background of the environment
  - What the agent's goal or objective is
  - Key elements or constraints the agent should be aware of
- Keep it simple and focused - avoid technical implementation details
- Example format: "You are in a maze environment. Your task is to navigate from the starting position to the exit while avoiding obstacles."

### Action Space Description:
- Output a **JSON list** describing the action space, to be saved in a `.txt` file.
- For each action:
  - The `"name"` field **must exactly match** the parameter accepted by the environment's `transition` method.
  - The `"description"` should clearly explain what the action does, but **do not reveal** which state variables it changes or what rewards it may yield.
  - The `"parameters"` field should list all parameters required for the action, with a brief description for each.
- **CRITICAL**: After the JSON list, provide this exact formatting instruction:
  "Actions should be formatted as dictionaries with an 'action' key specifying the action name and a 'params' key containing the required parameters as a dictionary. For example: {{\"action\": \"ACTION_NAME\", \"params\": {{\"param1\": value1, \"param2\": value2}}}}"

Formatting rules
----------------
- Do **not** include markdown fences, comments, or extra text—**output only the content**.
- The output must be directly usable by downstream agents.
- Environment description should be wrapped in <env_desc></env_desc>
- Action space should be wrapped in <action_space></action_space>

Before writing, **think step-by-step (do not show your chain of thought)**.
Check that:
✓ Environment description is clear and goal-oriented
✓ All action names and parameters match those in the environment code  
✓ Action descriptions are clear and concise  
✓ No information about state changes or rewards is leaked

Below is an example for action space:

[
  {{
    "name": "move",
    "description": "Move the agent to the specified location.",
    "parameters": {{
      "dx": "horizontal displacement for movement",
      "dy": "vertical displacement for movement"
    }}
  }},
  ... more actions
]

Below is the information provided:

## Environment Description
{env_desc}

## Environment code
{env_code}

==== REPEAT YOUR TASK ====
Your job: **Given the requirements and the environment code, produce**
1. An agent instruction for agent understanding (wrapped in <agent_instruction></agent_instruction>)
2. A standard action space description for agent use (wrapped in <action_space></action_space>)

====  YOUR RESPONSE STARTS  ====
"""

# ==================== Coding Agents Prompt ====================

CODE_FOR_RUN_ENV_MAIN_USE = """
import os
import sys
main_use_current_dir = os.path.dirname(os.path.abspath(__file__))
main_use_project_root = os.path.dirname(os.path.dirname(main_use_current_dir))  # Go up two levels to reach AutoEnv root
if main_use_project_root not in sys.path:
    sys.path.insert(0, main_use_project_root)
"""


VALIDATOR_CHECKLIST = """

  1.LEVEL SOLVABILITY ANALYSIS: Critical check for impossible puzzles
   1. ACTION CONSTRAINT ANALYSIS**: Understand your environment's fundamental limitations
     1. What can each action type actually modify in the environment?
     2. What are the preconditions for each action to succeed?
     3. Are there irreversible operations that could block progress?
     4. Do actions have resource costs or cooldowns that limit usage?
   2. TARGET REACHABILITY**: For each level, verify target state is actually achievable
     1. Map initial state → target state: what transformations are required?
     2. Resource availability: are all required resources present or obtainable?
     3. State space connectivity: can you reach target from initial state through available actions?
     4. Step counting: is target reachable within max_steps limit?
   3. **COMMON IMPOSSIBLE PATTERNS TO AVOID**:
     1. Target requires modifying locked/protected elements with no unlock mechanism
     2. Required resources not available and not obtainable through any action sequence
     3. Circular dependencies where achieving goal A requires goal B, but goal B requires goal A
     4. Target state violates environment's invariants or physics rules
     5. Actions insufficient to bridge the gap between initial and target states
   4. **VALIDATION LOGIC FRAMEWORK**:
     ```python
     def check_level_solvability(initial_state, target_state, available_actions):
         # 1. Resource check: are target elements obtainable?
         # 2. Constraint check: do actions have sufficient power to reach target?
         # 3. Path existence: can you navigate from initial to target state?
         # 4. Step budget: is solution achievable within step limits?
         return is_solvable, blocking_issues
     ```

  2. REWARD STRUCTURE DESIGN: Critical check for incentive alignment
   1. GOAL-ORIENTED REWARDS: Design rewards that prioritize problem-solving over action usage
     1. **Target Achievement**: Highest rewards (15-20 points) should come from achieving the main objective
     2. **Progress Rewards**: Medium rewards (3-10 points) for measurable progress toward goals
     3. **Action Usage**: Low rewards (0.5-2 points) for basic operations, avoid over-rewarding exploration
   2. AVOID INCENTIVE MISALIGNMENT:
     1. **Action Grinding**: Don't make repeated action usage more rewarding than goal achievement
     2. **Exploration Loops**: Implement diminishing returns for repeated observations/information gathering
     3. **Action Farming**: Prevent agents from earning high scores through meaningless repetitive actions
   3. REWARD DESIGN PRINCIPLES:
     1. **Sparse > Dense**: Better to give fewer, meaningful rewards than constant small rewards
     2. **Achievement > Process**: Weight completion heavily over intermediate steps
     3. **Efficiency Incentive**: Bonus rewards for solving with fewer steps
     4. **Failure Cost**: Appropriate penalties to discourage random actions, but not so harsh as to prevent exploration
   4. VALIDATION QUESTIONS FOR REWARD DESIGN:
     ```python
     def validate_reward_structure():
         # Can agents achieve high scores without solving the actual problem?
         # Are tool operations rewarded proportionally to their contribution to goal achievement?
         # Does the reward structure encourage efficient problem-solving?
         # Are there any "reward loops" that agents can exploit for easy points?
         return is_well_designed, reward_issues
     ```
"""

ECODE_AGENT_CALCULATE_MAX_REWARD_PROMPT = """
🎯 MAXIMUM REWARD CALCULATION TASK
Target Environment: {env_id}
Working Directory: {workspace}

Your job:
1. Analyze each generated level in the ./levels/ directory
2. For each level, calculate the theoretical maximum reward an optimal agent could achieve
3. Generate a JSON file recording the maximum reward for each level

⚠️ CRITICAL REQUIREMENTS:
- Work in the environment directory: {workspace}
- All level files are located in ./levels/ directory (*.yaml files)
- You have access to the environment code (env_main.py), config (config.yaml), and validator
- Use the environment's reward structure and termination conditions to calculate maximum possible scores
- Consider all possible action sequences and state transitions

📋 Step-by-step approach:
1) 🔍 ANALYZE ENVIRONMENT STRUCTURE:
   - Read config.yaml to understand the reward structure
   - Examine env_main.py to understand how rewards are calculated
   - Check termination conditions and maximum steps
   - Understand the action space and state transitions

2) 📊 LEVEL ANALYSIS:
   - Load all level files from ./levels/ directory
   - For each level, analyze the initial state and target conditions
   - Identify all possible reward sources (completion rewards, progress rewards, bonus rewards, etc.)
   - Consider penalties and negative rewards that might affect the total

3) 🧮 REWARD CALCULATION LOGIC:
   - For each level, determine the optimal strategy that maximizes reward
   - Consider different scenarios:
     * Best case: All positive rewards obtained, no penalties
     * Realistic optimal: Achievable maximum considering constraints and step limits
   - Account for:
     * Success/completion rewards
     * Progress-based rewards
     * Efficiency bonuses
     * Time penalties (if any)
     * Action costs (if any)

4) 🔄 SYSTEMATIC CALCULATION:
   - Create a Python script to systematically calculate maximum rewards
   - For each level:
     a) Load the level YAML file
     b) Initialize the environment with this level
     c) Simulate or analyze optimal action sequences
     d) Calculate the maximum achievable reward
     e) Record the result

5) 📝 JSON OUTPUT GENERATION:
   - Create a JSON file: level_max_rewards.json
   - Format should be:
   ```json
   {{
     "environment_id": "{env_id}",
     "calculation_timestamp": "YYYY-MM-DD HH:MM:SS",
     "levels": {{
       "level_filename.yaml": {{
         "max_reward": 100.0,
         "calculation_method": "optimal_path_analysis",
         "notes": "Assumes perfect execution with all bonuses"
       }},
       ...
     }},
     "summary": {{
       "total_levels": 5,
       "average_max_reward": 85.5,
       "min_max_reward": 50.0,
       "max_max_reward": 120.0
     }}
   }}
   ```

6) ✅ VALIDATION:
   - Verify that calculated rewards are reasonable and achievable
   - Cross-check with environment's reward structure
   - Ensure JSON file is valid and well-formatted

🔧 IMPLEMENTATION GUIDELINES:
- Use the environment's own reward calculation methods when possible
- If direct simulation is too complex, use analytical approaches based on reward structure
- For cumulative reward environments: sum all possible positive rewards, subtract minimum penalties
- For binary reward environments: focus on success conditions and completion bonuses
- Document your calculation methodology in the JSON output

🚨 COMPLETION SIGNAL:
When level_max_rewards.json is successfully created with valid data for all levels, run:
echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'

⚠️ IMPORTANT NOTES:
- The calculation should be conservative but optimistic - assume optimal play but realistic constraints
- If exact calculation is impossible, provide best estimates with clear documentation
- Include error handling for levels that cannot be analyzed
"""

ECODE_AGENT_CODE_FIX_PROMPT = """
🎯 ENVIRONMENT CODE REPAIR TASK
Target Environment: {env_id}
Working Directory: {workspace}

Your Job: Fix and validate core environment code structure and basic functionality.

⚠️ YOU MUST KNOW:
1. You are working in the environment directory: {workspace}
2. The levels should be saved to ./levels/ (relative to current directory)
3. When you want to import basic abstract classes, you can use the following code:
  from base.env.base_generator import WorldGenerator
  from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
  from base.env.base_observation import ObservationPolicy
4. **Mandatory**: done() method signature: `def done(self, state=None) -> bool:`

🔧 CODE FIX TASKS:
1. **Environment Structure Analysis**:
   - Understand how the environment works: env_main_use.py, env_main.py, config.yaml, env_generate.py
   - Identify and fix import errors
   - Ensure proper inheritance from base classes

2. **Core Functionality Repair**:
   - Fix YAML generation format issues
   - Ensure environment can be properly initialized and loaded
   - Fix done() method signature: `def done(self, state=None) -> bool:`
   - Ensure all action handlers use `params.get('param_name')` for dictionary-format parameters
   - Verify action names match exactly between action_space.txt and transition method
   - Ensure observe_semantic() returns proper format

3. **Generator & Validator Repair**:
   - Test generator to create at least one working level
   - Read and fix env_validator.py if it has bugs
   - Ensure validator follows the comprehensive checklist:
     {validator_checklist}
   - Fix any validator issues found during testing

4. **Basic Integration Test**:
   - Generate ONLY 1-2 test levels to verify core functionality (for testing only)
   - Load test levels via the environment class
   - Write simple action sequences to test transition and reward logic
   - Test observation structure and content
   - **NOTE: These are temporary test levels, not final levels**

5. **Code Quality Check**:
   - Fix runtime errors and exceptions
   - Ensure consistent data structures
   - Validate configuration consistency

✅ SUCCESS CRITERIA:
- Environment code runs without import/syntax errors
- Generator can create valid level files (1-2 test levels are sufficient)
- Validator can validate levels correctly
- Environment can load levels and execute basic actions
- Core observation system works
- **TEST LEVELS ONLY: Final level generation happens in next stage**

🚨 COMPLETION: When core code structure is fixed and basic functionality works, run:
echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'

"""


ECODE_AGENT_LEVEL_GENERATION_PROMPT = """
🎯 ENVIRONMENT LEVEL GENERATION & VALIDATION TASK
Target Environment: {env_id}
Working Directory: {workspace}

Your Job: Generate comprehensive validated levels and perform integration testing.

🎯 **CRITICAL REQUIREMENT: GENERATE EXACTLY 15 FINAL LEVELS**

📋 **RECOMMENDED EXECUTION STRATEGY:**
1. **CLEAR ALL EXISTING LEVELS**: `rm -f levels/*.yaml` (remove any test levels from previous stages)
2. Create systematic level generation script that produces exactly 15 levels
3. Use naming convention: level_01.yaml, level_02.yaml, ..., level_15.yaml  
4. After each generation attempt: `ls levels/*.yaml | wc -l` to verify count
5. If count ≠ 15: delete all and regenerate
6. Only complete task when `ls levels/*.yaml | wc -l` shows exactly 15

⚠️ **IMPORTANT**: This stage is responsible for producing the final 15 levels. Previous stages may have generated 1-2 test levels for validation, but this stage must clear them and generate the final set.

⚠️ PREREQUISITES:
- Environment code structure should already be fixed
- Basic generator and validator should be working
- Core environment functionality should be operational

🔧 LEVEL GENERATION TASKS:
1. **Final Level Generation** (THIS IS THE MAIN TASK):
   - **STEP 1: Clear all previous levels**: `rm -f levels/*.yaml` 
   - **STEP 2: Generate EXACTLY 15 final levels** using the validator and generator
   - **STEP 3: Use systematic naming**: level_01.yaml to level_15.yaml
   - **STEP 4: Validate each level** by loading it into the environment
   - **STEP 5: Ensure all levels are solvable** and meet quality standards
   - **STEP 6: Verify count** with `ls levels/*.yaml | wc -l` must equal 15

2. **Quality Assurance** (SECONDARY TASKS):
   - Test with actual SolverAgent on a few levels:
     * Use: `python ../../scripts/run_solver.py --env ENV_ID --level LEVEL_NAME --max-steps 5`
     * Replace ENV_ID with your environment folder name
     * Test with 2-3 different level files (not all 15)
   - Verify SolverAgent can interact without errors
   - Check that observations are reasonable and actionable

3. **Observation Validation**:
   - Agents must see all information needed for meaningful decisions
   - Check env_obs.py and run commands to verify observation quality
   - Ensure agents know which actions are available and their effects
   - Write action sequences to test observation completeness

4. **Level Quality Check**:
   - Each level should pass both validator checks and environment loading tests
   - Levels should be strategically interesting and learnable
   - Test edge cases and boundary conditions on a few sample levels

5. **Final System Integration**:
   - **MANDATORY VERIFICATION: Count levels with `ls levels/*.yaml | wc -l`**
   - **REQUIREMENT: Must show exactly 15 levels**
   - If not 15 levels, delete all and regenerate from scratch
   - Test environment functionality with the 15 final levels
   - Environment handles various gameplay scenarios
   - Robust error handling for invalid actions
   - Consistent reward and termination behavior

6. **Final Count Verification & Completion**:
   - **Step 1: Run `ls levels/*.yaml | wc -l` to verify exactly 15 levels**
   - **Step 2: Run `ls levels/` to show all level files**  
   - **Step 3: If count ≠ 15, delete all levels and start over**
   - **Step 4: Document final level structure and key features**
   - Create backup folder for any old code if needed
   - Write comprehensive test results

✅ SUCCESS CRITERIA:
- **EXACTLY 15 levels generated (verified by `ls levels/*.yaml | wc -l` = 15)**
- All 15 levels pass validator checks and environment loading tests
- SolverAgent can successfully interact with multiple levels
- Observations provide sufficient information for decision-making
- System integration testing completed successfully
- **FINAL VERIFICATION: Show level count before completion**

🔧 Final Result Documentation:
- If successful: Write JSON file with fields: class name, levels count, max_steps in config
- If failed: Document what caused failure and current status

🚨 COMPLETION: **BEFORE COMPLETING, MANDATORY VERIFICATION:**
1. Run: `ls levels/*.yaml | wc -l` (must show 15)
2. Run: `ls levels/` (to list all level files)  
3. Only when count is exactly 15, run: echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
4. If count ≠ 15, delete all levels and restart generation process

"""