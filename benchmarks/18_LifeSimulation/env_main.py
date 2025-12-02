import sys
sys.path.append("/Users/didi/Documents/GitHub/AutoEnv")
from base.env.base_env import SkinEnv
from env_obs import FiveByFiveObs  
from env_generate import ValleyFarmGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class Animal:
    def __init__(self, species: str):
        self.species = species
        self.hungry = True
        self.timer = 0
        self.product_count = 0
    
    def feed(self):
        self.hungry = False
        self.timer = 10
    
    def tick(self):
        if not self.hungry and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.hungry = True
        
        if not self.hungry and self.timer % 5 == 0 and self.timer > 0:
            self.product_count += 1

class ValleyFarmEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = FiveByFiveObs()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        with open("./config.yaml", "r") as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load" and world_id:
            self._state = self._load_world(world_id)
        else:
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        filepath = f"./levels/{world_id}.yaml"
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = ValleyFarmGenerator(str(self.env_id), self.configs["generator"])
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        action_name = action["action"]
        params = action.get("params", {})
        
        # Reset action result
        self._last_action_result = {"action": action_name, "success": False, "message": ""}
        
        if action_name == "MoveNorth":
            self._move_agent(0, -1)
        elif action_name == "MoveSouth":
            self._move_agent(0, 1)
        elif action_name == "MoveEast":
            self._move_agent(1, 0)
        elif action_name == "MoveWest":
            self._move_agent(-1, 0)
        elif action_name == "PlantSeed":
            self._plant_seed()
        elif action_name == "WaterCrop":
            self._water_crop()
        elif action_name == "HarvestCrop":
            self._harvest_crop()
        elif action_name == "FeedAnimal":
            self._feed_animal()
        elif action_name == "CollectProduct":
            self._collect_product()
        elif action_name == "GiveGift":
            self._give_gift()
        elif action_name == "SellAtMarket":
            self._sell_at_market()
        elif action_name == "Wait":
            self._wait()
        
        # Update animal timers
        self._update_animals()
        
        return self._state
    
    def _move_agent(self, dx: int, dy: int):
        x, y = self._state["agent"]["pos"]
        new_x, new_y = x + dx, y + dy
        
        if 0 <= new_x < 15 and 0 <= new_y < 15:
            # Check for obstacles (none in this environment)
            self._state["agent"]["pos"] = [new_x, new_y]
            self._last_action_result["success"] = True
            self._last_action_result["message"] = f"Moved to ({new_x}, {new_y})"
    
    def _plant_seed(self):
        if self._state["agent"]["inventory"]["seeds"] <= 0:
            return
        
        x, y = self._state["agent"]["pos"]
        
        # Find field at current position
        for field in self._state["objects"]["crop_fields"]:
            if field["pos"] == [x, y] and field["crop"]["stage"] == "empty":
                field["crop"]["stage"] = "seedling"
                field["crop"]["waterings"] = 0
                self._state["agent"]["inventory"]["seeds"] -= 1
                self._last_action_result["success"] = True
                self._last_action_result["message"] = "Planted seed"
                return
    
    def _water_crop(self):
        if self._state["agent"]["inventory"]["water"] <= 0:
            return
        
        x, y = self._state["agent"]["pos"]
        
        # Find field at current position
        for field in self._state["objects"]["crop_fields"]:
            if field["pos"] == [x, y] and field["crop"]["stage"] != "empty":
                crop = field["crop"]
                if crop["stage"] == "seedling":
                    crop["stage"] = "growing"
                    crop["waterings"] += 1
                elif crop["stage"] == "growing":
                    crop["stage"] = "mature"
                    crop["waterings"] += 1
                
                self._state["agent"]["inventory"]["water"] -= 1
                self._last_action_result["success"] = True
                self._last_action_result["message"] = f"Watered crop, now {crop['stage']}"
                return
    
    def _harvest_crop(self):
        x, y = self._state["agent"]["pos"]
        
        # Find field at current position
        for field in self._state["objects"]["crop_fields"]:
            if field["pos"] == [x, y] and field["crop"]["stage"] == "mature":
                field["crop"]["stage"] = "empty"
                field["crop"]["waterings"] = 0
                self._state["agent"]["inventory"]["crops"] += 1
                self._last_action_result["success"] = True
                self._last_action_result["message"] = "Harvested crop"
                self._last_action_result["crops_harvested"] = 1
                return
    
    def _feed_animal(self):
        if self._state["agent"]["inventory"]["animal_feed"] <= 0:
            return
        
        x, y = self._state["agent"]["pos"]
        
        # Find barn at current position
        for barn in self._state["objects"]["barns"]:
            if barn["pos"] == [x, y]:
                animal = barn["animal"]
                if animal["hungry"]:
                    animal["hungry"] = False
                    animal["timer"] = 10
                    self._state["agent"]["inventory"]["animal_feed"] -= 1
                    self._last_action_result["success"] = True
                    self._last_action_result["message"] = f"Fed {animal['species']}"
                    return
    
    def _collect_product(self):
        x, y = self._state["agent"]["pos"]
        
        # Find barn at current position
        for barn in self._state["objects"]["barns"]:
            if barn["pos"] == [x, y]:
                animal = barn["animal"]
                if animal["product_count"] > 0:
                    products_collected = animal["product_count"]
                    animal["product_count"] = 0
                    self._state["agent"]["inventory"]["animal_products"] += products_collected
                    self._last_action_result["success"] = True
                    self._last_action_result["message"] = f"Collected {products_collected} products"
                    self._last_action_result["products_collected"] = products_collected
                    return
    
    def _give_gift(self):
        if self._state["agent"]["inventory"]["gifts"] <= 0:
            return
        
        x, y = self._state["agent"]["pos"]
        
        # Check adjacent positions for villagers
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_x, check_y = x + dx, y + dy
                
                for villager in self._state["objects"]["villagers"]:
                    if villager["pos"] == [check_x, check_y]:
                        old_relationship = villager["relationship"]
                        villager["relationship"] += 5  # Fixed increment
                        villager["mood"] = "friendly" if villager["relationship"] >= 20 else "neutral"
                        self._state["agent"]["inventory"]["gifts"] -= 1
                        
                        self._last_action_result["success"] = True
                        self._last_action_result["message"] = f"Gave gift to {villager['id']}"
                        self._last_action_result["relationship_gain"] = 5
                        self._last_action_result["villager_id"] = villager["id"]
                        return
    
    def _sell_at_market(self):
        x, y = self._state["agent"]["pos"]
        market_pos = self._state["objects"]["market"]
        
        if [x, y] != market_pos:
            return
        
        crops = self._state["agent"]["inventory"]["crops"]
        products = self._state["agent"]["inventory"]["animal_products"]
        
        if crops == 0 and products == 0:
            return
        
        # Calculate relationship multiplier
        total_relationship = sum(v["relationship"] for v in self._state["objects"]["villagers"])
        avg_relationship = total_relationship / 3
        relationship_multiplier = 1.0 + (avg_relationship / 100)  # 1.0 to 1.4 multiplier
        
        # Base prices
        crop_price = 1
        product_price = 2
        
        total_coins = int((crops * crop_price + products * product_price) * relationship_multiplier)
        
        self._state["agent"]["inventory"]["crops"] = 0
        self._state["agent"]["inventory"]["animal_products"] = 0
        self._state["agent"]["inventory"]["coins"] += total_coins
        
        self._last_action_result["success"] = True
        self._last_action_result["message"] = f"Sold goods for {total_coins} coins"
        self._last_action_result["coins_earned"] = total_coins
    
    def _wait(self):
        self._last_action_result["success"] = True
        self._last_action_result["message"] = "Waited"
    
    def _update_animals(self):
        for barn in self._state["objects"]["barns"]:
            animal = barn["animal"]
            if not animal["hungry"] and animal["timer"] > 0:
                animal["timer"] -= 1
                if animal["timer"] <= 0:
                    animal["hungry"] = True
                elif animal["timer"] % 5 == 0:
                    animal["product_count"] += 1
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        reward_total = 0.0
        events = []
        reward_info = {}
        
        if self._last_action_result and self._last_action_result["success"]:
            # Harvest reward
            if "crops_harvested" in self._last_action_result:
                crops = self._last_action_result["crops_harvested"]
                harvest_reward = crops * self.configs["reward"]["harvest_reward"]
                reward_total += harvest_reward
                events.append("HarvestCrop")
                reward_info["harvest_reward"] = harvest_reward
            
            # Product collection reward
            if "products_collected" in self._last_action_result:
                products = self._last_action_result["products_collected"]
                product_reward = products * self.configs["reward"]["product_reward"]
                reward_total += product_reward
                events.append("CollectProduct")
                reward_info["product_reward"] = product_reward
            
            # Relationship gain reward
            if "relationship_gain" in self._last_action_result:
                gain = self._last_action_result["relationship_gain"]
                relationship_reward = gain * self.configs["reward"]["relationship_reward_multiplier"]
                reward_total += relationship_reward
                events.append("RelationshipGain")
                reward_info["relationship_reward"] = relationship_reward
            
            # Market sale reward
            if "coins_earned" in self._last_action_result:
                coins = self._last_action_result["coins_earned"]
                market_reward = coins * self.configs["reward"]["market_reward_multiplier"]
                reward_total += market_reward
                events.append("MarketSale")
                reward_info["market_reward"] = market_reward
        
        return reward_total, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        # Convert visible tiles to ASCII
        tiles_ascii = ""
        for row in omega["visible_tiles"]:
            row_str = ""
            for tile in row:
                if tile["type"] == "market":
                    row_str += "M "
                elif tile["type"] == "barn":
                    if tile["entity"] == "cow":
                        row_str += "C " if tile["state"] == "sated" else "c "
                    elif tile["entity"] == "sheep":
                        row_str += "S " if tile["state"] == "sated" else "s "
                    elif tile["entity"] == "chicken":
                        row_str += "H " if tile["state"] == "sated" else "h "
                elif tile["type"] == "cottage":
                    row_str += "□ "
                elif tile["type"] == "villager":
                    row_str += "V "
                elif tile["type"] == "crop":
                    if tile["state"] == "seedling":
                        row_str += "· "
                    elif tile["state"] == "growing":
                        row_str += "o "
                    elif tile["state"] == "mature":
                        row_str += "O "
                elif tile["type"] == "soil":
                    row_str += "- "
                elif tile["type"] == "boundary":
                    row_str += "# "
                else:  # grass
                    row_str += ". "
            tiles_ascii += row_str + "\n"
        
        # Agent position marker (center of 5x5 grid)
        lines = tiles_ascii.strip().split("\n")
        if len(lines) >= 3:
            center_line = lines[2]
            lines[2] = center_line[:4] + "@" + center_line[5:]
        tiles_ascii = "\n".join(lines)
        
        # Create a simple namespace object for template formatting
        class SimpleNamespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        agent = SimpleNamespace(
            inventory=SimpleNamespace(
                coins=omega["inventory_coins"],
                seeds=omega["inventory_seeds"],
                water=omega["inventory_water"],
                animal_feed=omega["inventory_animal_feed"],
                crops=omega["inventory_crops"],
                animal_products=omega["inventory_animal_products"],
                gifts=omega["inventory_gifts"]
            )
        )
        
        villagers = SimpleNamespace(
            relationships_rounded=omega["villagers_relationships_rounded"]
        )
        
        return self.configs["skin"]["template"].format(
            t_remaining=omega["t_remaining"],
            agent=agent,
            villagers=villagers,
            tiles_ascii=tiles_ascii
        )    
    def done(self, state=None) -> bool:
        """Check if the environment episode is complete"""
        return self._t >= 50  # Max steps from config
