import sys
sys.path.append("/Users/didi/Documents/GitHub/AutoEnv")
from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class FiveByFiveObs(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        agent_x, agent_y = agent_pos
        
        # Extract 5x5 grid centered on agent
        visible_tiles = []
        for dy in range(-2, 3):
            row = []
            for dx in range(-2, 3):
                x, y = agent_x + dx, agent_y + dy
                if 0 <= x < 15 and 0 <= y < 15:
                    tile_info = self._get_tile_info(env_state, x, y)
                    row.append(tile_info)
                else:
                    row.append({"type": "boundary", "entity": None, "state": None})
            visible_tiles.append(row)
        
        # Round villager relationships to nearest 5
        relationships_rounded = {}
        for villager in env_state["objects"]["villagers"]:
            vid = villager["id"]
            rel = villager["relationship"]
            relationships_rounded[vid] = round(rel / 5) * 5
        
        return {
            "visible_tiles": visible_tiles,
            "agent_pos": agent_pos,
            "inventory_seeds": env_state["agent"]["inventory"]["seeds"],
            "inventory_water": env_state["agent"]["inventory"]["water"],
            "inventory_crops": env_state["agent"]["inventory"]["crops"],
            "inventory_animal_feed": env_state["agent"]["inventory"]["animal_feed"],
            "inventory_animal_products": env_state["agent"]["inventory"]["animal_products"],
            "inventory_gifts": env_state["agent"]["inventory"]["gifts"],
            "inventory_coins": env_state["agent"]["inventory"]["coins"],
            "villagers_relationships_rounded": relationships_rounded,
            "t_remaining": 50 - t
        }
    
    def _get_tile_info(self, env_state: Dict[str, Any], x: int, y: int) -> Dict[str, Any]:
        # Check for market
        market = env_state["objects"]["market"]
        if market and market[0] == x and market[1] == y:
            return {"type": "market", "entity": "market", "state": None}
        
        # Check for barns and animals
        for barn in env_state["objects"]["barns"]:
            bx, by = barn["pos"]
            if bx == x and by == y:
                animal = barn["animal"]
                hunger_state = "hungry" if animal["hungry"] else "sated"
                return {"type": "barn", "entity": animal["species"], "state": hunger_state}
        
        # Check for cottages and villagers
        for cottage in env_state["objects"]["cottages"]:
            cx, cy = cottage["pos"]
            if cx == x and cy == y:
                return {"type": "cottage", "entity": "cottage", "state": None}
        
        for villager in env_state["objects"]["villagers"]:
            vx, vy = villager["pos"]
            if vx == x and vy == y:
                return {"type": "villager", "entity": f"villager_{villager['id']}", "state": villager["mood"]}
        
        # Check for crops
        for field in env_state["objects"]["crop_fields"]:
            fx, fy = field["pos"]
            if fx == x and fy == y:
                crop = field["crop"]
                crop_state = crop["stage"] if crop["stage"] != "empty" else None
                tile_type = "soil" if crop["stage"] == "empty" else "crop"
                return {"type": tile_type, "entity": crop_state, "state": crop_state}
        
        # Default grass tile
        return {"type": "grass", "entity": None, "state": None}