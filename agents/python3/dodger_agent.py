import numpy as np
from typing import Dict
import random

class DodgerAgent():

    actions = ["up", "down", "left", "right", "bomb", "detonate"]

    agent_id = "a"

    current_state = {}

    def set_agent_id(self, new_id: str):
        self.agent_id = new_id

    def _is_in_bounds(self, location):
        width = self.current_state.get("world").get("width")
        height = self.current_state.get("world").get("height")

        return (location[0] >= 0 & location[0] < width & location[1] >= 0 & location[1] < height)
    
    def _get_surrounding_tiles(self, location):
        tile_north = [location[0], location[1]+1]
        tile_south = [location[0], location[1]-1]
        tile_west = [location[0]-1, location[1]]
        tile_east = [location[0]+1, location[1]]

        surrounding_tiles = [tile_north, tile_south, tile_west, tile_east]

        for tile in surrounding_tiles:
            if not self._is_in_bounds(tile):
                surrounding_tiles.remove(tile)

        return surrounding_tiles
    
    def _is_occupied(self, location):
        entities = self.current_state.get("entities")
        units = self.current_state.get("unit_state")

        list_of_entity_locations = [[entity[c] for c in ['x', 'y']] for entity in entities]
        list_of_unit_locations = [units[u]["coordinates"] for u in ['c','d','e','f','g']]
        list_of_occupied_locations = list_of_entity_locations + list_of_unit_locations

        return location in list_of_occupied_locations
    
    def _get_empty_tiles(self, tiles):
        empty_tiles = []

        for tile in tiles:
            if not self._is_occupied(tile):
                empty_tiles.append(tile)

        return empty_tiles
    
    def _move_to_tile(self, tile, location):
        diff = tuple(x-y for x, y in zip(tile, location))

        if diff == (0,1):
            action = 'up'
        elif diff == (0,-1):
            action = 'down'
        elif diff == (1,0):
            action = 'right'
        elif diff == (-1,0):
            action = 'left'
        else:
            action = ''

        return action
    
    def _get_danger_grid(self):
        w = self.current_state.get("world").get("width")
        h = self.current_state.get("world").get("height")
        grid = np.zeros((h, w))

        entities = self.current_state.get("entities")
        directions = [[0,1], [0, -1], [1, 0], [-1,0]]

        bombs = list(filter(lambda entity: entity.get("type") == "b", entities))
        for bomb in bombs:
            bomb_x, bomb_y = bomb.get("x"), bomb.get("y")
            bomb_blast_diameter = bomb.get("blast_diameter")
            radius = round((bomb_blast_diameter - 1) / 2)

            for direction in directions:
                for d in range(1, radius + 1):
                    new_x, new_y = bomb_x + direction[0]*d, bomb_y + direction[1]*d
                    if self._is_occupied([new_x, new_y]) or not self._is_in_bounds([new_x, new_y]):
                        break
                    else:
                        grid[new_x, new_y] += radius - d

        return grid
    
    def get_actions(self, raw_state: Dict):

        self.current_state = raw_state

        my_units = raw_state.get("agents").get(self.agent_id).get("unit_ids")

        # update danger grid
        danger_grid = self._get_danger_grid()

        res = {}

        for unit_id in my_units:

            # 3% of the time they'll place a bomb
            # Essentially, a bomb is expected every 3.3 seconds
            randint = random.randrange(0,100)
            
            if randint < 3:
                action = "bomb"
            else:
                # this unit's location
                unit_location = raw_state["unit_state"][unit_id]["coordinates"]   

                # get our surrounding tiles
                surrounding_tiles = self._get_surrounding_tiles(unit_location)

                # get list of empty tiles around us
                empty_tiles = self._get_empty_tiles(surrounding_tiles)

                if empty_tiles:
                    # if in danger... go to biggest danger decrease
                    if danger_grid[unit_location[0]][unit_location[1]] > 0:
                        min_danger = 999
                        chosen_tile = unit_location
                        for tile in empty_tiles:
                            if danger_grid[tile] < min_danger:
                                chosen_tile = tile
                        action = self._move_to_tile(chosen_tile, unit_location)
                    else:
                        # choose an empty tile to walk to
                        random_tile = random.choice(empty_tiles)
                        action = self._move_to_tile(random_tile, unit_location)
                else:
                    # we're trapped
                    action = 'nothing'

            res[unit_id] = action

        return res