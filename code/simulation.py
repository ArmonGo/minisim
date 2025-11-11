
from collections import deque

def check_state(sim):
        car_num = len(sim.cars)
        car_state = {cid: car.pos for cid, car in sim.cars.items()}
        return car_state, car_num

def count_tiles(grid):
    roads = [pos for pos, tile in grid.items() if tile == 'ROAD']
    return len(roads)

def summary_road_usage(sim, keys):
    summary = sim.road_stats_summary()
    value_by_pos = {}
    for row in summary:
        value_by_pos[row["pos"]] = {"pos" : row["pos"]
                               }
        update_keys = { key : float(row.get(key, 0.0)) for key in keys}
        value_by_pos[row["pos"]].update(update_keys)
    return value_by_pos

def are_all_depots_connected(grid):
    """
    Check if every 'depot_in' is connected to at least one 'depot_out'
    via adjacent 'road' tiles (4-directional movement).
    
    Parameters:
        grid (dict): Dictionary mapping (x, y) -> tile_type (str)
    
    Returns:
        bool: True if all depot_in are connected to some depot_out, False otherwise.
    """
    # Extract positions
    depot_ins = [pos for pos, tile in grid.items() if tile == 'DEPOT_IN']
    depot_outs = set(pos for pos, tile in grid.items() if tile == 'DEPOT_OUT')
    
    if not depot_ins:
        return True  # No depot_in to connect
    if not depot_outs:
        return False  # depot_in exists but no depot_out
    
    # Directions for 4-connected grid
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def is_connected(start):
        """BFS from start (a depot_in) to see if any depot_out is reachable."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                # Skip if already visited
                if neighbor in visited:
                    continue
                
                # If neighbor is a depot_out, we're done
                if neighbor in depot_outs:
                    return True
                
                # Only move through 'road' tiles
                if grid.get(neighbor) == 'ROAD':
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False

    # Check every depot_in
    non_connect_depot_in = []
    for depot_in in depot_ins:
        if not is_connected(depot_in):
            non_connect_depot_in.append(depot_in)
    return non_connect_depot_in

def simulation(sim):
    # get the grid 
    grid = {}
    # add map 
    for y in range(sim.h):
        for x in range(sim.w):
            p = (x, y)
            if sim.grid[x][y].name =='DEPOT':
                depot_type =  sim.depots[sim.depot_at[p]].kind
                grid[p] = sim.grid[x][y].name + '_' + depot_type.upper()
            else:
                grid[p] = sim.grid[x][y].name
    # first check if all deport in are connected 
    non_connect_depot_locations = are_all_depots_connected(grid)
    connect_comment = ''
    if len(non_connect_depot_locations) ==0 :
        connect_comment ='All DEPOT_IN are connected at least with one DEPOT_OUT.'
    else:
        connect_comment = f'DEPOT_IN at {non_connect_depot_locations} are not connected to any DEPOT_OUT. Please try to connect it to other DEPOT_OUT!'
    # count the road tiles 
    num_road = count_tiles(grid)
    # check the simualtion results 
    last_car_states, last_car_num = check_state(sim)
    simulation_comment = ''
    steps = 1000
    simulation_comment = f'all {steps} steps are done, and need to extend the simulation steps!'
    for s in range(steps):
        sim.step()
        current_car_states, current_car_num = check_state(sim)
        if sim.total_demand() == 0:
            simulation_comment = 'All DEPOT_IN (demand depots) are fully supplied.'
            break
        elif last_car_states == current_car_states or current_car_num == 0:
            simulation_comment = f"""There are still {sim.total_demand()} demand need to be supplied, 
                                    {sim.total_supply()} left in the DEPOT_OUT, 
                                    and {current_car_num} cars stuck on the way."""
            break
        last_car_states = current_car_states
    keys = ['occupancy_ratio', 'flow_per_tick', 'entries', 'blocked']
    road_flow_comment = summary_road_usage(sim, keys)
    sim_conclusion = f"""DEPOT connection:  {connect_comment} 
                         number of road tiles: {num_road}
                         road flow usage: {road_flow_comment}
                         travel ticks: {s}
                         demand and supply: {simulation_comment}
                     """
    # reset the sim world
    sim.reset_stats()
    return sim_conclusion