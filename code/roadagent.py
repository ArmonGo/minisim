import os 
from agents import Agent, RunContextWrapper, function_tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple, Literal, Optional
from collections import deque
from minisim import TrafficSim
from simulation import are_all_depots_connected, count_tiles, check_state, summary_road_usage


class RoadPosition(BaseModel):
    x: int
    y: int


@function_tool
async def get_grid_world(wrapper: RunContextWrapper[TrafficSim]) -> dict:
    """
    summarise the grid world in form of dictionary
    return: 
        {type, pos, amount}
    """
    # define the grid tiles 
    grid = {}
    # add map 
    for y in range(wrapper.context.h):
        for x in range(wrapper.context.w):
            p = (x, y)
            tile = {}
            tile["pos"] = p
            tile["amount"] = "not applicable"

            if wrapper.context.grid[x][y].name =='DEPOT':
                depot_type =  wrapper.context.depots[wrapper.context.depot_at[p]].kind
                tile["type"] = wrapper.context.grid[x][y].name + '_' + depot_type.upper()
                tile["amount"] = wrapper.context.depots[wrapper.context.depot_at[p]].amount
            else:
                tile["type"] = wrapper.context.grid[x][y].name
            grid[p] = tile
    wrapper.context.log[-1].append({'get_grid_world' : None })
    return grid

@function_tool
async def add_roads(wrapper: RunContextWrapper[TrafficSim], pl : List[RoadPosition]) -> List[RoadPosition]:
    """add road tiles to the grid world, returns the tiles which were placed"""
    done = []
    for p in pl:
        if wrapper.context.add_road((p.x, p.y)) :
            done.append(p)
    wrapper.context.log[-1].append({'add_roads' : {'input' : pl, 'output' : done} })
    return done

@function_tool
async def remove_roads(wrapper: RunContextWrapper[TrafficSim], pl : List[RoadPosition]) -> List[RoadPosition]:
    """remove road tiles to the grid world, returns the tiles which were removed"""
    done = []
    for p in pl:
        if wrapper.context.remove_road((p.x, p.y)) :
            done.append(p)
    wrapper.context.log[-1].append({'remove_roads' : {'input' : pl, 'output' : done} })
    return done

@function_tool
async def add_bridges(wrapper: RunContextWrapper[TrafficSim], pl : List[RoadPosition]) -> List[RoadPosition]:
    """add bridge tiles to the grid world, returns the tiles which were placed"""
    done = []
    for p in pl:
        if wrapper.context.add_bridge((p.x, p.y)) :
            done.append(p)
    wrapper.context.log[-1].append({'add_bridges' : {'input' : pl, 'output' : done} })
    return done

@function_tool
async def remove_bridges(wrapper: RunContextWrapper[TrafficSim], pl : List[RoadPosition]) -> List[RoadPosition]:
    """remove bridge tiles to the grid world, returns the tiles which were removed"""
    done = []
    for p in pl:
        if wrapper.context.remove_bridge((p.x, p.y)) :
            done.append(p)
    wrapper.context.log[-1].append({'remove_bridges' : {'input' : pl, 'output' : done} })
    return done


@function_tool
async def add_tunnels(wrapper: RunContextWrapper[TrafficSim], pl : List[RoadPosition]) -> List[RoadPosition]:
    """add tunnel tiles to the grid world, returns the tiles which were placed"""
    done = []
    for p in pl:
        if wrapper.context.add_tunnel((p.x, p.y)) :
            done.append(p)
    wrapper.context.log[-1].append({'add_tunnels' : {'input' : pl, 'output' : done} })
    return done

@function_tool
async def remove_tunnels(wrapper: RunContextWrapper[TrafficSim], pl : List[RoadPosition]) -> List[RoadPosition]:
    """remove tunnel tiles to the grid world, returns the tiles which were removed"""
    done = []
    for p in pl:
        if wrapper.context.remove_tunnel((p.x, p.y)) :
            done.append(p)
    wrapper.context.log[-1].append({'remove_tunnels' : {'input' : pl, 'output' : done} })
    return done


@function_tool
async def run_simulation(wrapper: RunContextWrapper[TrafficSim]):
    """return the simulation results including DEPOT connection, 
     number of road tiles, road flow usage, travel ticks, and left demand and left supply. 
    """
    # get the grid 
    grid = {}
    # add map 
    for y in range(wrapper.context.h):
        for x in range(wrapper.context.w):
            p = (x, y)
            if wrapper.context.grid[x][y].name =='DEPOT':
                depot_type =  wrapper.context.depots[wrapper.context.depot_at[p]].kind
                grid[p] = wrapper.context.grid[x][y].name + '_' + depot_type.upper()
            else:
                grid[p] = wrapper.context.grid[x][y].name
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
    last_car_states, last_car_num = check_state(wrapper.context)
    simulation_comment = ''
    steps = 1000
    simulation_comment = f'all {steps} steps are done, and need to extend the simulation steps!'
    for s in range(steps):
        wrapper.context.step()
        current_car_states, current_car_num = check_state(wrapper.context)
        if wrapper.context.total_demand() == 0:
            simulation_comment = 'All DEPOT_IN (demand depots) are fully supplied.'
            break
        elif last_car_states == current_car_states or current_car_num == 0:
            simulation_comment = f"""There are still {wrapper.context.total_demand()} demand need to be supplied, 
                                    {wrapper.context.total_supply()} left in the DEPOT_OUT, 
                                    and {current_car_num} cars stuck on the way."""
            break
        last_car_states = current_car_states
    keys = ['occupancy_ratio', 'flow_per_tick', 'entries', 'blocked']
    road_flow_comment = summary_road_usage(wrapper.context, keys)
    budget_comment = 'No budget limitation.'
    if wrapper.context.open_budget: 
        budget_comment = f"""Intial budget is {wrapper.context.initial_budget}
                             Left budget is {wrapper.context.budget}
                             Used budget is {wrapper.context.initial_budget-wrapper.context.budget}"""
    sim_conclusion = f"""DEPOT connection:  {connect_comment} 
                         number of road tiles: {num_road}
                         road flow usage: {road_flow_comment}
                         travel ticks: {s}
                         demand and supply: {simulation_comment}
                         budget: {budget_comment}
                     """
    wrapper.context.log[-1].append({'run_simulation' : sim_conclusion })
    # reset the sim world
    wrapper.context.reset_stats()
    return sim_conclusion


# ----- Output -----
# --- Position class ---
class RoadPosition(BaseModel):
    x: int = Field(description="x-coordinate in the grid.")
    y: int = Field(description="y-coordinate in the grid.")


class RoadAction(BaseModel):
        tool:  Literal["add_roads",
                        "remove_roads",
                        "add_bridges", 
                        "remove_bridges",
                        "add_tunnels", 
                        "remove_tunnels",
                        "run_simulation",
                        "get_grid_world"
                        ] = Field(description="Which operation to perform.")
        location: Optional[List[RoadPosition]] = Field(
        default=None,
        description=(
            "Grid coordinate where the operation applies. "
            "Required for add_roads and remove_roads; omit for other tools."
        )
    )

# --- Full output class ---
class RoadPlannerOutput(BaseModel):
    actions: Optional[List[RoadAction]] = Field(
        default=None,
        description=(
            "List of tool actions executed or planned by the agent. "
            "Each action may or may not include a location depending on the tool."
        )
    )
    explanation: str = Field(description="Reasoning behind the planning decision.")



def get_agent(prompt, name = "road planner", model_name ="gpt-5-mini", tool_list = [ get_grid_world, add_roads, remove_roads, run_simulation]):
    agent = Agent(
        name=name,
        instructions=prompt,
        model=model_name, 
        tools = tool_list,
        output_type = RoadPlannerOutput
    )
    return agent 



def generate_prompt(initial_budget, with_water = False, with_mountain = False, with_bridge = False, with_tunnel = False):
    prompt_intro = f"""
    You are a road planner agent operating in a simulated grid world.
    The grid world consists of different grid types.
    Your objective is to design an efficient road network connecting SUPPLY depots to DEMAND depots{' with limited budget ' + str(initial_budget) if initial_budget is not None else ''}.
    Please use the given tools to execute your plan immediately and DO NOT ask for confirmation."""
    prompt_world_des = f"""

    World Description:
        Each grid cell belongs to one of the following types:
        * GRASS: empty buildable land
        * ROAD: an existing road tile, can only be built on grass tile.
        * DEPOT_IN: input depot (demand) 
        * DEPOT_OUT: output depot (supply)"""
    if with_water: prompt_world_des  = prompt_world_des + """
        * WATER: an existing water tile"""
    if with_mountain: prompt_world_des  = prompt_world_des + """
        * MOUNTAIN: an existing mountain tile"""
    if with_bridge: prompt_world_des  = prompt_world_des + """
        * BRIDGE: an existing bridge tile, can only be built on water tile."""
    if with_tunnel: prompt_world_des  = prompt_world_des + """
        * TUNNEL: an existing tunnel tile, can only be built on mountain tile."""
    available_grid_type = 'ROAD' 
    if with_bridge: available_grid_type = 'ROAD, BRIDGE'
    if with_tunnel: available_grid_type = 'ROAD, TUNNEL'
    if with_bridge and with_tunnel: available_grid_type = 'ROAD, BRIDGE, TUNNEL'
    prompt_rules_goals = f"""
    
    Rules and Goals

        * depots 
            * Each depot has its own demand or supply value and the values might be different. Ensure that all DEPOT_IN (demand depots) are fully supplied by building roads that connect DEPOT_OUT (supply depots) to DEPOT_IN.
            * Careful do not use DEPOT as road path, they are only endpoints. All DEPOT_OUT can only deliver via {available_grid_type} to DEPOT_IN without any other DEPOT (both in and out) on the way to block it. 
        * simulation
            * You can use tool run_simulation() to simulate and receive the feedback regarding the unconnected DEPOT_IN, the used travel ticks, the values of left supply and demand, the number of used road tiles, and the usage flow of road. 
            * After each simulation, adjust your plan accordingly to reduce travel ticks and unsatisfied demand or supply.
            * After each simulation, the supply and demand will be reset. So you can always start from scratch.
        * tool using:
            * You can use each tool for multiple times and adjust your design by adding or removing tiles. 
            * Use your tools, and DO NOT ask for confirmation, execute as your plan, and return the final map with tiles you have built. """
    if initial_budget is not None:
        prompt_rules_goals = prompt_rules_goals +  f"""
        * budget:
            * You have initial budget of {initial_budget}.
            * Each tile has its own cost, ROAD costs 10, BRIDGE costs 30, and TUNNEL costs 100. Please use those tile within given budget. 
            * Adding tiles will use your budget, remove tiles will return the budget. So you can always replan.  """
    passable = """
        * passable rules:
            * Roads can only pass through GRASS or existing ROAD tiles."""
    if with_bridge: 
        passable = """
        * passable rules:
            * Roads can only pass through GRASS or existing ROAD tiles.
            * Bridge can only pass through WATER or existing BRIDGE tiles."""
    if with_tunnel:
        passable = """
        * passable rules:
            * Roads can only pass through GRASS or existing ROAD tiles.
            * Bridge can only pass through WATER or existing BRIDGE tiles.
            * Tunnel can only pass through MOUNTAIN or existing TUNNEL tiles."""
    goal = """
        * goal:
            * Try to design the path efficiently and remove the uncessary tiles.
            * Make sure all demands can be supplied. """
    prompt_rules_goals = prompt_rules_goals + passable + goal
    prompt_tool = """

    Available Tools:

        * get_grid_world(): returns the current grid layout and type of each cell.
        * run_simulation(): get the simulation results, and the grid world will be reset after each simulation. 
        * add_roads(locations): build a sequence of road tiles with given grid coordinates.
        * remove_roads(locations): remove a sequence of road tiles with given grid coordinates (turns it back into grass)."""
    if with_bridge:
        prompt_tool = prompt_tool + """ 
        * add_bridges(locations): build a sequence of bridge tiles with given grid coordinates.
        * remove_bridges(locations): remove a sequence of bridge tiles with given grid coordinates (turns it back into water)."""
    if with_tunnel:
        prompt_tool = prompt_tool +"""
        * add_tunnels(locations): build a sequence of tunnel tiles with given grid coordinates.
        * remove_tunnels(locations): remove a sequence of tunnel tiles with given grid coordinates (turns it back into mountain)."""

    prompt_output = """

    Output Requirements:

        * Explain your reasoning for your road design.
        * Propose actions using the available tools."""
    prompt = prompt_intro + prompt_world_des + prompt_rules_goals + prompt_tool + prompt_output
    return prompt
