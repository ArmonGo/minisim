
import sys
from minisim import import_map

# -------------------------
# Config / cost mapping
# -------------------------
import json
import argparse
from itertools import product
from collections import defaultdict
import pulp
from contextlib import redirect_stdout
import io

# ---------- CONFIG ----------
TERRAIN_BUILD_COST = {
    'GRASS': 10,        # road
    'WATER': 30,        # bridge
    'MOUNTAIN': 100,    # tunnel
    # we force depots built; set cost 0 here to avoid double-charging
    'DEPOT_OUT': 0,
    'DEPOT_IN': 0,
}
TRAVERSABLE = set(['GRASS', 'WATER', 'MOUNTAIN', 'DEPOT_OUT', 'DEPOT_IN'])
NEIGHBORS = [(1,0),(-1,0),(0,1),(0,-1)]

# -------------------------
# Helper functions
# -------------------------
def get_grid_world(sim) -> dict:
    """
    summarise the grid world in form of dictionary
    return: 
        {type, pos, amount}
    """
    # define the grid tiles 
    grid = {}
    # add map 
    for y in range(sim.h):
        for x in range(sim.w):
            p = (x, y)
            tile = {}
            tile["pos"] = p
            tile["amount"] = "not applicable"

            if sim.grid[x][y].name =='DEPOT':
                depot_type =  sim.depots[sim.depot_at[p]].kind
                tile["type"] = sim.grid[x][y].name + '_' + depot_type.upper()
                tile["amount"] = sim.depots[sim.depot_at[p]].amount
            else:
                tile["type"] = sim.grid[x][y].name
            grid[p] = tile
    return grid


def neighbors(pos):
    x,y = pos
    for dx,dy in NEIGHBORS:
        yield (x+dx, y+dy)

def optimize_multisupply(grid, budget, time_limit_sec=300, verbose=False):
    # --- index sets
    nodes = [n for n in grid if grid[n]['type'] in TRAVERSABLE]
    node_set = set(nodes)
    supplies = [n for n in nodes if grid[n]['type']=='DEPOT_OUT']
    demands  = [n for n in nodes if grid[n]['type']=='DEPOT_IN']

    supply_amount = {s: int(grid[s].get('amount',0)) for s in supplies}
    demand_amount  = {d: int(grid[d].get('amount',0)) for d in demands}
    total_supply = sum(supply_amount.values())
    total_demand = sum(demand_amount.values())

    if total_demand == 0:
        return {'status':'no_demand', 'reason':'No DEPOT_IN found with non-zero demand.'}

    if total_supply < total_demand:
        # still allow solve but return infeasible note
        print(f"Warning: total_supply ({total_supply}) < total_demand ({total_demand}) - problem may be infeasible")

    # directed edges between adjacent traversable nodes
    edges = [(u,v) for u in nodes for v in neighbors(u) if v in node_set]

    # Big-M for capacity coupling (max flow on an edge ≤ BIGM)
    BIGM = max(1, total_supply)

    # --- Build pulp problem
    prob = pulp.LpProblem("multisupply_network_design", pulp.LpMinimize)

    # build variables per node
    y = {n: pulp.LpVariable(f"y_{n[0]}_{n[1]}", cat='Binary') for n in nodes}
    # force depots built
    for n in supplies + demands:
        prob += y[n] == 1

    # flow variables: one commodity per supply s, for each directed edge (u,v)
    f = {(u,v): pulp.LpVariable(f"f_{u}_{v}", lowBound=0, cat='Continuous') for (u,v) in edges}

    # --- Constraints

    # Net supply at each node: +k for DEPOT_OUT, -k for DEPOT_IN, else 0
    b = {n: 0 for n in nodes}
    for s in supplies:
        b[s] += supply_amount[s]
    for d in demands:
        b[d] -= demand_amount[d]

    # 1) Flow conservation (single commodity)
    for n in nodes:
        out_arcs = [f[(n,v)] for (a,v) in edges if a == n]
        in_arcs  = [f[(u,n)] for (u,b_) in edges if b_ == n]
        prob += pulp.lpSum(out_arcs) - pulp.lpSum(in_arcs) == b[n], f"conserve_{n}"

    # no outflow from demands
    for d in demands:
        prob += pulp.lpSum(f[(d,v)] for (u,v) in edges if u == d) == 0, f"no_out_from_demand_{d}"
    # no inflow to supplies
    for s in supplies:
        prob += pulp.lpSum(f[(u,s)] for (u,v) in edges if v == s) == 0, f"no_in_to_supply_{s}"

    # 2) Edge capacity coupling with build decisions (needs both endpoints built)
    for (u,v) in edges:
        prob += f[(u,v)] <= BIGM * y[u], f"edge_cap_u_{u}_{v}"
        prob += f[(u,v)] <= BIGM * y[v], f"edge_cap_v_{u}_{v}"

    # 3) Budget constraint – sum of node build costs <= budget
    total_build_cost = pulp.lpSum(TERRAIN_BUILD_COST.get(grid[n]['type'], 1e6) * y[n] for n in nodes)
    prob += total_build_cost <= budget, "budget_constraint"

    # Objective: minimize total build cost (add tiny flow-length penalty if you want tie-breaking)
    prob += total_build_cost, "min_total_build_cost"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit_sec)
    status_ = prob.solve(solver)
    status_str = pulp.LpStatus[prob.status]
    sol_status_str = pulp.LpSolution[prob.sol_status]

    # Extract solution
    raw_nodes = {n: pulp.value(y[n]) for n in nodes}
    built_nodes = [n for n in nodes if pulp.value(y[n]) is not None and pulp.value(y[n]) > 0.5]
    flows = {}
    # store flows keyed by (s, u, v)
    for (u,v), var in f.items():
        val = pulp.value(var)
        if val is not None and val > 1e-8:
            flows[(u,v)] = float(val)

    # compute per-supply used, per-demand received (sanity)
    used_by_supply = {s: 0.0 for s in supplies}
    received_by_demand = {d: 0.0 for d in demands}
    for (u,v), val in flows.items():
        if u in supplies:
            used_by_supply[u] += val
        if v in demands:
            received_by_demand[v] += val


    return {
        'status': sol_status_str,
        'raw_nodes': raw_nodes,
        'built_nodes': built_nodes,
        'flows': flows,
        'total_cost': float(pulp.value(total_build_cost)),
        'used_by_supply': used_by_supply,
        'received_by_demand': received_by_demand,
        'total_supply': total_supply,
        'total_demand': total_demand,
    }

def dump_solution(res, out_prefix='solution'):
    # Save built nodes and flows to json, robustly formatting tuple keys
    built = res.get('built_nodes', [])
    flows = res.get('flows', {})
    state = res.get('status')
    with open(out_prefix + '_built.json', 'w') as f:
        json.dump({'built': [list(n) for n in built], 'total_cost': res.get('total_cost'), 'state': state}, f, indent=2)

    # flows -> list of dicts
    flows_list = []
    for (u,v), val in flows.items():
        flows_list.append({
            'from': list(u),
            'to': list(v),
            'flow': val
        })
    with open(out_prefix + '_flows.json', 'w') as f:
        json.dump(flows_list, f, indent=2)

    print('Saved', out_prefix + '_built.json', 'and', out_prefix + '_flows.json')

# ---------------- CLI ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='the targeted map file')
    parser.add_argument('--budget', type=float, required=True, help='budget for building (total cost)')
    parser.add_argument('--time_limit', type=int, default=20, help='CBC time limit seconds')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    sim = import_map(f'./map_collection/{args.file}.pkl')
    grid = get_grid_world(sim)
    res = optimize_multisupply(grid, args.budget, time_limit_sec=args.time_limit, verbose=args.verbose)

    print(res.get('status'))
    print('Total build cost:', res.get('total_cost'))
    print('Built tiles (count):', len(res.get('built_nodes')))
    print('Built tile coordinates:', res.get('built_nodes'))
    print('Number of positive flow arcs:', len(res.get('flows')))
    dump_solution(res, out_prefix=f'./results/solver/{args.file}_solution')
