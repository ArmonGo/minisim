from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
import random
import collections
import copy 
import os
from PIL import Image, ImageDraw
import pickle

Pos = Tuple[int, int]
DIR4 = [(1,0), (-1,0), (0,1), (0,-1)]
BUDGET_MAP = {'road' : 10, 
              'bridge' : 30, 
              'tunnel' : 100}

class TileType(Enum):
    GRASS = 0
    WATER = 1
    ROAD  = 2
    DEPOT = 3
    MOUNTAIN = 4
    BRIDGE = 5
    TUNNEL = 6

@dataclass
class Depot:
    id: int
    kind: str   # "out" or "in"
    amount: int # remaining supply if out, remaining demand if in
    pos: Pos

@dataclass
class Car:
    id: int
    pos: Pos
    target_depot_id: Optional[int] = None
    path: List[Pos] = field(default_factory=list)  # next steps (excluding current pos)


def load_tile_images(folder: str, scale: int = 16) -> dict[str, list[Image.Image]]:
        """
        Load and group tile images from a folder.

        Expected naming: grass_1.png, road_2.png, water_1.png, etc.
        Returns dict like { "grass": [PIL.Image, PIL.Image, ...], "road": [...] }
        """
        tile_images: dict[str, list[Image.Image]] = {}
        for fn in os.listdir(folder):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            base = fn.split("_")[0].lower()
            path = os.path.join(folder, fn)
            try:
                img = Image.open(path).convert("RGBA").resize((scale, scale), Image.LANCZOS)
            except Exception as e:
                print(f"Failed to load {fn}: {e}")
                continue
            tile_images.setdefault(base, []).append(img)
        return tile_images
        
def import_map(path):
    with open(path, 'rb') as f:
        map_elements = pickle.load(f)
    map_params = map_elements['map_params']
    base_sim =TrafficSim(**map_params)
    # add depots
    base_sim.depots = map_elements['depots'] 
    base_sim.depot_at = map_elements['depot_at'] 
    base_sim.init_depots = map_elements['init_depots']
    # add layout 
    base_sim.grid = map_elements['grids'] 
    # add roads 
    base_sim.road_stats = map_elements['road_stats'] 
    return base_sim

class TrafficSim:
    """
    Simple grid traffic-flow simulation.

    Rules:
      - World is grass/water/mountain initially with random rivers, pools, and mountain regions.
      - Roads can be built on grass only.
      - Bridges can be built on water only.
      - Tunnels can be built on mountain only.
      - Depots can be built on grass only.
      - Cars can pass road, bridge, and tunnel (all tracked in road_stats).
      - One car per traversable tile per tick (exclusive occupancy).
    """

    def __init__(self, width: int, height: int, seed: Optional[int] = None, initial_budget = None, with_water = False , with_mountain = False, export_p='./exported_map.pkl', load_p=None):
        self.w, self.h, self.seed = width, height, seed
        self.rng = random.Random(self.seed)
        self.grid: List[List[TileType]] = [[TileType.GRASS for _ in range(self.h)] for _ in range(self.w)]
        self.depot_at: Dict[Pos, int] = {}  # pos -> depot_id
        self.depots: Dict[int, Depot] = {}
        self.init_depots: Dict[int, Depot] = {}
        self.next_depot_id = 1
        self.road_stats: Dict[Pos, Dict[str, int]] = {}  # per-road/bridge/tunnel metrics

        self.cars: Dict[int, Car] = {}
        self.next_car_id = 1
        self.with_water = with_water
        self.with_mountain = with_mountain

        # Occupancy for ROAD, BRIDGE, TUNNEL tiles (pos of cars)
        self.occupied: Set[Pos] = set()
        self.tick_count = 0
        self.export_p = export_p
        self.load_p = load_p
        
        # Set budget 
        self.budget = initial_budget
        self.initial_budget = initial_budget
        self.open_budget = False
        if self.budget is not None:
            self.open_budget = True
        if with_water:
            self._generate_water()
        if with_mountain:
            self._generate_mountains()
        self.log = [[]]

    def _reset_budget(self, budget):
         # Set budget 
        self.budget = budget
        self.initial_budget = budget
        self.open_budget = False
        if self.budget is not None:
            self.open_budget = True

    def _ensure_road_stats(self, p: Pos):
        if p not in self.road_stats:
            self.road_stats[p] = {
                "entries": 0,          # times a car entered this tile
                "exits": 0,            # times a car left this tile
                "ticks_occupied": 0,   # how many ticks this tile was occupied
                "blocked": 0,          # times cars contended for this tile and lost
                "created_at": self.tick_count,
                "removed_at": -1,      # -1 = still present
            }
    # ---------- Budget control ----------

    def can_afford(self, cost: int) -> bool:
        return self.budget >= cost

    def spend(self, cost: int) -> bool:
        if self.can_afford(cost):
            self.budget -= cost
            return True
        return False

    def refund(self, amount: int):
        self.budget += amount

    # ---------- World generation ----------
    def _in_bounds(self, p: Pos) -> bool:
        x,y = p
        return 0 <= x < self.w and 0 <= y < self.h

    def _neighbors4(self, p: Pos):
        x,y = p
        for dx,dy in DIR4:
            q = (x+dx, y+dy)
            if self._in_bounds(q):
                yield q

    def _generate_water(self):
        pool_prob = 0.05 # 0.05
        for x in range(self.w):
            for y in range(self.h):
                if self.rng.random() < pool_prob:
                    self.grid[x][y] = TileType.WATER

        rivers = max(1, (self.w * self.h) // 800)
        for _ in range(rivers):
            edge = self.rng.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                p = (self.rng.randrange(self.w), 0)
                direction = (0, 1)
            elif edge == 'bottom':
                p = (self.rng.randrange(self.w), self.h-1)
                direction = (0, -1)
            elif edge == 'left':
                p = (0, self.rng.randrange(self.h))
                direction = (1, 0)
            else:
                p = (self.w-1, self.rng.randrange(self.h))
                direction = (-1, 0)

            length = self.w + self.h
            for _ in range(length * 2):
                if not self._in_bounds(p):
                    break
                x,y = p
                self.grid[x][y] = TileType.WATER
                for q in self._neighbors4(p):
                    if self.rng.random() < 0.25:
                        self.grid[q[0]][q[1]] = TileType.WATER
                if self.rng.random() < 0.3:
                    direction = self.rng.choice(DIR4)
                p = (p[0] + direction[0], p[1] + direction[1])

    def _generate_mountains(self):
        # Generate mountain clusters
        num_clusters = max(3, (self.w * self.h) // 300)
        for _ in range(num_clusters):
            if self.rng.random() < 0.4:  # 40% chance to spawn a cluster
                start_x = self.rng.randrange(self.w)
                start_y = self.rng.randrange(self.h)
                p = (start_x, start_y)
                if self.grid[p[0]][p[1]] != TileType.GRASS:
                    continue
                cluster_size = self.rng.randint(3, 25)
                frontier = [p]
                placed = set()
                for _ in range(cluster_size):
                    if not frontier:
                        break
                    cur = frontier.pop()
                    if cur in placed or not self._in_bounds(cur):
                        continue
                    placed.add(cur)
                    self.grid[cur[0]][cur[1]] = TileType.MOUNTAIN
                    for q in self._neighbors4(cur):
                        if q not in placed and self.grid[q[0]][q[1]] == TileType.GRASS:
                            if self.rng.random() < 0.6:
                                frontier.append(q)

        # Add sparse sparkle mountains
        sparkle_prob = 0.005
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y] == TileType.GRASS and self.rng.random() < sparkle_prob:
                    self.grid[x][y] = TileType.MOUNTAIN

    # ---------- Editing ----------
    def can_place_road(self, p: Pos) -> bool:
        if not self._in_bounds(p): return False
        return self.grid[p[0]][p[1]] == TileType.GRASS

    def add_road(self, p: Pos) -> bool:
        if not self.can_place_road(p): return False
        if self.open_budget:
            if not self.spend(BUDGET_MAP['road']): return False 
        self.grid[p[0]][p[1]] = TileType.ROAD
        self._ensure_road_stats(p)
        self.road_stats[p]["removed_at"] = -1
        return True

    def remove_road(self, p: Pos) -> bool:
        if not self._in_bounds(p): return False
        if p in self.occupied:
            return False
        if self.grid[p[0]][p[1]] == TileType.ROAD:
            self.grid[p[0]][p[1]] = TileType.GRASS
            self._ensure_road_stats(p)
            self.road_stats[p]["removed_at"] = self.tick_count
            if self.open_budget:
                self.refund(BUDGET_MAP['road'])
            return True
        return False

    def can_place_bridge(self, p: Pos) -> bool:
        if not self._in_bounds(p): return False
        return self.grid[p[0]][p[1]] == TileType.WATER

    def add_bridge(self, p: Pos) -> bool:
        if not self.can_place_bridge(p): return False
        if self.open_budget:
            if not self.spend(BUDGET_MAP['bridge']): return False 
        self.grid[p[0]][p[1]] = TileType.BRIDGE
        self._ensure_road_stats(p)
        self.road_stats[p]["removed_at"] = -1
        return True

    def remove_bridge(self, p: Pos) -> bool:
        if not self._in_bounds(p): return False
        if p in self.occupied:
            return False
        if self.grid[p[0]][p[1]] == TileType.BRIDGE:
            self.grid[p[0]][p[1]] = TileType.WATER
            self._ensure_road_stats(p)
            self.road_stats[p]["removed_at"] = self.tick_count
            if self.open_budget:
                self.refund(BUDGET_MAP['bridge'])
            return True
        return False

    def can_place_tunnel(self, p: Pos) -> bool:
        if not self._in_bounds(p): return False
        return self.grid[p[0]][p[1]] == TileType.MOUNTAIN

    def add_tunnel(self, p: Pos) -> bool:
        if not self.can_place_tunnel(p): return False
        if self.open_budget:
            if not self.spend(BUDGET_MAP['tunnel']): return False 
        self.grid[p[0]][p[1]] = TileType.TUNNEL
        self._ensure_road_stats(p)
        self.road_stats[p]["removed_at"] = -1
        return True

    def remove_tunnel(self, p: Pos) -> bool:
        if not self._in_bounds(p): return False
        if p in self.occupied:
            return False
        if self.grid[p[0]][p[1]] == TileType.TUNNEL:
            self.grid[p[0]][p[1]] = TileType.MOUNTAIN
            self._ensure_road_stats(p)
            self.road_stats[p]["removed_at"] = self.tick_count
            if self.open_budget:
                self.refund(BUDGET_MAP['tunnel'])
            return True
        return False

    def add_depot(self, p: Pos, kind: str, amount: int) -> Optional[int]:
        if kind not in ("out", "in"):
            return None
        if not self._in_bounds(p): return None
        if self.grid[p[0]][p[1]] != TileType.GRASS:
            return None
        depot_id = self.next_depot_id; self.next_depot_id += 1
        self.depots[depot_id] = Depot(depot_id, kind, max(0, int(amount)), p)
        self.depot_at[p] = depot_id
        self.init_depots[depot_id] = Depot(depot_id, kind, max(0, int(amount)), p)
        self.grid[p[0]][p[1]] = TileType.DEPOT
        return depot_id

    # ---------- Pathfinding ----------
    def _bfs_path(self, start: Pos, goals: Set[Pos]) -> Optional[List[Pos]]:
        if not goals:
            return None
        Q = collections.deque([start])
        came = {start: None}

        def passable(q: Pos) -> bool:
            t = self.grid[q[0]][q[1]]
            if t in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL):
                return True
            if t == TileType.DEPOT and q in goals:
                return True
            return False

        while Q:
            cur = Q.popleft()
            if cur in goals:
                path = []
                while cur is not None and cur != start:
                    path.append(cur)
                    cur = came[cur]
                path.reverse()
                return path
            for q in self._neighbors4(cur):
                if not passable(q): continue
                if q in came: continue
                came[q] = cur
                Q.append(q)
        return None

    def _connected_in_depots_with_demand(self) -> Set[int]:
        return {d.id for d in self.depots.values() if d.kind == "in" and d.amount > 0}

    # ---------- Spawning ----------
    def _spawn_from_out_depots(self):
        target_in_ids = self._connected_in_depots_with_demand()
        if not target_in_ids:
            return
        target_positions = {self.depots[i].pos for i in target_in_ids}

        for d in self.depots.values():
            if d.kind != "out" or d.amount <= 0:
                continue
            adj_roads = [q for q in self._neighbors4(d.pos)
                         if self._in_bounds(q) and self.grid[q[0]][q[1]] in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL)
                         and q not in self.occupied]
            if not adj_roads:
                continue
            spawn_pos = self.rng.choice(adj_roads)

            path = self._bfs_path(spawn_pos, target_positions)
            if not path:
                continue

            goal_pos = path[-1]
            goal_id = self.depot_at.get(goal_pos)
            if goal_id is None or self.depots[goal_id].amount <= 0:
                continue

            car_id = self.next_car_id; self.next_car_id += 1
            self.cars[car_id] = Car(car_id, spawn_pos, goal_id, path)
            self.occupied.add(spawn_pos)
            self._ensure_road_stats(spawn_pos)
            self.road_stats[spawn_pos]["entries"] += 1
            d.amount -= 1

    # ---------- Movement ----------
    def _move_cars(self):
        active_in_ids = self._connected_in_depots_with_demand()
        active_in_positions = {self.depots[i].pos for i in active_in_ids}

        new_occupied = set(self.occupied)
        to_remove = []

        for car_id in sorted(self.cars.keys()):
            car = self.cars[car_id]

            need_repath = False
            if not active_in_ids:
                car.path = []
            else:
                if car.target_depot_id is None or self.depots[car.target_depot_id].amount <= 0:
                    need_repath = True
                elif not car.path:
                    need_repath = True
                if need_repath:
                    new_path = self._bfs_path(car.pos, active_in_positions)
                    if new_path:
                        car.path = new_path
                        last = new_path[-1]
                        car.target_depot_id = self.depot_at.get(last)
                    else:
                        car.path = []

            if not car.path:
                continue

            nxt = car.path[0]
            cur = car.pos
            nxt_tile = self.grid[nxt[0]][nxt[1]]
            cur_tile = self.grid[cur[0]][cur[1]]

            nxt_is_traversable = nxt_tile in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL)
            nxt_is_depot = (nxt_tile == TileType.DEPOT)

            if nxt_is_traversable:
                if nxt not in new_occupied:
                    # Free current tile if it's traversable
                    if cur_tile in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL) and cur in new_occupied:
                        new_occupied.remove(cur)
                        self._ensure_road_stats(cur)
                        self.road_stats[cur]["exits"] += 1

                    car.pos = nxt
                    car.path.pop(0)
                    new_occupied.add(nxt)

                    self._ensure_road_stats(nxt)
                    self.road_stats[nxt]["entries"] += 1
                else:
                    self._ensure_road_stats(nxt)
                    self.road_stats[nxt]["blocked"] += 1

            elif nxt_is_depot:
                if cur_tile in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL) and cur in new_occupied:
                    new_occupied.remove(cur)
                    self._ensure_road_stats(cur)
                    self.road_stats[cur]["exits"] += 1

                car.pos = nxt
                car.path.pop(0)

                depot_id = self.depot_at.get(nxt)
                dep = self.depots.get(depot_id)
                if dep and dep.kind == "in" and dep.amount > 0:
                    dep.amount -= 1
                    to_remove.append(car_id)

        for cid in to_remove:
            self.cars.pop(cid, None)

        self.occupied = new_occupied

    # ---------- Simulation ----------
    def step(self):
        self._move_cars()
        self._spawn_from_out_depots()
        for p in self.occupied:
            tile = self.grid[p[0]][p[1]]
            if tile in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL):
                self._ensure_road_stats(p)
                self.road_stats[p]["ticks_occupied"] += 1
        self.tick_count += 1
    
    # ---------- Export and Load ----------
    def export_map(self):
        # export parts
        map_elements = {
            'depots': self.depots,
            'grids' :self.grid,
            'depot_at':self.depot_at,
            'init_depots': self.init_depots,
            'road_stats': self.road_stats,
            'map_params': {'width': self.w, 'height': self.h,
                            'seed':self.seed, 'initial_budget' : self.budget,
                              'with_water' : self.with_water, 'with_mountain': self.with_mountain}}
        
        if self.export_p is None:
            raise ValueError('The export path cannot be None, please define your export path')
        else:
            with open(self.export_p, 'wb') as f:
                pickle.dump(map_elements, f)


    # ---------- Rendering ----------
    def render_ascii(self) -> str:
        """
        Legend:
          . grass   ~ water   # road   O out-depot   I in-depot   c car-on-road
          ^ mountain   = bridge   T tunnel
        If a car stands on a depot tile, depot symbol is shown.
        """
        rows: List[str] = []
        car_positions = {c.pos: c for c in self.cars.values()}
        for y in range(self.h):
            line = []
            for x in range(self.w):
                p = (x,y)
                t = self.grid[x][y]
                if t == TileType.GRASS:
                    ch = "."
                elif t == TileType.WATER:
                    ch = "~"
                elif t == TileType.ROAD:
                    ch = "c" if p in car_positions else "#"
                elif t == TileType.MOUNTAIN:
                    ch = "^"
                elif t == TileType.BRIDGE:
                    ch = "c" if p in car_positions else "="
                elif t == TileType.TUNNEL:
                    ch = "c" if p in car_positions else "T"
                else:  # DEPOT
                    did = self.depot_at[p]
                    d = self.depots[did]
                    ch = "O" if d.kind == "out" else "I"
                line.append(ch)
            rows.append("".join(line))
        return "\n".join(rows)
    
    def render_image(self, scale: int = 12, tiles: Optional[Dict[str, str | "Image.Image"]] = None):

        img = Image.new("RGBA", (self.w * scale, self.h * scale), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        car_positions = {c.pos for c in self.cars.values()}
        tile_folder = './tiles_image/'

        tile_imgs = load_tile_images(tile_folder, scale) if tile_folder else {}
        for y in range(self.h):
            for x in range(self.w):
                p = (x, y)
                x0, y0 = x * scale, y * scale
                t = self.grid[x][y]

                if t == TileType.GRASS:
                    key, color = "grass", (180, 220, 180)
                elif t == TileType.WATER:
                    key, color = "water", (80, 120, 200)
                elif t == TileType.ROAD:
                    key, color = "road", (60, 60, 60)
                elif t == TileType.MOUNTAIN:
                    key, color = "mountain", (100, 80, 60)
                elif t == TileType.BRIDGE:
                    key, color = "bridge", (180, 160, 120)
                elif t == TileType.TUNNEL:
                    key, color = "tunnel", (50, 50, 50)
                else:  # DEPOT
                    d = self.depots[self.depot_at[p]]
                    key = "out" if d.kind == "out" else "in"
                    color = (240, 200, 60) if d.kind == "out" else (70, 200, 90)

                 # pick random variant if available
                tile_list = tile_imgs.get(key)
                if tile_list:
                    tile_img = random.choice(tile_list)
                    img.paste(tile_img, (x0, y0), mask=tile_img)
                else:
                    draw.rectangle([x0, y0, x0 + scale, y0 + scale], fill=color)

                if t in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL) and p in car_positions:
                    car_img = tile_imgs.get("car")
                    if car_img:
                        img.paste(car_img, (x0, y0), mask=car_img)
                    else:
                        pad = max(1, scale // 5)
                        draw.ellipse(
                            [x0 + pad, y0 + pad, x0 + scale - pad, y0 + scale - pad],
                            fill=(230, 90, 60)
                        )

        return img

    def render_heatmap(
        self,
        key: str = "occupancy_ratio",
        scale: int = 16,
        vmin: float | None = None,
        vmax: float | None = None,
        gradient: tuple[tuple[int,int,int], tuple[int,int,int]] = ((255, 255, 255), (200, 0, 0)),
    ):
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None

        summary = self.road_stats_summary()
        value_by_pos: dict[Pos, float] = {row["pos"]: float(row.get(key, 0.0)) for row in summary}

        values = []
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[x][y] in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL):
                    v = value_by_pos.get((x, y), 0.0)
                    values.append(v)

        if vmin is None:
            vmin = min(values) if values else 0.0
        if vmax is None:
            vmax = max(values) if values else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        def _lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        low, high = gradient
        grey = (160, 160, 160)
        black = (0, 0, 0)

        img = Image.new("RGB", (self.w * scale, self.h * scale), grey)
        draw = ImageDraw.Draw(img)

        for y in range(self.h):
            for x in range(self.w):
                p = (x, y)
                x0, y0 = x * scale, y * scale
                x1, y1 = x0 + scale, y0 + scale
                t = self.grid[x][y]

                if t == TileType.DEPOT:
                    color = black
                elif t in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL):
                    v = value_by_pos.get(p, 0.0)
                    tnorm = (v - vmin) / (vmax - vmin)
                    tnorm = max(0.0, min(1.0, tnorm))
                    color = (
                        int(_lerp(low[0], high[0], tnorm)),
                        int(_lerp(low[1], high[1], tnorm)),
                        int(_lerp(low[2], high[2], tnorm)),
                    )
                else:
                    color = grey

                draw.rectangle([x0, y0, x1, y1], fill=color)

        return img

    # ---------- Inspection ----------
    def snapshot(self) -> Dict:
        return {
            "tick": self.tick_count,
            "cars": [
                {"id": c.id, "pos": c.pos, "target": c.target_depot_id, "path_len": len(c.path)}
                for c in self.cars.values()
            ],
            "depots": [
                {"id": d.id, "kind": d.kind, "amount": d.amount, "pos": d.pos}
                for d in self.depots.values()
            ],
            "occupied": sorted(list(self.occupied)),
        }
    
    def road_stats_summary(self) -> List[Dict]:
        T = max(1, self.tick_count)
        out = []
        for p, s in self.road_stats.items():
            tile = self.grid[p[0]][p[1]]
            if s["removed_at"] != -1 :
                pass # remove the historical roads 
            else:
                active = (tile in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL))
                out.append({
                    "pos": p,
                    "active": active,
                    "entries": s["entries"],
                    "exits": s["exits"],
                    "ticks_occupied": s["ticks_occupied"],
                    "occupancy_ratio": s["ticks_occupied"] / T,
                    "flow_per_tick": s["entries"] / T,
                    "blocked": s["blocked"],
                    "created_at": s["created_at"],
                    "removed_at": s["removed_at"],
                })
        return out

    def top_congested(self, k: int = 10) -> List[Dict]:
        stats = self.road_stats_summary()
        stats.sort(key=lambda r: (r["occupancy_ratio"], r["blocked"]), reverse=True)
        return stats[:k]

    def reset_stats(self):
        for p in list(self.road_stats.keys()):
            tile = self.grid[p[0]][p[1]]
            self.road_stats[p].update({
                "entries": 0, "exits": 0, "ticks_occupied": 0, "blocked": 0,
                "created_at": self.tick_count,
                "removed_at": -1 if tile in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL) else self.tick_count
            })
        self.cars.clear()
        self.occupied.clear()
        self.next_car_id = 1
        self.depots = copy.deepcopy(self.init_depots)
        self.tick_count = 0
        self.log.append([])

    def export_road_stats_csv(self, path: str):
        import csv
        rows = self.road_stats_summary()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["pos","active","entries","exits","ticks_occupied","occupancy_ratio","flow_per_tick","blocked","created_at","removed_at"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # ---------- Utilities ----------
    def total_supply(self) -> int:
        return sum(d.amount for d in self.depots.values() if d.kind == "out")

    def total_demand(self) -> int:
        return sum(d.amount for d in self.depots.values() if d.kind == "in")

    def set_seed(self, seed: int):
        self.rng.seed(seed)


# ---------- quick demo ----------
if __name__ == "__main__":
    sim = TrafficSim(30, 16, seed=42)

    # Build road across middle
    y = sim.h // 2
    for x in range(sim.w):
        p = (x, y)
        if sim.grid[x][y] == TileType.GRASS:
            sim.add_road(p)

    # Add bridge over water if present
    for x in range(sim.w):
        p = (x, y+2)
        if sim.grid[x][y+2] == TileType.WATER:
            sim.add_bridge(p)

    # Add tunnel through mountain if present
    for x in range(sim.w):
        p = (x, y-2)
        if sim.grid[x][y-2] == TileType.MOUNTAIN:
            sim.add_tunnel(p)

    # Depots
    out1 = sim.add_depot((2, y), "out", 10)
    in1  = sim.add_depot((sim.w-3, y), "in", 12)

    print("Initial world:")
    print(sim.render_ascii())
    print("Supply >= Demand ?", sim.total_supply(), ">=", sim.total_demand())

    for _ in range(12):
        sim.step()
        print("\nTick", sim.tick_count)
        print(sim.render_ascii())
        print(sim.snapshot())