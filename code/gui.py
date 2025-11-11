import sys
import os
import pygame
from pygame.locals import *
from typing import Optional, Tuple
from PIL import Image
import io
import time

# Import your simulation class
from minisim import TrafficSim, TileType  # adjust if needed

# ---------- Config ----------
small = (20, 16)
medium = (40, 32)
large = (60, 32)
# map setting 
WITH_WATER = True
WITH_MOUNTAIN = True
(W, H) = medium           # grid size (match to taste)
SEED = 30
INITIAL_BUDGET = None 

SCALE = 28               # pixels per tile
FPS = 60
FONT_NAME = None         # default pygame font
BG_COLOR = (30, 30, 30)
GRID_COLOR = (40, 40, 40)
PANEL_H = 80
# Colors for on-screen draw
COLORS = {
    "grass": (180, 220, 180),
    "water": (80, 120, 200),
    "road": (60, 60, 60),
    "mountain": (100, 80, 60),
    "bridge": (180, 160, 120),
    "tunnel": (120, 60, 160),
    "car": (230, 90, 60),
    "out": (240, 200, 60),
    "in": (70, 200, 90),
    "grey": (160, 160, 160),
    "black": (0, 0, 0),
    "white": (240, 240, 240),
}

HEAT_KEYS = ["occupancy_ratio", "flow_per_tick", "entries", "ticks_occupied", "blocked"]

# ---------- Helpers ----------
def pil_to_surface(pil_img: Image.Image) -> pygame.Surface:
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()
    return pygame.image.fromstring(data, size, mode)

def clamp(v, a, b): return max(a, min(b, v))

# ---------- UI App ----------
class App:
    def __init__(self):
        pygame.init()
        self.sim = TrafficSim(W, H, seed=SEED, initial_budget = INITIAL_BUDGET, with_water = WITH_WATER, with_mountain = WITH_MOUNTAIN)
        self.screen = pygame.display.set_mode((W * SCALE, H * SCALE + PANEL_H))
        pygame.display.set_caption("TrafficSim — R=road B=bridge U=tunnel | 1=OUT 2=IN | SPACE run")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.small = pygame.font.Font(FONT_NAME, 14)

        self.running_sim = False
        self.drag_left = False
        self.drag_right = False
        self.mode = "road"  # "road", "bridge", "tunnel", "out", "in"
        self.default_depot_amt = 10

        self.show_heatmap = False
        self.heat_key_idx = 0
        self.heat_surface: Optional[pygame.Surface] = None
        self.heat_vmin = None
        self.heat_vmax = None
        self.load_tile_images()
        
    # ----- Load assets ---
    def load_tile_images(self, asset_dir="gui_assets"):
        """Load all tile images as pygame Surfaces."""
        tile_names = {
            "grass": TileType.GRASS,
            "water": TileType.WATER,
            "road": TileType.ROAD,
            "mountain": TileType.MOUNTAIN,
            "bridge": TileType.BRIDGE,
            "tunnel": TileType.TUNNEL,
            "out": "out",
            "in": "in",
        }
        self.tile_surfaces = {}
        self.car_surface = None

        for name, key in tile_names.items():
            path = os.path.join(asset_dir, f"{name}.png")
            try:
                img = pygame.image.load(path).convert_alpha()
                self.tile_surfaces[name] = pygame.transform.scale(img, (SCALE, SCALE))
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                self.tile_surfaces[name] = None

        # Load car separately
        try:
            car_img = pygame.image.load(os.path.join(asset_dir, "car.png")).convert_alpha()
            self.car_surface = pygame.transform.scale(car_img, (SCALE, SCALE))
        except Exception as e:
            print(f"Warning: Could not load car image: {e}")
            self.car_surface = None
        print('tile_surfaces', self.tile_surfaces)

    # ----- Map ops -----
    def new_map(self):
        seed = int(time.time()) % 10_000
        self.sim = TrafficSim(W, H, seed=seed)
        self.invalidate_heatmap()

    def clear_roads(self):
        to_remove = []
        for x in range(self.sim.w):
            for y in range(self.sim.h):
                t = self.sim.grid[x][y]
                if t in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL):
                    to_remove.append((x, y))
        for p in to_remove:
            if t == TileType.ROAD:
                self.sim.remove_road(p)
            elif t == TileType.BRIDGE:
                self.sim.remove_bridge(p)
            elif t == TileType.TUNNEL:
                self.sim.remove_tunnel(p)
        self.invalidate_heatmap()

    # ----- Heatmap -----
    def invalidate_heatmap(self):
        self.heat_surface = None

    def make_heatmap(self):
        key = HEAT_KEYS[self.heat_key_idx]
        img = self.sim.render_heatmap(key=key, scale=SCALE, vmin=self.heat_vmin, vmax=self.heat_vmax)
        if img:
            self.heat_surface = pil_to_surface(img)

    # ----- Interaction -----
    def grid_pos_from_mouse(self, mx, my) -> Optional[Tuple[int, int]]:
        if my >= H * SCALE:
            return None
        gx = mx // SCALE
        gy = my // SCALE
        if 0 <= gx < W and 0 <= gy < H:
            return (gx, gy)
        return None

    def handle_click(self, pos, button):
        if pos is None:
            return
        x, y = pos
        if button == 1:  # LMB: add
            if self.mode == "road":
                if self.sim.add_road((x, y)):
                    self.invalidate_heatmap()
            elif self.mode == "bridge":
                if self.sim.add_bridge((x, y)):
                    self.invalidate_heatmap()
            elif self.mode == "tunnel":
                if self.sim.add_tunnel((x, y)):
                    self.invalidate_heatmap()
            elif self.mode == "out":
                self.sim.add_depot((x, y), "out", self.default_depot_amt)
                self.invalidate_heatmap()
            elif self.mode == "in":
                self.sim.add_depot((x, y), "in", self.default_depot_amt)
                self.invalidate_heatmap()
        elif button == 3:  # RMB: remove
            t = self.sim.grid[x][y]
            if t == TileType.ROAD:
                if self.sim.remove_road((x, y)):
                    self.invalidate_heatmap()
            elif t == TileType.BRIDGE:
                if self.sim.remove_bridge((x, y)):
                    self.invalidate_heatmap()
            elif t == TileType.TUNNEL:
                if self.sim.remove_tunnel((x, y)):
                    self.invalidate_heatmap()
            # Note: depot removal not implemented (optional)

    # ----- Sim control -----
    def step(self, n=1):
        for _ in range(n):
            self.sim.step()
        if self.show_heatmap:
            self.make_heatmap()

    # ----- Render -----
    def draw_world_basic(self, surf):
        car_positions = {c.pos for c in self.sim.cars.values()}
        for y in range(H):
            for x in range(W):
                p = (x, y)
                t = self.sim.grid[x][y]

                # Choose surface key
                if t == TileType.DEPOT:
                    d = self.sim.depots[self.sim.depot_at[p]]
                    key = "out" if d.kind == "out" else "in"
                else:
                    # Map TileType to string key
                    key_map = {
                        TileType.GRASS: "grass",
                        TileType.WATER: "water",
                        TileType.ROAD: "road",
                        TileType.MOUNTAIN: "mountain",
                        TileType.BRIDGE: "bridge",
                        TileType.TUNNEL: "tunnel",
                    }
                    key = key_map.get(t)

                # Get surface
                
                surface = self.tile_surfaces.get(key)
                if surface:
                    surf.blit(surface, (x * SCALE, y * SCALE))
                else:
                    # Fallback to color
                    color = COLORS.get(key, COLORS["grey"])
                    rect = pygame.Rect(x * SCALE, y * SCALE, SCALE, SCALE)
                    surf.fill(color, rect)

                # Draw car on top if present
                if t in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL) and p in car_positions:
                    if self.car_surface:
                        surf.blit(self.car_surface, (x * SCALE, y * SCALE))
                    else:
                        # Fallback ellipse
                        pad = max(1, SCALE // 5)
                        pygame.draw.ellipse(surf, COLORS["car"],
                            (x * SCALE + pad, y * SCALE + pad, SCALE - 2 * pad, SCALE - 2 * pad))

        # Optional: keep grid lines for clarity
        for x in range(W + 1):
            pygame.draw.line(surf, GRID_COLOR, (x * SCALE, 0), (x * SCALE, H * SCALE))
        for y in range(H + 1):
            pygame.draw.line(surf, GRID_COLOR, (0, y * SCALE), (W * SCALE, y * SCALE))

    def draw_world(self):
        if self.show_heatmap and self.heat_surface is not None:
            self.screen.blit(self.heat_surface, (0, 0))
        else:
            self.draw_world_basic(self.screen)

    def draw_panel(self, hover: Optional[Tuple[int, int]]):
        panel_rect = pygame.Rect(0, H * SCALE, W * SCALE, PANEL_H)
        self.screen.fill(BG_COLOR, panel_rect)

        running = "RUN" if self.running_sim else "PAUSE"
        key = HEAT_KEYS[self.heat_key_idx]
        line1 = f"[{running}] tick={self.sim.tick_count}  budget=${self.sim.budget}  cars={len(self.sim.cars)}  supply={self.sim.total_supply()} demand={self.sim.total_demand()}  mode={self.mode} (depot_amt={self.default_depot_amt})"
        line2 = f"LMB=add | RMB=remove | R=road B=bridge U=tunnel | 1=OUT 2=IN | SPACE run/pause | S/F/G step | H heatmap (M={key}) | N new | V csv | P screenshot | E export | [ ] depot amt"

        txt1 = self.font.render(line1, True, COLORS["white"])
        txt2 = self.small.render(line2, True, COLORS["white"])
        self.screen.blit(txt1, (10, H * SCALE + 8))
        self.screen.blit(txt2, (10, H * SCALE + 36))

        if hover:
            x, y = hover
            t = self.sim.grid[x][y]
            info = f"hover=({x},{y}) type={t.name}"
            # Show stats for any traversable infrastructure
            if t in (TileType.ROAD, TileType.BRIDGE, TileType.TUNNEL):
                s = self.sim.road_stats.get((x, y))
                if s:
                    occ_ratio = s['ticks_occupied'] / max(1, self.sim.tick_count)
                    info += f" | entries={s['entries']} occ={s['ticks_occupied']} ratio={occ_ratio:.2f} blocked={s['blocked']}"
            elif t == TileType.DEPOT:
                did = self.sim.depot_at[(x, y)]
                d = self.sim.depots[did]
                info += f" | depot #{did} {d.kind} amount={d.amount}"
            txt3 = self.small.render(info, True, COLORS["white"])
            self.screen.blit(txt3, (10, H * SCALE + 58))


    # ----- Main loop -----
    def run(self):
        while True:
            dt = self.clock.tick(FPS)
            hover = None

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        self.running_sim = not self.running_sim
                    elif event.key == K_s:
                        self.step(1)
                    elif event.key == K_f:
                        self.step(10)
                    elif event.key == K_g:
                        self.step(50)
                    elif event.key == K_h:
                        self.show_heatmap = not self.show_heatmap
                        if self.show_heatmap:
                            self.make_heatmap()
                    elif event.key == K_m:
                        self.heat_key_idx = (self.heat_key_idx + 1) % len(HEAT_KEYS)
                        if self.show_heatmap:
                            self.make_heatmap()
                    elif event.key == K_v:
                        self.sim.export_road_stats_csv("road_stats.csv")
                        print("Exported: road_stats.csv")
                    elif event.key == K_p:
                        pygame.image.save(self.screen, "screenshot.png")
                        print("Saved: screenshot.png")
                    elif event.key == K_n:
                        self.new_map()
                    elif event.key == K_c:
                        self.clear_roads()
                    elif event.key == K_r:
                        self.mode = "road"
                    elif event.key == K_b:
                        self.mode = "bridge"
                    elif event.key == K_u:
                        self.mode = "tunnel"
                    elif event.key == K_1:
                        self.mode = "out"
                    elif event.key == K_2:
                        self.mode = "in"
                    elif event.key == K_LEFTBRACKET:
                        self.default_depot_amt = clamp(self.default_depot_amt - 1, 0, 9999)
                    elif event.key == K_RIGHTBRACKET:
                        self.default_depot_amt = clamp(self.default_depot_amt + 1, 0, 9999)
                    elif event.key == K_e:
                        self.sim.export_map()
                        print("Exported: exported_map.json")

                elif event.type == MOUSEBUTTONDOWN:
                    pos = self.grid_pos_from_mouse(*event.pos)
                    if event.button == 1:
                        self.drag_left = True
                        self.handle_click(pos, 1)
                    elif event.button == 3:
                        self.drag_right = True
                        self.handle_click(pos, 3)

                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drag_left = False
                    elif event.button == 3:
                        self.drag_right = False

                elif event.type == MOUSEMOTION:
                    pos = self.grid_pos_from_mouse(*event.pos)
                    hover = pos
                    if self.drag_left:
                        self.handle_click(pos, 1)
                    if self.drag_right:
                        self.handle_click(pos, 3)

            if self.running_sim:
                self.step(1)

            self.screen.fill(BG_COLOR)
            self.draw_world()
            if hover is None:
                pos = self.grid_pos_from_mouse(*pygame.mouse.get_pos())
                hover = pos
            self.draw_panel(hover)
            pygame.display.flip()

if __name__ == "__main__":
    App().run()