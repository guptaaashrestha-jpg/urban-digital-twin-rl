"""
Pygame Renderer for the Urban Digital Twin.

Dark-themed, premium visualization with:
- Asphalt roads with lane markings
- Color-coded vehicles (wait-time gradient)
- Glowing traffic lights
- Real-time stats sidebar
"""

import pygame
import math
from .vehicle import Direction, VehicleState
from .traffic_light import Phase


# ── Color Palette (Dark Theme) ─────────────────────────────────
COLORS = {
    "bg":             (12, 12, 24),
    "grass":          (18, 30, 22),
    "road":           (42, 42, 54),
    "road_edge":      (60, 60, 75),
    "lane_marking":   (180, 180, 190),
    "center_line":    (230, 200, 60),
    "crosswalk":      (180, 180, 190),
    "sidebar":        (16, 16, 28),
    "sidebar_card":   (26, 26, 42),
    "sidebar_border": (45, 45, 65),
    "text_bright":    (235, 235, 245),
    "text_dim":       (120, 120, 145),
    "text_accent":    (100, 180, 255),
    "green_glow":     (0, 230, 100),
    "red_glow":       (240, 50, 50),
    "yellow_glow":    (250, 210, 40),
    "metric_good":    (72, 199, 142),
    "metric_warn":    (251, 191, 36),
    "metric_bad":     (239, 68, 68),
}


class Renderer:
    """Premium dark-themed Pygame renderer for the traffic simulation."""

    def __init__(self, sim_size=800, sidebar_width=400, fps=60):
        pygame.init()
        self.sim_size = sim_size
        self.sidebar_width = sidebar_width
        self.width = sim_size + sidebar_width
        self.height = sim_size
        self.fps = fps

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Urban Digital Twin — Traffic RL Optimizer")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI", 16)
        self.font_metric_label = pygame.font.SysFont("Segoe UI", 14)
        self.font_metric_value = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 12)
        self.font_phase = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.particles = []

        # Pre-render static road surface
        self.road_surface = self._build_road_surface()

    def _build_road_surface(self):
        """Pre-render the static road, markings, and grass."""
        surface = pygame.Surface((self.sim_size, self.sim_size))
        surface.fill(COLORS["bg"])

        cx = self.sim_size // 2
        cy = self.sim_size // 2
        rw = 100  # road width (2 lanes x 50px)
        hw = rw // 2

        # Draw grass/ground patches (subtle grid)
        for gx in range(0, self.sim_size, 40):
            for gy in range(0, self.sim_size, 40):
                shade = 18 + ((gx + gy) % 80) // 10
                c = (shade, shade + 12, shade)
                pygame.draw.rect(surface, c, (gx, gy, 40, 40))

        # Draw roads
        # Vertical road
        pygame.draw.rect(surface, COLORS["road"],
                         (cx - hw, 0, rw, self.sim_size))
        # Horizontal road
        pygame.draw.rect(surface, COLORS["road"],
                         (0, cy - hw, self.sim_size, rw))

        # Road edge lines (solid white)
        for offset in [-hw, hw]:
            # Vertical road edges
            pygame.draw.line(surface, COLORS["road_edge"],
                             (cx + offset, 0), (cx + offset, cy - hw), 2)
            pygame.draw.line(surface, COLORS["road_edge"],
                             (cx + offset, cy + hw), (cx + offset, self.sim_size), 2)
            # Horizontal road edges
            pygame.draw.line(surface, COLORS["road_edge"],
                             (0, cy + offset), (cx - hw, cy + offset), 2)
            pygame.draw.line(surface, COLORS["road_edge"],
                             (cx + hw, cy + offset), (self.sim_size, cy + offset), 2)

        # Center dashed lines (yellow)
        dash_len, gap = 20, 15
        # Vertical center line (above intersection)
        y = 0
        while y < cy - hw:
            pygame.draw.line(surface, COLORS["center_line"],
                             (cx, y), (cx, min(y + dash_len, cy - hw)), 2)
            y += dash_len + gap
        # Vertical center line (below intersection)
        y = cy + hw
        while y < self.sim_size:
            pygame.draw.line(surface, COLORS["center_line"],
                             (cx, y), (cx, min(y + dash_len, self.sim_size)), 2)
            y += dash_len + gap
        # Horizontal center line (left)
        x = 0
        while x < cx - hw:
            pygame.draw.line(surface, COLORS["center_line"],
                             (x, cy), (min(x + dash_len, cx - hw), cy), 2)
            x += dash_len + gap
        # Horizontal center line (right)
        x = cx + hw
        while x < self.sim_size:
            pygame.draw.line(surface, COLORS["center_line"],
                             (x, cy), (min(x + dash_len, self.sim_size), cy), 2)
            x += dash_len + gap

        # Crosswalk stripes at each approach
        stripe_w, stripe_gap = 6, 8
        stripe_color = (100, 100, 115)
        # North approach (above intersection)
        for i in range(6):
            sx = cx - hw + 8 + i * (stripe_w + stripe_gap)
            pygame.draw.rect(surface, stripe_color,
                             (sx, cy - hw - 14, stripe_w, 10))
        # South approach
        for i in range(6):
            sx = cx - hw + 8 + i * (stripe_w + stripe_gap)
            pygame.draw.rect(surface, stripe_color,
                             (sx, cy + hw + 4, stripe_w, 10))
        # West approach
        for i in range(6):
            sy = cy - hw + 8 + i * (stripe_w + stripe_gap)
            pygame.draw.rect(surface, stripe_color,
                             (cx - hw - 14, sy, 10, stripe_w))
        # East approach
        for i in range(6):
            sy = cy - hw + 8 + i * (stripe_w + stripe_gap)
            pygame.draw.rect(surface, stripe_color,
                             (cx + hw + 4, sy, 10, stripe_w))

        return surface

    def render(self, intersection):
        """Render one frame of the simulation."""
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Draw static road
        self.screen.blit(self.road_surface, (0, 0))

        # Draw traffic lights
        self._draw_traffic_lights(intersection)

        # Draw V2X data beams
        self._draw_v2x_beams(intersection)

        # Draw vehicles
        self._draw_vehicles(intersection)

        # Draw weather overlay
        if hasattr(intersection, "current_weather"):
            self._draw_weather(intersection.current_weather)

        # Draw sidebar
        self._draw_sidebar(intersection)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_traffic_lights(self, intersection):
        """Draw glowing traffic light indicators near each stop line."""
        tl = intersection.traffic_light
        cx = intersection.center_x
        cy = intersection.center_y
        hw = intersection.road_width // 2

        # Signal positions: just outside the intersection box
        positions = {
            "ns": [
                (cx + hw + 20, cy - hw + 15),   # North-South indicator (top-right)
                (cx - hw - 20, cy + hw - 15),    # North-South indicator (bottom-left)
            ],
            "ew": [
                (cx - hw + 15, cy - hw - 20),   # East-West indicator (top-left)
                (cx + hw - 15, cy + hw + 20),    # East-West indicator (bottom-right)
            ],
        }

        ns_color_name = tl.ns_color
        ew_color_name = tl.ew_color

        color_map = {
            "green": COLORS["green_glow"],
            "red": COLORS["red_glow"],
            "yellow": COLORS["yellow_glow"],
        }

        # Draw NS signals
        for pos in positions["ns"]:
            self._draw_light_circle(pos, color_map[ns_color_name])

        # Draw EW signals
        for pos in positions["ew"]:
            self._draw_light_circle(pos, color_map[ew_color_name])

    def _draw_light_circle(self, pos, color):
        """Draw a traffic light with a glow effect."""
        # Outer glow (larger, semi-transparent)
        glow_surf = pygame.Surface((50, 50), pygame.SRCALPHA)
        for r in range(25, 8, -1):
            alpha = int(60 * (1 - (r - 8) / 17))
            pygame.draw.circle(glow_surf, (*color, alpha), (25, 25), r)
        self.screen.blit(glow_surf, (pos[0] - 25, pos[1] - 25))

        # Solid center
        pygame.draw.circle(self.screen, color, pos, 8)
        # Bright core
        pygame.draw.circle(self.screen, (255, 255, 255), pos, 3)

    def _draw_v2x_beams(self, intersection):
        import math
        cx, cy = intersection.center_x, intersection.center_y
        beam_surf = pygame.Surface((self.sim_size, self.sim_size), pygame.SRCALPHA)
        alpha = int(80 + 50 * math.sin(pygame.time.get_ticks() / 150.0))
        for vehicle in intersection.get_all_vehicles():
            if getattr(vehicle, "is_v2x_equipped", False) and getattr(vehicle, "v2x_target_speed", None) is not None:
                pygame.draw.line(beam_surf, (0, 255, 255, alpha), (cx, cy), (vehicle.x, vehicle.y), 2)
        self.screen.blit(beam_surf, (0, 0))

    def _draw_vehicles(self, intersection):
        """Draw all vehicles with wait-time color coding."""
        for vehicle in intersection.get_all_vehicles():
            # Vehicle rectangle
            w = vehicle.render_width
            h = vehicle.render_height
            
            vx, vy = vehicle.x, vehicle.y
            if getattr(vehicle, "pulled_over", False):
                shift = 15
                if vehicle.direction in (Direction.NORTH, Direction.SOUTH):
                    vx += shift
                else:
                    vy += shift

            rect = pygame.Rect(vx - w / 2, vy - h / 2, w, h)

            # Body color
            if vehicle.is_emergency:
                ticks = pygame.time.get_ticks()
                color = (255, 30, 30) if (ticks // 200) % 2 == 0 else (30, 30, 255)
            else:
                color = vehicle.body_color
                if vehicle.state == VehicleState.WAITING:
                    color = vehicle.get_wait_color()

            # Draw body with rounded corners
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

            # Darker outline
            darker = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, darker, rect, 1, border_radius=4)

            # Headlights (small bright dots at front)
            self._draw_headlights(vehicle, rect)
            # Fined indicator
            if getattr(vehicle, "was_fined", False):
                pygame.draw.rect(self.screen, COLORS["metric_bad"], rect, 2, border_radius=4)
                fine_text = self.font_small.render("$$$", True, COLORS["metric_bad"])
                self.screen.blit(fine_text, (vx - 10, vy - h/2 - 15))

            # V2X Glow
            if getattr(vehicle, "is_v2x_equipped", False):
                pygame.draw.rect(self.screen, (0, 255, 255), rect, 2, border_radius=4)

    def _draw_headlights(self, vehicle, rect):
        """Draw tiny headlight dots at the front of the vehicle."""
        hl_color = (255, 255, 200)
        hl_r = 2
        offset = 4

        if vehicle.direction == Direction.NORTH:
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.left + offset), int(rect.top + 3)), hl_r)
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.right - offset), int(rect.top + 3)), hl_r)
        elif vehicle.direction == Direction.SOUTH:
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.left + offset), int(rect.bottom - 3)), hl_r)
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.right - offset), int(rect.bottom - 3)), hl_r)
        elif vehicle.direction == Direction.EAST:
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.right - 3), int(rect.top + offset)), hl_r)
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.right - 3), int(rect.bottom - offset)), hl_r)
        else:
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.left + 3), int(rect.top + offset)), hl_r)
            pygame.draw.circle(self.screen, hl_color,
                               (int(rect.left + 3), int(rect.bottom - offset)), hl_r)

    def _draw_sidebar(self, intersection):
        """Draw the stats sidebar on the right side."""
        sx = self.sim_size  # Sidebar start x
        sw = self.sidebar_width

        # Sidebar background
        pygame.draw.rect(self.screen, COLORS["sidebar"], (sx, 0, sw, self.height))
        # Left border accent
        pygame.draw.line(self.screen, COLORS["sidebar_border"],
                         (sx, 0), (sx, self.height), 2)

        y = 20

        # ── Title ──
        title = self.font_title.render("URBAN DIGITAL TWIN", True, COLORS["text_bright"])
        self.screen.blit(title, (sx + 20, y))
        y += 40

        subtitle = self.font_subtitle.render("Traffic RL Optimizer", True, COLORS["text_accent"])
        self.screen.blit(subtitle, (sx + 20, y))
        y += 35

        # Time of Day badge
        tod_text = intersection.current_phase_name
        badge_color = COLORS["text_accent"]
        badge = self.font_small.render(tod_text, True, badge_color)
        badge_rect = badge.get_rect()
        badge_bg = pygame.Rect(sx + 18, y - 2, badge_rect.width + 16, badge_rect.height + 8)
        pygame.draw.rect(self.screen, (*badge_color, 30), badge_bg, border_radius=4)
        pygame.draw.rect(self.screen, badge_color, badge_bg, 1, border_radius=4)
        self.screen.blit(badge, (sx + 26, y + 2))
        y += 45

        # Weather badge
        if hasattr(intersection, "current_weather"):
            from .vehicle import Weather
            w = intersection.current_weather
            if w == Weather.CLEAR:
                w_text, w_color = "CLEAR", COLORS["metric_good"]
            elif w == Weather.RAIN:
                w_text, w_color = "RAIN", (100, 150, 255)
            else:
                w_text, w_color = "SNOW", (200, 200, 200)

            w_badge = self.font_small.render(f"WEATHER: {w_text}", True, w_color)
            w_rect = w_badge.get_rect()
            w_bg = pygame.Rect(sx + 18 + badge_rect.width + 26, y - 45 - 2, w_rect.width + 16, w_rect.height + 8)
            pygame.draw.rect(self.screen, (*w_color, 30), w_bg, border_radius=4)
            pygame.draw.rect(self.screen, w_color, w_bg, 1, border_radius=4)
            self.screen.blit(w_badge, (sx + 26 + badge_rect.width + 26, y - 45 + 2))

        # Divider
        pygame.draw.line(self.screen, COLORS["sidebar_border"],
                         (sx + 20, y), (sx + sw - 20, y), 1)
        y += 20

        # ── Phase Indicator ──
        tl = intersection.traffic_light
        phase_label = self.font_metric_label.render("CURRENT PHASE", True, COLORS["text_dim"])
        self.screen.blit(phase_label, (sx + 20, y))
        y += 22

        # Phase visualization
        ns_color = {"green": COLORS["green_glow"], "yellow": COLORS["yellow_glow"],
                    "red": COLORS["red_glow"]}[tl.ns_color]
        ew_color = {"green": COLORS["green_glow"], "yellow": COLORS["yellow_glow"],
                    "red": COLORS["red_glow"]}[tl.ew_color]

        # NS indicator
        pygame.draw.circle(self.screen, ns_color, (sx + 30, y + 10), 6)
        ns_label = self.font_small.render(f"N-S: {tl.ns_color.upper()}", True, COLORS["text_bright"])
        self.screen.blit(ns_label, (sx + 45, y + 3))

        # EW indicator
        pygame.draw.circle(self.screen, ew_color, (sx + 200, y + 10), 6)
        ew_label = self.font_small.render(f"E-W: {tl.ew_color.upper()}", True, COLORS["text_bright"])
        self.screen.blit(ew_label, (sx + 215, y + 3))
        y += 30

        # Timer bar
        progress = tl.phase_progress
        bar_w = sw - 40
        bar_h = 8
        # Background
        pygame.draw.rect(self.screen, COLORS["sidebar_card"],
                         (sx + 20, y, bar_w, bar_h), border_radius=4)
        # Fill
        fill_color = ns_color if tl.ns_color != "red" else ew_color
        pygame.draw.rect(self.screen, fill_color,
                         (sx + 20, y, int(bar_w * progress), bar_h), border_radius=4)
        y += 18

        remaining = self.font_small.render(
            f"{tl.phase_time_remaining_seconds:.1f}s remaining", True, COLORS["text_dim"])
        self.screen.blit(remaining, (sx + 20, y))
        y += 35

        # Divider
        pygame.draw.line(self.screen, COLORS["sidebar_border"],
                         (sx + 20, y), (sx + sw - 20, y), 1)
        y += 20

        # ── Metric Cards ──
        metrics = [
            ("AVG WAIT TIME", f"{intersection.avg_wait_time:.1f}s",
             self._wait_color(intersection.avg_wait_time)),
            ("VEHICLES ACTIVE", f"{intersection.total_vehicles}",
             COLORS["text_accent"]),
            ("THROUGHPUT", f"{intersection.throughput_per_minute:.0f} / min",
             COLORS["metric_good"]),
            ("TOTAL CLEARED", f"{intersection.total_vehicles_cleared}",
             COLORS["text_accent"]),
            ("FINES COLLECTED", f"${getattr(intersection, 'total_fines_collected', 0):,}",
             COLORS["metric_bad"] if getattr(intersection, 'total_fines_collected', 0) > 0 else COLORS["text_dim"]),
            ("V2X SYNCED", f"{getattr(intersection, 'v2x_synced_count', 0)}",
             (0, 255, 255) if getattr(intersection, 'v2x_synced_count', 0) > 0 else COLORS["text_dim"]),
        ]

        for label, value, color in metrics:
            y = self._draw_metric_card(sx + 20, y, sw - 40, label, value, color)
            y += 10

        y += 5

        # Divider
        pygame.draw.line(self.screen, COLORS["sidebar_border"],
                         (sx + 20, y), (sx + sw - 20, y), 1)
        y += 20

        # ── Queue Lengths ──
        q_label = self.font_metric_label.render("QUEUE LENGTHS", True, COLORS["text_dim"])
        self.screen.blit(q_label, (sx + 20, y))
        y += 22

        queues = intersection.queue_lengths
        directions = [("N", "NORTH"), ("S", "SOUTH"), ("E", "EAST"), ("W", "WEST")]
        bar_max_w = (sw - 60) // 2 - 40

        for i, (short, full) in enumerate(directions):
            col = i % 2
            row = i // 2
            bx = sx + 20 + col * ((sw - 40) // 2)
            by = y + row * 35

            dir_label = self.font_small.render(short, True, COLORS["text_bright"])
            self.screen.blit(dir_label, (bx, by + 2))

            count = queues[full]
            count_text = self.font_small.render(str(count), True, COLORS["text_bright"])
            self.screen.blit(count_text, (bx + 20, by + 2))

            # Mini bar
            bar_fill = min(count / 10, 1.0)
            bar_color = COLORS["metric_good"] if count < 4 else (
                COLORS["metric_warn"] if count < 8 else COLORS["metric_bad"])
            pygame.draw.rect(self.screen, COLORS["sidebar_card"],
                             (bx + 45, by + 4, bar_max_w, 10), border_radius=3)
            if bar_fill > 0:
                pygame.draw.rect(self.screen, bar_color,
                                 (bx + 45, by + 4, int(bar_max_w * bar_fill), 10),
                                 border_radius=3)

        y += 80

        # ── Sim Time ──
        sim_time = intersection.sim_time_seconds
        minutes = int(sim_time // 60)
        seconds = int(sim_time % 60)
        time_text = self.font_small.render(
            f"Sim Time: {minutes:02d}:{seconds:02d}", True, COLORS["text_dim"])
        self.screen.blit(time_text, (sx + 20, self.height - 50))

        fps_text = self.font_small.render(
            f"FPS: {self.clock.get_fps():.0f}", True, COLORS["text_dim"])
        self.screen.blit(fps_text, (sx + 20, self.height - 30))

    def _draw_metric_card(self, x, y, w, label, value, color):
        """Draw a metric card with label and large value."""
        card_h = 65
        # Card background
        pygame.draw.rect(self.screen, COLORS["sidebar_card"],
                         (x, y, w, card_h), border_radius=8)
        pygame.draw.rect(self.screen, COLORS["sidebar_border"],
                         (x, y, w, card_h), 1, border_radius=8)

        # Label
        lbl = self.font_metric_label.render(label, True, COLORS["text_dim"])
        self.screen.blit(lbl, (x + 12, y + 8))

        # Value
        val = self.font_metric_value.render(value, True, color)
        self.screen.blit(val, (x + 12, y + 26))

        return y + card_h

    def _wait_color(self, wait_seconds):
        if wait_seconds < 5:
            return COLORS["metric_good"]
        elif wait_seconds < 15:
            return COLORS["metric_warn"]
        else:
            return COLORS["metric_bad"]

    def _draw_weather(self, weather):
        from .vehicle import Weather
        import random
        
        # Spawn particles
        if weather == Weather.RAIN:
            if random.random() < 0.5:
                for _ in range(5):
                    x = random.randint(0, self.width)
                    y = -20
                    vy = random.uniform(15, 25)
                    self.particles.append([x, y, vy, "rain"])
        elif weather == Weather.SNOW:
            if random.random() < 0.3:
                for _ in range(3):
                    x = random.randint(0, self.width)
                    y = -10
                    vy = random.uniform(2, 6)
                    self.particles.append([x, y, vy, "snow"])
        
        # Update and draw
        to_remove = []
        for i, p in enumerate(self.particles):
            p[1] += p[2]
            if p[3] == "rain":
                pygame.draw.line(self.screen, (150, 150, 255), (p[0], p[1]), (p[0]-2, p[1]+10), 2)
            else:
                p[0] += random.uniform(-1, 1)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(p[0]), int(p[1])), 2)
            
            if p[1] > self.height:
                to_remove.append(i)
                
        for i in reversed(to_remove):
            self.particles.pop(i)

    def close(self):
        pygame.quit()
