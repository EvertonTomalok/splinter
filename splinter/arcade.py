from __future__ import annotations

import random
from dataclasses import dataclass, field

GAMES = {"pong": "Pong", "dino": "Dino", "ship": "Star patrol", "naval": "Battleship"}
WIDTH = 64
HEIGHT = 24
#: Dino jump, in pixels per tick (20 ticks a second): a quick takeoff, a short
#: hang at the top and a faster fall, like a runner, not a balloon.
JUMP_SPEED = 2.2
RISE_GRAVITY = 0.22
FALL_GRAVITY = 0.40
RUN_SPEED = 1.0


@dataclass
class Arcade:
    kind: str
    rng: random.Random = field(default_factory=random.Random)
    score: int = 0
    cpu_score: int = 0
    lives: int = 3
    over: bool = False
    ticks: int = 0
    player: float = 12
    opponent: float = 12
    x: float = 32
    y: float = 12
    vx: float = 0.8
    vy: float = 0.4
    jump: float = 0
    jump_speed: float = 0
    objects: list[tuple[float, float]] = field(default_factory=list)
    bullets: list[tuple[float, float]] = field(default_factory=list)
    fleet: set[tuple[int, int]] = field(default_factory=set)
    shots: set[tuple[int, int]] = field(default_factory=set)
    cursor: tuple[int, int] = (0, 0)
    ships: list[set[tuple[int, int]]] = field(default_factory=list, init=False)
    naval_message: str = ""
    _elapsed_ticks: float = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._elapsed_ticks = float(self.ticks)
        if self.kind == "ship":
            self.player = 32
        if self.kind == "dino":
            self.objects = [(58, 0)]
        if self.kind == "naval":
            for length in (3, 2, 2):
                while True:
                    vertical = self.rng.choice((True, False))
                    x = self.rng.randrange(8 if vertical else 9 - length)
                    y = self.rng.randrange(6 - length if vertical else 5)
                    cells = {
                        (x + (0 if vertical else n), y + (n if vertical else 0))
                        for n in range(length)
                    }
                    if not cells & self.fleet:
                        self.fleet.update(cells)
                        self.ships.append(cells)
                        break

    @property
    def controls(self) -> str:
        return {
            "pong": "↑ ↓ / W S move paddle · first to 7 wins",
            "dino": "Space / ↑ jump · clear the cacti",
            "ship": "← → / A D move · Space fire · 3 shields",
            "naval": (
                f"{self.result} · Space restart" if self.over else
                f"{self.naval_message} · Click / Space fire · arrows aim" if self.naval_message else
                "Click a square to fire · 3 ships / 7 cells · 18 shots"
            ),
        }[self.kind]

    @property
    def status(self) -> str:
        if self.kind == "pong":
            return f"YOU {self.score:02}   CPU {self.cpu_score:02}"
        if self.kind == "naval":
            return f"HITS {len(self.shots & self.fleet)}/7   SHOTS {len(self.shots)}/18"
        return f"SCORE {self.score:04}" + (
            f"   SHIELDS {self.lives}" if self.kind == "ship" else ""
        )

    @property
    def result(self) -> str:
        if (
            self.kind == "naval"
            and self.fleet <= self.shots
            or self.kind == "pong"
            and self.score == 7
        ):
            return "You win!"
        return "Game over"

    def key(self, key: str) -> None:
        if self.over:
            return
        if self.kind == "pong":
            self.player = min(
                20, max(3, self.player + ({"up": -2, "w": -2, "down": 2, "s": 2}.get(key, 0)))
            )
        elif self.kind == "dino" and key in {"space", "up", "w"} and self.jump == 0:
            self.jump_speed = JUMP_SPEED
        elif self.kind == "ship":
            self.player = min(
                59, max(4, self.player + ({"left": -2, "a": -2, "right": 2, "d": 2}.get(key, 0)))
            )
            if key == "space" and len(self.bullets) < 8:
                self.bullets.append((self.player, 19))
        elif self.kind == "naval":
            x, y = self.cursor
            self.cursor = (
                min(7, max(0, x + {"left": -1, "right": 1}.get(key, 0))),
                min(4, max(0, y + {"up": -1, "down": 1}.get(key, 0))),
            )
            if key == "space":
                coordinate = f"{chr(65 + self.cursor[0])}{self.cursor[1] + 1}"
                if self.cursor in self.shots:
                    self.naval_message = f"{coordinate}: already tried"
                    return
                self.shots.add(self.cursor)
                sunk = any(self.cursor in ship and ship <= self.shots for ship in self.ships)
                outcome = "SUNK" if sunk else "HIT" if self.cursor in self.fleet else "MISS"
                self.naval_message = f"{coordinate}: {outcome}"
                self.over = self.fleet <= self.shots or len(self.shots) >= 18

    def tick(self, seconds: float = 0.05) -> None:
        if self.over or self.kind == "naval":
            return
        step = max(0, min(seconds, 0.1)) * 20
        before = self.ticks
        self._elapsed_ticks += step
        self.ticks = int(self._elapsed_ticks + 1e-8)
        if self.kind == "pong":
            self._pong(step)
        elif self.kind == "dino":
            self._dino(step)
        else:
            self._ship(step, before)

    def _pong(self, step: float) -> None:
        self.opponent += min(0.45 * step, max(-0.45 * step, self.y - self.opponent))
        old_x = self.x
        self.x += self.vx * step
        self.y += self.vy * step
        if self.y <= 1 or self.y >= 22:
            self.vy *= -1
            self.y = min(22, max(1, self.y))
        for paddle_x, paddle_y in ((4, self.player), (59, self.opponent)):
            approaching = self.vx < 0 if paddle_x == 4 else self.vx > 0
            if approaching and min(old_x, self.x) <= paddle_x <= max(old_x, self.x):
                if abs(self.y - paddle_y) <= 3:
                    self.x = float(paddle_x)
                    self.vx *= -1.04
                    self.vx = min(1.5, max(-1.5, self.vx))
                    self.vy = (self.y - paddle_y) * 0.3 or 0.2
        if self.x < 0 or self.x > 63:
            self.score += int(self.x > 63)
            self.cpu_score += int(self.x < 0)
            self.x, self.y = 32, 12
            self.vx = self.rng.choice((-0.8, 0.8))
            self.over = self.score >= 7 or self.cpu_score >= 7

    def _dino(self, step: float) -> None:
        gravity = RISE_GRAVITY if self.jump_speed > 0 else FALL_GRAVITY
        self.jump_speed -= gravity * step
        self.jump = max(0.0, self.jump + self.jump_speed * step)
        if self.jump == 0:
            self.jump_speed = 0
        speed = RUN_SPEED + min(0.5, self.score * 0.03)
        moved = [(x - speed * step, y) for x, y in self.objects]
        self.score += sum(1 for (old, _), (new, _) in zip(self.objects, moved) if old >= 5 > new)
        self.objects = [(x, y) for x, y in moved if x > -3]
        if not self.objects or self.objects[-1][0] < 32:
            self.objects.append((64 + self.rng.randrange(12), 0))
        self.over = any(8 < x + 2 and x < 14 and self.jump < 4 for x, _ in self.objects)

    def _ship(self, step: float, before: int) -> None:
        if before < 1 <= self.ticks or self.ticks // 22 > before // 22:
            self.objects.append((self.rng.randrange(5, 59), 1))
        self.bullets = [(x, y - step) for x, y in self.bullets if y > 0]
        enemies = []
        for x, y in self.objects:
            y += (0.18 + min(0.15, self.score / 1000)) * step
            hit = next((b for b in self.bullets if abs(b[0] - x) <= 2 and abs(b[1] - y) <= 2), None)
            if hit is not None:
                self.bullets.remove(hit)
                self.score += 10
            elif y >= 23 or (y >= 18 and abs(x - self.player) <= 4):
                self.lives -= 1
            else:
                enemies.append((x, y))
        self.objects = enemies
        self.over = self.lives <= 0

    def pixels(self, scale: int = 1) -> list[list[int]]:
        width, height = WIDTH * scale, HEIGHT * scale
        grid = [[0] * width for _ in range(height)]

        def dot(x: float, y: float, color: int = 1) -> None:
            left, top = int(x * scale), int(y * scale)
            for py in range(max(0, top), min(height, top + scale)):
                for px in range(max(0, left), min(width, left + scale)):
                    grid[py][px] = color

        def sprite(x: float, y: float, rows: tuple[str, ...], color: int = 1) -> None:
            for dy, row in enumerate(rows):
                for dx, cell in enumerate(row):
                    if cell != " ":
                        dot(x + dx, y + dy, color)

        if self.kind == "pong":
            for y in range(0, HEIGHT, 3):
                dot(32, y, 3)
            for center, x, color in ((self.player, 3, 1), (self.opponent, 60, 2)):
                for offset in range(-2, 3):
                    dot(x, center + offset, color)
                    dot(x + 1, center + offset, color)
            sprite(self.x, self.y, ("xx", "xx"))
        elif self.kind == "dino":
            for cloud_x, cloud_y in ((18, 4), (47, 7)):
                sprite(cloud_x, cloud_y, ("  xxxx  ", "xxxxxxxx"), 3)
            for x in range(WIDTH):
                dot(x, 22, 3)
                if (x + self.ticks // 2) % 9 == 0:
                    dot(x, 23, 3)
            sprite(
                7,
                15 - self.jump,
                (
                    "    xxx",
                    "    x x",
                    "    xxx",
                    "x  xx  ",
                    "xxxxx  ",
                    " xxx   ",
                    " x x   " if self.ticks % 6 < 3 else "  x x  ",
                ),
            )
            for obstacle_x, _ in self.objects:
                sprite(obstacle_x, 18, (" x ", "xxx", " x ", " x "), 2)
        elif self.kind == "ship":
            for x, y in ((6, 3), (22, 14), (41, 5), (56, 17), (14, 19), (49, 10)):
                dot(x, (y + self.ticks // 4) % HEIGHT, 3)
            sprite(self.player - 3, 19, ("   x   ", "  xxx  ", "xxxxxxx", "x xxx x"))
            for enemy_x, enemy_y in self.objects:
                sprite(enemy_x - 2, enemy_y, ("x   x", " xxx ", "xxxxx"), 2)
            for bullet_x, bullet_y in self.bullets:
                dot(bullet_x, bullet_y)
                dot(bullet_x, bullet_y + 1)
        else:
            for y in range(5):
                for x in range(8):
                    cell = x, y
                    hit = cell in self.shots and cell in self.fleet
                    color = 2 if hit else 4
                    pattern = ("x x", " x ", "x x") if hit else ("xxx", "x x", "xxx")
                    if cell in self.shots and not hit:
                        pattern = ("   ", " x ", "   ")
                    if cell == self.cursor:
                        color = 1
                    sprite(9 + x * 6, 3 + y * 4, pattern, color)
        return grid
