from __future__ import annotations

import random

import pytest

from splinter.arcade import GAMES, HEIGHT, WIDTH, Arcade


@pytest.mark.parametrize("kind", GAMES)
def test_games_stay_inside_the_board(kind: str) -> None:
    game = Arcade(kind, rng=random.Random(4))
    for n in range(250):
        game.key(("space", "left", "up", "down", "right")[n % 5])
        game.tick()
    pixels = game.pixels()
    assert len(pixels) == HEIGHT
    assert all(len(row) == WIDTH for row in pixels)
    assert {color for row in pixels for color in row} <= {0, 1, 2, 3, 4}


def test_pong_paddle_bounces_ball_and_miss_scores() -> None:
    game = Arcade("pong", x=4.5, y=12, vx=-0.8, vy=0)
    game.tick()
    assert game.vx > 0
    assert game.cpu_score == 0
    game.x, game.y, game.vx = 0.1, 1, -0.8
    game.tick()
    assert game.cpu_score == 1
    assert game.status == "YOU 00   CPU 01"
    game.x, game.vx, game.score = 63, 0.8, 6
    game.tick()
    assert game.over and game.result == "You win!"


def test_dino_jump_clears_obstacle_but_ground_collision_ends_game() -> None:
    game = Arcade("dino")
    game.objects = [(14, 0)]
    game.tick()
    assert game.over
    game = Arcade("dino")
    game.objects = [(18, 0)]
    game.key("space")
    for _ in range(25):
        game.tick()
    assert not game.over
    assert game.score == 1


def test_ship_shoots_enemies_and_loses_shields_on_misses() -> None:
    game = Arcade("ship", ticks=2)
    game.objects = [(32, 17)]
    game.key("space")
    game.tick()
    assert game.score == 10
    assert not game.objects
    assert not game.bullets
    for _ in range(3):
        game.objects = [(5, 23)]
        game.tick()
    assert game.over and game.lives == 0


def test_naval_repeat_shots_free_and_all_hits_win() -> None:
    game = Arcade("naval", rng=random.Random(5))
    assert len(game.fleet) == 7
    assert all(0 <= x < 8 and 0 <= y < 5 for x, y in game.fleet)
    for cell in sorted(game.fleet):
        game.cursor = cell
        game.key("space")
        game.key("space")
    assert game.over and game.result == "You win!"
    assert len(game.shots) == 7


def test_naval_exhausting_shots_loses() -> None:
    game = Arcade("naval", rng=random.Random(5))
    misses = [(x, y) for x in range(8) for y in range(5) if (x, y) not in game.fleet]
    for cell in misses[:18]:
        game.cursor = cell
        game.key("space")
    assert game.over and game.result == "Game over"


@pytest.mark.parametrize("fps", [60, 120])
def test_pong_speed_is_independent_of_frame_rate(fps: int) -> None:
    slow = Arcade("pong")
    fast = Arcade("pong")
    for _ in range(20):
        slow.tick(1 / 20)
    for _ in range(fps):
        fast.tick(1 / fps)
    assert fast.x == pytest.approx(slow.x)
    assert fast.y == pytest.approx(slow.y)
    assert fast.ticks == slow.ticks


@pytest.mark.parametrize("x, vx", [(58.9, 0.8), (4.1, -0.8)])
def test_pong_leaves_the_paddle_after_bouncing(x: float, vx: float) -> None:
    game = Arcade("pong", x=x, vx=vx, y=12, vy=0)
    game.tick(1 / 60)
    assert game.vx * vx < 0
    for _ in range(10):
        game.tick(1 / 60)
        assert game.vx * vx < 0
    assert 6 < game.x < 57


def test_pong_keeps_moving_through_many_rallies() -> None:
    game = Arcade("pong", vy=0)
    bounces = 0
    previous_vx = game.vx
    for _ in range(60 * 45):
        game.player = game.y
        game.opponent = game.y
        game.tick(1 / 60)
        if game.vx * previous_vx < 0:
            bounces += 1
        previous_vx = game.vx
    assert bounces >= 10
    assert game.score == game.cpu_score == 0


@pytest.mark.parametrize("fps", [60, 120])
def test_ship_spawns_once_per_interval(fps: int) -> None:
    game = Arcade("ship", rng=random.Random(2))
    for _ in range(fps):
        game.tick(1 / fps)
    assert len(game.objects) == 1
    for _ in range(fps):
        game.tick(1 / fps)
    assert len(game.objects) == 2


@pytest.mark.parametrize("kind", ["dino", "ship"])
def test_scaled_sprites_use_fractional_positions(kind: str) -> None:
    game = Arcade(kind)
    game.objects = [(40.1, 5.1)]
    small_before, large_before = game.pixels(), game.pixels(2)
    game.objects = [(40.6, 5.6)]
    assert game.pixels() == small_before
    assert game.pixels(2) != large_before
    assert len(large_before) == HEIGHT * 2
    assert all(len(row) == WIDTH * 2 for row in large_before)
