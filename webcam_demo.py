"""Existing live application with probability display added, without modifying it."""
import live_app as app
from photo_demo import probability_panel

original_panel = app.draw_panel

def draw_panel(game, hands, fps, vision_ms):
    panel = original_panel(game, hands, fps, vision_ms)
    score = game.score
    return probability_panel(panel, game, f"Actual results: W {score['win']} | T {score['tie']} | L {score['loss']}")

if __name__ == '__main__':
    app.draw_panel = draw_panel
    app.main()
