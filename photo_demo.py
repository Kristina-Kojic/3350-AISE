"""Controlled RPS-1 photo demo. Run beside live_app.py."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import live_app as app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rotate', action='store_true')
    args = parser.parse_args()
    folder = Path(args.source)
    files = sorted(folder.glob('photo*.jpg'))
    if not files:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise SystemExit('No photographs found.')
    ns = app.load_notebook_code()
    vision = app.Vision(ns)
    game = app.Game(ns, seed=args.seed)
    opponent_rng = np.random.default_rng(args.seed + 1)
    window = 'RPS-1 Photo Demo (simulated)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, app.W, app.H)
    index = 0
    try:
        while index < len(files):
            game.new_round()
            frame = cv2.imread(str(files[index]))
            if frame is None:
                print('Unreadable:', files[index]); index += 1; continue
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            start = time.perf_counter()
            hands = vision.analyse(frame)
            ms = 1000 * (time.perf_counter() - start)
            for _ in range(app.STEADY):
                game.update(hands, time.time())
            valid = game.decision is not None
            removal = ('LEFT', 'RIGHT')[1-game.decision['keep']] if valid else 'N/A'
            print(f'Photo {index+1}: {files[index].name}; remove {removal}')
            if valid:
                print('Keep probabilities:', game.decision['keep_probs'])
            while True:
                video, box = app.fit_video(frame)
                # The photograph is unchanged; do not mark all four hands as winners.
                phase = game.phase
                if phase == 'RESULT':
                    game.phase = 'DECISION'
                try:
                    video = app.draw_video(video, hands, game, box)
                finally:
                    game.phase = phase
                panel = app.draw_panel(game, hands, 0, ms)
                img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(img)
                app.centred(d, 8, f'PHOTO {index+1}/{len(files)} | SIMULATION', app.F['smallb'], app.PLUM)
                score = game.score
                app.centred(d, 485, f"W {score['win']}   T {score['tie']}   L {score['loss']}", app.F['h2'], app.PLUM)
                lead = score['win'] - score['loss']
                app.centred(d, 520, f'Ahead by {lead}' if lead > 0 else f'Behind by {-lead}' if lead < 0 else 'Even', app.F['bodyb'], app.PLUM)
                if phase == 'RESULT':
                    app.centred(d, 550, f"{app.NAME[game.result['me']]} vs {app.NAME[game.result['them']]}", app.F['body'], app.PLUM)
                    app.centred(d, 580, f'Removed {removal} | N: next photo', app.F['smallb'], app.PLUM)
                elif valid:
                    app.centred(d, 555, 'SPACE: simulate showdown', app.F['smallb'], app.PLUM)
                else:
                    app.centred(d, 555, 'Invalid detection | N: skip', app.F['smallb'], app.PLUM)
                app.centred(d, 625, 'R: restart   Q: quit', app.F['small'], app.SOFT)
                panel = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv2.imshow(window, np.hstack([video, panel]))
                key = cv2.waitKey(30) & 255
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key == ord('r'):
                    game = app.Game(ns, seed=args.seed)
                    opponent_rng = np.random.default_rng(args.seed + 1)
                    index = 0
                    break
                if key == 32 and valid and phase == 'DECISION':
                    # Opponent chooses uniformly; score is based on recognized gestures.
                    opp = int(opponent_rng.integers(2))
                    remaining = [dict(game.decision['mine'][game.decision['keep']]), dict(game.decision['theirs'][opp])]
                    for _ in range(app.STEADY):
                        game.update(remaining, time.time())
                    print('Simulated result:', game.result, 'Score:', game.score)
                if key == ord('n') and (game.phase == 'RESULT' or not valid):
                    if index == len(files)-1:
                        print('Final simulated score:', game.score, '(R to restart or Q to quit)')
                    else:
                        index += 1
                        break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
