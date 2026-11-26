"""Independent still-image rounds using the existing RPS-1 application."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import live_app as app


def probability_panel(panel, game, footer):
    img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    if game.decision and game.revealed and game.phase == 'DECISION':
        dec = game.decision
        app.centred(d, 480, 'Optimal retention probabilities', app.F['bodyb'], app.PLUM)
        for i, side in enumerate(('LEFT', 'RIGHT')):
            p = float(dec['keep_probs'][i])
            y = 515 + 48*i
            app.centred(d, y, f"{side} ({app.NAME[dec['mine'][i]['letter']]}) : {p:.1%}", app.F['body'], app.PLUM)
            app.rounded(d, (65,y+27,375,y+35), (255,228,240), r=4)
            if p > 0:
                app.rounded(d, (65,y+27,65+max(1,int(310*p)),y+35), app.HOT, r=4)
        app.centred(d, 615, f"Guaranteed expected payoff: {dec['guarantee']:+.3f}", app.F['small'], app.PLUM)
    app.centred(d, 660, footer, app.F['small'], app.PLUM)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rotate', action='store_true')
    args = parser.parse_args()
    folder = Path(args.source)
    files = sorted(folder.glob('photo*.jpg')) or sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not files:
        raise SystemExit('No photographs found.')
    ns = app.load_notebook_code()
    vision = app.Vision(ns)
    game = app.Game(ns, seed=args.seed)
    window = 'RPS-1 Photo Demo'
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
            hands = vision.analyse(frame)
            for _ in range(app.STEADY):
                game.update(hands, time.time())
            while True:
                if game.decision:
                    dec = game.decision
                    print(f"Photo {index+1}: {files[index].name}; keep probabilities {dec['keep_probs']}; remove {('LEFT','RIGHT')[1-dec['keep']]}")
                else:
                    print('Invalid detection:', files[index])
                while True:
                    video, box = app.fit_video(frame)
                    video = app.draw_video(video, hands, game, box)
                    panel = app.draw_panel(game, hands, 0, 0)
                    img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
                    d = ImageDraw.Draw(img)
                    d.rectangle((0,0,app.PANEL,65), fill=app.PINK_TOP)
                    app.centred(d, 30, f'PHOTO ROUND {index+1}/{len(files)}', app.F['smallb'], app.PLUM)
                    panel = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    panel = probability_panel(panel, game, 'N: next | S: resample | R: restart | Q: quit')
                    cv2.imshow(window, np.hstack([video,panel]))
                    key = cv2.waitKey(30) & 255
                    if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                        return
                    if key in (ord('n'),ord('r'),ord('s')):
                        break
                if key == ord('s'):
                    if game.decision:
                        game.decision['keep'] = int(game.rng.choice(2,p=game.decision['keep_probs']))
                    continue
                if key == ord('r'):
                    index = 0
                    game = app.Game(ns,seed=args.seed)
                else:
                    index += 1
                break
        print('All photo rounds complete. Actual outcomes not observed.')
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
