# FaceFunBot

A playful photo compatibility algorithm for friends.

It compares two face photos or two palm photos using lightweight image keypoints,
color mood, brightness, line texture, and simple composition metrics, then prints
a funny couple verdict. The scoring is for entertainment only, not identity
recognition, palmistry, or a serious relationship claim.

## Setup

The prototype uses only `numpy` and `Pillow`.

```bash
python3 -m pip install -r requirements.txt
```

## Run The CLI

```bash
python3 couple_match.py path/to/person_a.jpg path/to/person_b.jpg
```

Palm-photo mode:

```bash
python3 couple_match.py path/to/palm_a.jpg path/to/palm_b.jpg --mode palm
```

Optional JSON output:

```bash
python3 couple_match.py person_a.jpg person_b.jpg --mode face --json
```

## Run The Telegram Bot

1. Create a bot with BotFather and copy its token.
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Start the bot:

```bash
export TELEGRAM_BOT_TOKEN="123456:your-token-here"
python3 telegram_bot.py
```

By default, every received photo is also saved on the server:

- face mode: `uploaded_photos/faces/`
- palm mode: `uploaded_photos/palms/`

To choose a different archive directory:

```bash
export FACEFUNBOT_UPLOAD_DIR="/path/to/photo-archive"
python3 telegram_bot.py
```

In Telegram:

- `/face` compares two face photos.
- `/palm` compares two palm photos.
- `/reset` clears the current pair.

Send exactly two photos after choosing the mode. The bot replies with the same
fun couple-match report as the CLI.

## How It Works

1. Converts each photo to grayscale.
2. Detects corner-like keypoints with a Harris response.
3. Builds small normalized patch descriptors around those points.
4. Matches descriptors between photos with a ratio test.
5. In palm mode, compares line direction and crease density.
6. Combines visual similarity with color, lighting, and composition.
7. Generates a silly explanation and relationship archetype.

## Notes

- Use clear photos with one main face or one open palm each for best comedy value.
- Use `--mode palm` for palms so the verdict talks about palm lines instead of
  face rhythm.
- If a photo has too few detectable details, the algorithm still produces a
  verdict using color and composition fallback signals.
- This is intentionally fun, transparent, and tweakable.
