# Annotation specification

- **Task:** `sentiment_classification`
- **Modality:** English review text
- **Unit of annotation:** one complete product or game review
- **Output:** exactly one of `negative` or `positive`

## Class definitions

### `negative`

The author's overall judgement is unfavourable: dissatisfaction, a failed experience, a warning not to buy/play, or criticism that outweighs praise.

### `positive`

The author's overall judgement is favourable: satisfaction, recommendation, enjoyment, or praise that outweighs criticism.

## Decision rules

1. Label the author's overall conclusion, not isolated sentiment words.
2. Respect negation, contrast (`but`, `although`, `yet`) and sarcasm when evident.
3. For mixed reviews, use the final recommendation or the dominant judgement.
4. Judge the reviewed item, not delivery/support, unless that drives the conclusion.
5. If evidence is genuinely balanced, keep the auto-label but flag it for review.

## Real examples from this dataset

### `negative` examples

- `000cd97def182d2bd4d3` — “everything after this update has way to much health at the start of the game, elites have around 10k hp at level 5 my force wave hits for like 100 damage what the hell are these numbers? later on the game feels better as you get more gear, devoitons and other things to help bala…”
- `00707958cf27af2c7d3c` — “I really really want to like this game. This futuristic Sci Fi Open World Setting is exactly what I want in game. I played hundreds of hours. BUT the amount of gamebreaking bugs in the game after years of release is just Bethesda spitting in every buyers face. And it's getting w…”
- `00aaaecae0210e213e00` — “not really worth it I wore this for my wedding since I hadn't lost all of my pregnancy weight. It was great for a little bit. But then started getting a crease around my waist. I can't wear it anymore. Plus it made me feel very sick if I wore it for more than a couple hours. The…”

### `positive` examples

- `000e9f2a4199848013a6` — “these are the least hideous stockings ever! Hi. I ordered a whole selection of stockings from diverse makers, just in case, and these were the classiest as well as the most sheer. They don't cut into your upper thighs, they only have a hint of stick-um at the top, and so you rea…”
- `0279b3a8e9be7ad28cb2` — “It had been a while since I'd played an open-world game, all because the last few I played weren't very enjoyable, having to wander all the time. However, this game is different; instead of walking or driving, you swing over Manhattan, which makes the game much more distinctive…”
- `02b2b4aa5c0f4dd48aed` — “Good movie I like all of the transformers movies. This one is fun too. I would recommend it to any fan of the first two movies.”

## Boundary cases from this dataset

- `2f05e4a2914e9d39b33b` (low_confidence) — “Based on the reviews here I bought one and I'm glad I did! This VCR/DVD was an early Christmas present to myself after deciding to join the rest of the world in DVD-land but not wanting to let go of my VHS movies quite yet. Based on the reviews and price, and because I own a JVC…”
- `f904c7ecb9b8a260f2ef` (low_confidence) — “A solid, tactical shooter at its core and certainly worth thinking about buying, even at full price. There's plenty of enjoyment to be had. That being said the enjoyment mostly comes from the interaction with other players rather than the game itself. Many games are like that th…”
- `a083e1aed00f587e79c7` (low_confidence) — “Just Okay This book is 'just okay.' I came away from reading the book not knowing too much about the Amish. Sue prefers to keep most of the information to herself. She does, however, make sure she gets in adequate writing about all of her 'accomplishments' -- and makes apparent…”
- `453e7eef87cacb6977a4` (low_confidence) — “absolutely love this game. LOTS to learn, IMO. But man its so bada$$ dude. Looks amazing, plays great, ots of interactive in game menus, i mean its just alot, but its not just a chore, everything has a function, you really feel like your operating a spaceship dude lol. Get this…”
- `aef3a167b3eabd5e5eea` (low_confidence) — “I used to love this game. Elite Dangerous was once one of the best space simulators ever made. I still remember the wonder I felt the first time I launched into space, jumping from system to system, staring at the full explorable galaxy map, and setting off on long voyages into…”

## Human-review procedure

Review every queued row in full context. Enter `negative` or `positive`; an empty correction explicitly confirms the auto-label. Record the reviewer name and preserve `record_id` so corrections can be validated and merged safely.
