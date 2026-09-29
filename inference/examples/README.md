# Custom controls

Use your own first-frame image and one grayscale/binary mask for each visible
character. The following writes a complete 25-step controls file for a two-character
HNM clip; it deliberately uses a published NPC sentence so no T5 encoder is needed.

```python
import json

controls = {
    "subjects": [
        {"name": "P1", "actions": ["no_op"] * 8 + ["throw_shuriken"] + ["no_op"] * 16},
        {"name": "P2", "prompt": "If the target player throws a shuriken, then jump to dodge it."},
    ]
}
with open("controls.json", "w") as f:
    json.dump(controls, f, indent=2)
```

For SF3, select an action such as `light_punch` and use a sentence from the SF3
checkpoint's `text/control_text.json`, e.g.:

```text
If the opponent launches a threatening projectile, then jump once to evade it when it gets close.
```

The checkpoint JSON's `spec.actions` and `spec.npc_prompts` list exact supported
cached strings. HNM and SF3 have different action vocabularies. Reassigning P1/P2
controls changes their roles; keep their mask positions matched to the same
physical character. HNM supports two to six characters; the released SF3 model
was trained on two fighters.
