# Notification Router

## Run

From the repository root, using Python 3.10+ and only the standard library:

```powershell
python code/main.py
python code/evaluation/main.py
```

The first command writes `output.csv` at the repository root. The second
command checks the submission contract and reports action/type accuracy on the
provided solved examples. An exit code of zero means the output is valid.

Optional OCR: if `tesseract` is available on `PATH`, the router reads image
content locally. Without it, it still runs using message captions and metadata.
No network access, API key, or third-party Python package is required.

## Design

`main.py` combines deterministic safety policy with personalized retrieval:

- Credential/OTP coercion and prompt-injection-like instructions always mute.
- Group mute state is respected except for time-sensitive operational updates
  and direct mentions.
- Sender roles, group type, business verification/domain consistency, opt-outs,
  and user-business activity establish context.
- Historical messages are ranked by lexical similarity, source identity, and
  category. Their opens, replies, dismissals, mutes, and reports influence the
  route and are returned as evidence IDs.
- Image captions are enriched with local OCR through `media.py` when available.

The router deliberately emits a short, decision-specific reason and a bounded
confidence for every row so the output remains auditable in the AI Judge
interview.

## Files

- `main.py` — loads the participant-facing dataset and creates predictions.
- `media.py` — optional local image OCR and media-reference validation.
- `evaluation/main.py` — sample evaluation and output-schema validation.
