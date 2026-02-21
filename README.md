# simplebayes
A memory-based, optional-persistence naive Bayesian text classification package and web API for Python.

---

## Why?
```
Bayesian text classification is useful for things like spam detection,
sentiment determination, and general category routing.

You gather representative samples for each category, train the model,
then classify new text based on learned token patterns.

Once the model is trained, you can:
- classify input into a best-fit category
- inspect relative per-category scores
- persist and reload model state
```

## Installation
Requires Python 3.10 or newer.

```
$ git clone https://github.com/hickeroar/simplebayes.git
$ cd simplebayes
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

If you only want to use simplebayes as a library:
```
$ pip install simplebayes
```

---

## Run as an API Server
```
$ simplebayes-server --port 8000
```

CLI options:
```
--host        Host interface to bind. (default: 0.0.0.0)
--port        Port to bind. (default: 8000)
--auth-token  Optional bearer token for non-probe endpoints.
```

Environment variable equivalents:
```
SIMPLEBAYES_HOST
SIMPLEBAYES_PORT
SIMPLEBAYES_AUTH_TOKEN
```

When `--auth-token` is configured, all API endpoints except `/healthz` and `/readyz` require:
```
Authorization: Bearer <token>
```

## Use as a Library in Your App

Import and create a classifier:
```python
from simplebayes import SimpleBayes

classifier = SimpleBayes()

classifier.train("spam", "buy now limited offer click here")
classifier.train("ham", "team meeting schedule for tomorrow")

classification = classifier.classify_result("limited offer today")
print(f"category={classification.category} score={classification.score}")

scores = classifier.score("team schedule update")
print(scores)

classifier.untrain("spam", "buy now limited offer click here")
```

Persistence example:
```python
from simplebayes import SimpleBayes

classifier = SimpleBayes()
classifier.train("spam", "buy now limited offer click here")

classifier.save_to_file("/tmp/simplebayes-model.json")

loaded = SimpleBayes()
loaded.load_from_file("/tmp/simplebayes-model.json")
print(loaded.classify_result("limited offer today"))
```

Notes for library usage:
- Classifier operations are thread-safe.
- Scores are relative values; compare scores within the same model.
- Default tokenization applies Unicode NFKC normalization, lowercasing, non-word splitting, and English stemming.
- Category names accepted by `train`/`untrain` match `^[-_A-Za-z0-9]{1,64}$`.

Stream APIs are available:
- `save(stream)`
- `load(stream)`

File API notes:
- `save_to_file("")` and `load_from_file("")` use `/tmp/simplebayes-model.json`.
- Provided file paths must be absolute.

## Development Checks
```
$ ./.venv/bin/pytest tests/ --cov=simplebayes --cov-fail-under=100 -v
$ ./.venv/bin/flake8 simplebayes tests
$ ./.venv/bin/pylint simplebayes tests --exit-zero
```

---

## Using the HTTP API

### API Notes
- Category names in `/train/{category}` and `/untrain/{category}` must match `^[-_A-Za-z0-9]{1,64}$`.
- Request body size is capped at 1 MiB on text endpoints.
- Error responses for auth/size are JSON:
  - `{"error":"unauthorized"}`
  - `{"error":"request body too large"}`
- The HTTP service stores classifier state in memory; process restarts clear training data.

### Common Error Responses
| Status | When |
| --- | --- |
| `401` | Missing/invalid bearer token when auth is enabled |
| `405` | Wrong HTTP method |
| `413` | Request body exceeds 1 MiB |
| `422` | Invalid category route format |

### Training the Classifier

##### Endpoint:
```
/train/{category}
Example: /train/spam
Accepts: POST
Body: raw text/plain
```

Example:
```bash
curl -s -X POST "http://localhost:8000/train/spam" \
  -H "Content-Type: text/plain" \
  --data "buy now limited offer click here"
```

### Untraining the Classifier

##### Endpoint:
```
/untrain/{category}
Example: /untrain/spam
Accepts: POST
Body: raw text/plain
```

### Getting Classifier Status

##### Endpoint:
```
/info
Accepts: GET
```

Example response:
```json
{
  "categories": {
    "spam": {
      "tokenTally": 6,
      "probNotInCat": 0,
      "probInCat": 1
    }
  }
}
```

### Classifying Text

##### Endpoint:
```
/classify
Accepts: POST
Body: raw text/plain
```

Example response:
```json
{
  "category": "spam",
  "score": 3.2142857142857144
}
```

### Scoring Text

##### Endpoint:
```
/score
Accepts: POST
Body: raw text/plain
```

Example response:
```json
{
  "spam": 3.2142857142857144,
  "ham": 0.7857142857142857
}
```

### Flushing Training Data

##### Endpoint:
```
/flush
Accepts: POST
Body: raw text/plain (optional)
```

Example response:
```json
{
  "success": true,
  "categories": {}
}
```

### Health and Readiness
##### Liveness endpoint
```
/healthz
Accepts: GET
```

##### Readiness endpoint
```
/readyz
Accepts: GET
```

`/healthz` and `/readyz` are intentionally unauthenticated even when API auth is enabled.

## Operational Notes
- The HTTP server is in-memory by default; deploys/restarts wipe trained state.
- Use `save_to_file` and `load_from_file` in library workflows to persist/reload model state.
- `/readyz` returns `200` while accepting traffic and `503` when draining during shutdown.

## License
MIT, see `LICENSE`.
