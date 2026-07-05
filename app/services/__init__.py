# This file marks `app/services` as a Python sub-package.
#
# The `services/` layer is reserved for business-logic helpers that sit between
# the HTTP routing layer (main.py) and the infrastructure layer (vector_db.py,
# embedder.py). For example: deduplication checks, prompt-building utilities,
# or any logic that shouldn't live directly inside a route handler.
