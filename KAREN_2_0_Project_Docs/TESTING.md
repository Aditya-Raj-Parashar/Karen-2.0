# Testing and Benchmarking

## Principles

- Test existing behavior before changing it.
- Use the same clips and comparable settings for baseline and candidate comparisons.
- Separate unit tests, integration tests, end-to-end tests, and manual demo checks.
- Record environment, model version, configuration, input clips, and exact results.
- Never invent or extrapolate unmeasured accuracy improvements.
- If ground truth is missing, label conclusions as qualitative.

## Vision metrics to consider

Use only metrics appropriate to the available annotations:
- Detection precision and recall.
- MOTA/HOTA or other suitable tracking metrics when supported by ground truth.
- IDF1 and identity switches.
- Track fragmentation and continuity.
- Cross-camera identity match precision/recall.
- FPS and end-to-end latency.
- CPU, RAM, GPU memory, and network usage.
- Cold-start time and recovery after failures.

## Q&A tests

Test:
- Correct question with available evidence.
- No matching records.
- Ambiguous person identity.
- Uncertain cross-camera match.
- Invalid or out-of-range time range.
- Contradictory or incomplete records.
- Prompt injection contained in database text.
- Token Factory timeout, rate limit, invalid credentials, and server error.
- Offline mode with no network.
- Answer references point to valid camera IDs and timestamps.
- The model does not invent actions or identity links.

## Selective image analysis tests

Test:
- Valid recording and timestamp.
- Timestamp outside video duration.
- Missing/unreadable video.
- Frame extraction errors.
- Image-size and frame-count limits.
- Unsupported image-input model.
- API failure.
- Inconclusive image.
- Distinction between visual observation and tracking inference.

## Test report template

- Date and commit:
- Environment:
- Models and versions:
- Configuration:
- Test inputs:
- Commands run:
- Expected results:
- Actual results:
- Pass/fail:
- Metrics:
- Known limitations:
