# Security, Privacy, and External Model Use

## Secrets

- Store API keys in environment variables or an approved secrets manager.
- Never commit keys, tokens, credentials, or private endpoints.
- Do not print secrets in logs or audit reports.
- Provide a safe `.env.example` containing placeholder values only.

## Database and model boundaries

- The application controls database access.
- Use parameterized queries and validate inputs.
- Do not give a language model unrestricted SQL execution or arbitrary filesystem access.
- Treat database values, filenames, video metadata, and user-supplied text as untrusted input.
- Limit the records and fields sent to hosted inference to what is needed for the request.
- Do not send the entire database or full video by default.

## Video and identity data

- Inspect the data-handling implications before sending frames or identity-linked data to an external service.
- Prefer the minimum relevant frames and metadata.
- Avoid including unnecessary names or personal details in model prompts.
- Respect the data owner's permissions and any applicable institutional rules.

## Model output

- Treat model output as untrusted.
- Validate structured outputs before use.
- Do not allow model output to trigger arbitrary code, shell commands, SQL, or unrestricted file access.
- Preserve uncertainty in identity matching.
- Separate recorded evidence from model-generated interpretation.

## Failure behavior

- External API failures must not corrupt local records.
- Use bounded timeouts and retries.
- Never claim a cloud operation succeeded if the provider did not confirm it.
