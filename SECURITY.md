# Security Policy

Security fixes target the latest release on `main`.

## Generated code

The pipeline renders model-generated Python. Its validator detects reliability problems but does not provide process isolation. Treat generated code as untrusted and run jobs in constrained, ephemeral environments.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Include reproduction steps, affected versions, and impact. Do not publish the report before a fix is available.
