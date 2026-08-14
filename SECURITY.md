# Security Policy

## Supported version

Security fixes are applied to the latest version on the default branch.

## Reporting a vulnerability

Use the repository's **Security** tab and select **Report a vulnerability** when private vulnerability reporting is available. If it is unavailable, contact the maintainer privately through the GitHub profile before creating a public issue. Do not include credentials, private source code, exploit payloads, or personal data in a public report.

Include the affected URL or file, reproduction steps, impact, and a minimal proof of concept. Allow reasonable time for investigation before public disclosure.

## Architecture

PyMap is a static GitHub Pages application. It has no server-side application, account system, database, Supabase project, API keys, analytics, or cookies. Python source selected by a visitor is analyzed inside that visitor's browser and is not uploaded by PyMap.

The page downloads version-pinned Pyodide and D3 scripts from jsDelivr. Those scripts are protected with Subresource Integrity hashes and restricted by a Content Security Policy.
