"""
09_azure_openai.py — Azure OpenAI Service via edgequake-litellm.

Use the ``azure/`` provider prefix to route requests to your Azure OpenAI
deployment.  The model portion is the deployment name (not the base model
name) as configured in Azure AI Foundry.

Required environment variables (standard variant):

    AZURE_OPENAI_ENDPOINT        https://<resource>.openai.azure.com
    AZURE_OPENAI_API_KEY         your API key
    AZURE_OPENAI_DEPLOYMENT_NAME your-gpt4o-deployment

Or the CONTENTGEN variant (preferred for content-gen workloads):

    AZURE_OPENAI_CONTENTGEN_API_ENDPOINT
    AZURE_OPENAI_CONTENTGEN_API_KEY
    AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT

Run:
    python examples/09_azure_openai.py
"""
from __future__ import annotations

import os
import sys
import json

# ---------------------------------------------------------------------------
# Guard — skip gracefully when Azure creds are absent
# ---------------------------------------------------------------------------

def _azure_available() -> bool:
    has_standard = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    has_contentgen = (
        os.environ.get("AZURE_OPENAI_CONTENTGEN_API_KEY")
        and os.environ.get("AZURE_OPENAI_CONTENTGEN_API_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT")
    )
    return bool(has_standard or has_contentgen)


if not _azure_available():
    print("[09_azure_openai] Azure credentials not found — skipping example.")
    print("Set AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT_NAME")
    print("or the CONTENTGEN equivalents and re-run.")
    sys.exit(0)

try:
    import edgequake_litellm as litellm
except ImportError as exc:
    print(f"edgequake_litellm not installed: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Section 1 — Provider detection
# ---------------------------------------------------------------------------
print("=" * 60)
print("Section 1: Provider auto-detection")
print("=" * 60)

detected = litellm.core.detect_provider()
print(f"Auto-detected provider: {detected}")

# ---------------------------------------------------------------------------
# Section 2 — Simple chat via azure/ prefix
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Section 2: Simple chat (azure/<deployment>)")
print("=" * 60)

# Determine the deployment name from env
deployment = (
    os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    or os.environ.get("AZURE_OPENAI_CONTENTGEN_MODEL_DEPLOYMENT")
    or "gpt-4o"  # fallback placeholder
)

resp = litellm.completion(
    model=f"azure/{deployment}",
    messages=[{"role": "user", "content": "Write a single sentence about cloud AI."}],
)
print(f"Response: {resp.content}")
print(f"Model   : {resp.model}")
print(f"Tokens  : {resp.usage.total_tokens}")

# ---------------------------------------------------------------------------
# Section 3 — JSON mode
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Section 3: JSON mode")
print("=" * 60)

resp_json = litellm.completion(
    model=f"azure/{deployment}",
    messages=[{
        "role": "user",
        "content": (
            "Return a JSON object with keys: city (string), population (int). "
            "Example city: Paris."
        ),
    }],
    response_format={"type": "json_object"},
)
try:
    data = json.loads(resp_json.content)
    print(f"Parsed JSON: {data}")
except json.JSONDecodeError:
    print(f"Raw response: {resp_json.content}")

# ---------------------------------------------------------------------------
# Section 4 — Streaming
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Section 4: Streaming")
print("=" * 60)

print("Streamed: ", end="", flush=True)
for chunk in litellm.stream_completion(
    model=f"azure/{deployment}",
    messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# ---------------------------------------------------------------------------
# Section 5 — List available providers
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Section 5: List supported providers")
print("=" * 60)

providers = litellm.core.list_providers()
print("Supported providers:", providers)
assert "azure" in providers, "azure should appear in list_providers()"
print("✓ 'azure' is listed")

print("\nAll Azure OpenAI sections completed successfully.")
