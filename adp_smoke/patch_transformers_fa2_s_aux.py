"""Idempotent patch: transformers 5.6.0 FA2 integration crashes on models without
attention sinks — integrations/flash_attention.py does `s_aux=s_aux.to(query.dtype)`
unconditionally, but Qwen3.5's full-attention layers pass s_aux=None
(AttributeError: 'NoneType' object has no attribute 'to').

Guards the .to() with a None check. Run after any transformers reinstall:
    python patch_transformers_fa2_s_aux.py
"""
import sys
from pathlib import Path

import transformers

target = Path(transformers.__file__).parent / "integrations" / "flash_attention.py"
src = target.read_text()

BROKEN = "s_aux=s_aux.to(query.dtype),  # FA only accepts half precision"
FIXED = "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,  # FA only accepts half precision (patched: None-guard)"

if FIXED in src:
    print(f"already patched: {target}")
    sys.exit(0)
if BROKEN not in src:
    print(f"ERROR: expected line not found in {target} — transformers version changed? Inspect manually.")
    sys.exit(1)

target.write_text(src.replace(BROKEN, FIXED))
print(f"patched: {target}")
