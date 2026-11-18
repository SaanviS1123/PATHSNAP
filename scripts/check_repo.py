"""Check tracked submission files without printing suspected secret contents."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/name for name in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if name]
paths=[p for p in paths if p.is_file()]
patterns=[re.compile(r'AKIA[0-9A-Z]{16}'),re.compile(r'gh[pousr]_[A-Za-z0-9]{36,}'),
          re.compile(r'sk-proj-[A-Za-z0-9_-]{40,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
findings=[]
for path in paths:
    name=str(path.relative_to(ROOT))
    if name.startswith('data/raw/') and path.name!='.gitkeep':findings.append(name+': raw data tracked')
    if path.name=='.env':findings.append(name+': local environment tracked')
    if path.stat().st_size>=95_000_000:findings.append(name+': near GitHub file-size limit')
    if path.stat().st_size>2_000_000:continue
    try:text=path.read_text()
    except (UnicodeDecodeError,ValueError):continue
    if any(pattern.search(text) for pattern in patterns):findings.append(name+': possible credential pattern')
largest=sorted(paths,key=lambda path:path.stat().st_size,reverse=True)[:8]
result=dict(tracked_files_checked=len(paths),secret_scan='common credential patterns in UTF-8 files up to 2 MB; not a guarantee',
            findings=findings,largest_files=[dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size) for path in largest])
(ROOT/'results/metrics/repo_hygiene.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if findings:raise SystemExit(1)
