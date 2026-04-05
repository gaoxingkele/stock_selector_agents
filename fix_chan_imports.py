import os, re

root_dir = 'chan'
subpkgs = ['Bi','KLine','BuySellPoint','ChanConfig','Common','DataAPI','Math','Seg','ZS','Script','Combiner','App','Debug','Plot','ChanModel']

for entry in os.walk(root_dir):
    root = entry[0]
    files = entry[2]
    normalized = root.replace(chr(92), '/')
    actual_depth = max(0, normalized.count('/'))
    dots = '.' * (actual_depth + 1)

    for f in files:
        if not f.endswith('.py') or f == 'setup.py':
            continue
        fpath = os.path.join(root, f)
        with open(fpath, 'r', encoding='utf-8') as fh:
            content = fh.read()
        original = content

        for pkg in subpkgs:
            # Pattern 1: from BuySellPoint.X import Y  (no leading dot - Python 2 style)
            # Pattern 2: from .BuySellPoint.X import Y  (one dot - wrong)
            # Pattern 3: from ..BuySellPoint.X import Y  (two dots - wrong depth)
            # Replace ALL of these with: from <correct_dots>Pkg.X import Y

            # Match: from [+dots]pkg[.suffix][ import Y]
            # Group 1: indent, Group 2: existing dots (0-2), Group 3: pkg, Group 4: rest
            pat = re.compile(
                r'^(\s*)from (\.*)(' + pkg + r')(\.?\S*)(\s+import\s+.*)?',
                re.MULTILINE
            )
            def repl(m, d=dots, p=pkg):
                rest = m.group(4) + (m.group(5) or '')
                return m.group(1) + 'from ' + d + p + rest
            content = pat.sub(repl, content)

        if content != original:
            with open(fpath, 'w', encoding='utf-8') as fh:
                fh.write(content)
            print(f'Fixed depth={actual_depth}: {fpath}')
