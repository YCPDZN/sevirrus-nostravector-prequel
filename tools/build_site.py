"""Build the GitHub Pages site from the single-file game.

The game is written as one self-contained HTML file with every model and voice clip embedded
as base64. That file is too big to publish in some places, so for the website this script moves
each embedded asset out into its own file under assets/ and leaves a small loader in the page:
it fetches the assets, puts them back where the game expects them, then starts the game.

usage: python3 tools/build_site.py path/to/sevirrus-menu.html  (writes index.html and assets/)
"""
import base64, os, re, sys

HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>[hidden]{display:none!important} body{margin:0}</style>\n')

LOADER = r'''<script>
/* Asset loader (website build): fetch every model and voice file, hand them to the game as it
   expects them (base64 text inside their <script> tags), then start the game's 3D module. */
(() => {
  const tags = [...document.querySelectorAll('script[data-src]')];
  const el = document.getElementById('loading');
  let done = 0;
  const b64 = buf => new Promise(res => { const r = new FileReader(); r.onload = () => res(String(r.result).split(',')[1] || ''); r.readAsDataURL(new Blob([buf])); });
  Promise.all(tags.map(t => fetch(t.dataset.src).then(r => { if (!r.ok) throw new Error(t.dataset.src); return r.arrayBuffer(); })
    .then(b64).then(s => { t.textContent = s; done++; if (el) el.textContent = `Reconstituting subject… ${Math.round(done / tags.length * 100)}%`; })))
    .then(() => {
      const src = document.getElementById('gameModule').textContent;
      const m = document.createElement('script'); m.type = 'module'; m.textContent = src; document.body.appendChild(m);
    })
    .catch(e => { if (el) el.textContent = 'Could not load ' + e.message; });
})();
</script>
'''


def main(src, out='.'):
    s = open(src, encoding='utf-8').read()
    os.makedirs(os.path.join(out, 'assets'), exist_ok=True)
    total = 0

    def ext(id_):
        return '.glb' if id_.startswith('glb') else '.mp3'

    def take(m):
        nonlocal total
        id_, data = m.group(1), m.group(2).strip()
        raw = base64.b64decode(data)
        name = 'assets/' + id_ + ext(id_)
        open(os.path.join(out, name), 'wb').write(raw); total += len(raw)
        return f'<script type="application/octet-stream" id="{id_}" data-src="{name}"></script>'

    s = re.sub(r'<script type="application/octet-stream" id="([^"]+)">(.*?)</script>', take, s, flags=re.S)
    # hold the 3D module back until the assets are in
    i = s.index('<script type="module">')
    j = s.index('</script>', i)
    body = s[i + len('<script type="module">'):j]
    s = s[:i] + '<script type="text/plain" id="gameModule">' + body + '</script>\n' + LOADER + s[j + len('</script>'):]
    open(os.path.join(out, 'index.html'), 'w', encoding='utf-8').write(HEAD + s)
    print(f'index.html {os.path.getsize(os.path.join(out, "index.html")) // 1024} KB, assets {total // 1024} KB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '.')
