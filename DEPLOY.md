# Deploying Text-to-Image Retrieval (COMP5523)

Two ways to expose the app. Pick one:

- **A. Direct IP (no domain)** - fastest, `http://<SERVER_IP>:8000`.
- **B. Domain + HTTPS** - nicer and secure, requires DNS + ports 80/443.

In both cases you need the project source and the prebuilt `data/` directory
(50k gallery, FAISS index and SQLite DB). The data is published as a GitHub Release
asset, so the server does **not** rebuild anything.

---

## 1. Get the project and the data

```bash
git clone https://github.com/yuchokwong/Poly-COMP.git
cd Poly-COMP

# Download and extract data/ (~850 MB archive, ~800 MB extracted)
curl -L -o data_upload.tar \
  https://github.com/yuchokwong/Poly-COMP/releases/latest/download/data_upload.tar
tar -xf data_upload.tar -C .
```

This creates `./data` with:

| Path | Size | Notes |
| --- | --- | --- |
| `data/images/` | ~630 MB | 50,000 COCO images (45k train / 5k val) |
| `data/gallery.db` | ~43 MB | SQLite metadata, captions and FTS5 index |
| `data/index.faiss` + `data/index_ids.npy` | ~98 MB | 50k CLIP embeddings |
| `data/manifest.jsonl` | ~26 MB | image metadata |
| `data/eval_queries.json` | small | 200 evaluation queries |

Verify the key artifacts are present:

```bash
ls -lh data/index.faiss data/gallery.db data/images
```

The first search downloads the OpenCLIP weights (~577 MB) automatically. You do
**not** need to ship `.cache/open_clip`; to skip the download, pre-populate the
`openclip_cache` Docker volume yourself.

### Alternative - copy an existing local `data/`

If you already built the data on your machine, stop the server first (so SQLite is
flushed), then copy the project:

```powershell
Stop-Process -Id (Get-Content data\serve.pid) -Force
scp -r <path\to\project> <user>@<SERVER_IP>:~/Poly-COMP
```

Do **not** copy `data/.cache/coco_hf_raw` or `data/.cache/coco_hf_tars` - they are
disposable and only slow the transfer down.

### Alternative - rebuild on the server

Instead of downloading the archive, build the same 50k gallery on the server:

```bash
docker compose --profile tools run --rm build python -m tsr build --max-images 50000
```

This downloads COCO and recomputes the embeddings (network + ~1.5 GB, much slower).

---

## A. Direct IP (no domain)

### On the Ubuntu server

```bash
# 1. Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# 2. Open the app port (also open it in your cloud security group if any)
sudo ufw allow 8000/tcp

# 3. Start (base compose publishes port 8000)
cd ~/Poly-COMP
docker compose up -d --build

# 4. Find your public IP
curl -s ifconfig.me; echo
```

Others then open:

```
http://<SERVER_IP>:8000
```

Verify from anywhere:

```bash
curl http://<SERVER_IP>:8000/api/health   # {"status":"ok","ready":true}
```

**Optional - serve on port 80** (so the URL has no `:8000`): add `TSR_APP__PORT=80`
to `.env`, `sudo ufw allow 80/tcp`, then `docker compose up -d`.

> Note: plain IP means **no HTTPS** (Let's Encrypt only issues certs for domains).

---

## B. Domain + HTTPS

1. Create an **A record** for e.g. `search.example.com` pointing at the server's
   public IP. Verify with `dig +short search.example.com`.
2. Open ports `80` and `443` (`sudo ufw allow 80/tcp; sudo ufw allow 443/tcp` and in
   the cloud security group).
3. Configure and start:

```bash
cd ~/Poly-COMP
cp .env.example .env
# edit .env -> DOMAIN=search.example.com  and  ACME_EMAIL=you@example.com
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml logs -f caddy   # watch certificate issuance
```

Caddy obtains and renews the TLS certificate automatically. The app itself is only
exposed on the internal network (`expose: 8000`), never directly.

> No domain but still want HTTPS? Point `DOMAIN` at `<SERVER_IP>.sslip.io`, which
> resolves to that IP.

---

## Updating the app

```bash
cd ~/Poly-COMP
git pull                                   # data/ is gitignored and stays in place
docker compose up -d --build               # or -f docker-compose.prod.yml
```

Re-download `data_upload.tar` only when a release explicitly updates the gallery or
index; the old `data/` is otherwise reused.

## Operations

```bash
docker compose ps
docker compose logs -f app
docker compose restart app
docker compose down            # stop (data/ on the host is untouched)
```

## Native install (no Docker)

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
cd ~/Poly-COMP
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
pip install -r requirements.txt && pip install -e .
python -m tsr serve --host 0.0.0.0 --port 8000
```

Behind nginx + certbot for HTTPS:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
# proxy_pass http://127.0.0.1:8000;  then:
sudo certbot --nginx -d search.example.com
```

## Troubleshooting

- **Cannot connect**: almost always the port is closed - check `ufw` **and** the cloud
  security group. Test locally first with `curl http://localhost:8000/api/health`.
- **`ready: false`**: `data/` is missing `index.faiss` / `gallery.db`, or the model
  was changed without rebuilding. Check `docker compose logs app`.
- **Dynamic public IP**: an IP-based URL breaks when the address changes; use a static
  IP / Elastic IP or a domain.
- **Weak CPU**: use `TSR_CONFIG=configs/lite.yaml` and optionally
  `TSR_MODEL__BACKEND=onnx` after running `python -m tsr export-onnx`.
