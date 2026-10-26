# Deploying Beyond Vision

Two ways to expose the app. Pick one:

- **A. Direct IP (no domain)** - fastest, `http://<SERVER_IP>:8000`.
- **B. Domain + HTTPS** - nicer and secure, requires DNS + ports 80/443.

In both cases you ship the project together with the prebuilt `data/` directory
(the gallery, FAISS index and SQLite DB), so the server does **not** rebuild.

---

## What to copy

| Path | Size | Needed |
| --- | --- | --- |
| project source | small | always |
| `data/` | ~570 MB | always (avoids a 1-2 h rebuild) |
| `.cache/open_clip/` | ~600 MB | optional (skip the model download) |

Stop the server on your machine first so SQLite is flushed:

```powershell
Stop-Process -Id (Get-Content data\serve.pid) -Force
```

Copy to the server (from Windows):

```powershell
scp -r <path\to\project> <user>@<SERVER_IP>:~/beyond-vision
```

Do **not** copy `data/.cache/coco_hf_raw` or `data/.cache/coco_hf_tars` - they are
disposable and only slow the transfer down.

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
cd ~/beyond-vision
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
cd ~/beyond-vision
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
cd ~/beyond-vision
# bring the new source over (git pull / scp)
docker compose up -d --build                 # or -f docker-compose.prod.yml
```

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
cd ~/beyond-vision
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
