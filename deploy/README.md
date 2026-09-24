# Deploying the demo stack (single EC2 host)

Everything runs on one Amazon Linux 2023 instance as `ec2-user`, from
`/home/ec2-user/realtime-voice-chat` (a clone of this repository).

## Components

| Unit | What it runs | Listens on |
|---|---|---|
| `blue-machines-worker` | `uv run blue-machines-agent start` (LiveKit worker) | outbound only |
| `blue-machines-api` | `uv run uvicorn blue_machines_baseline.api:app` | `127.0.0.1:8000` |
| `blue-machines-dashboard` | `npm start` (Next.js production build) | `127.0.0.1:3000` |
| `caddy` | TLS + reverse proxy to the dashboard | public `:80`, `:443` |

The browser only ever talks to Caddy; the API stays on localhost and is
reached by the dashboard's server-side route handlers via `BASELINE_API_URL`.

## Public entry point

- **https://54-176-92-17.sslip.io** — `sslip.io` resolves to this instance's
  public IP, so it is eligible for a real Let's Encrypt certificate.
- `http://ec2-54-176-92-17.us-west-1.compute.amazonaws.com` — permanent
  redirect to the address above. The AWS default hostname itself can never
  get a public certificate: Let's Encrypt refuses `*.compute.amazonaws.com`
  by policy, so it cannot be used for HTTPS directly.

HTTPS matters beyond polish: browsers only expose the microphone
(`getUserMedia`) in a secure context, which the live conversation demo needs.

Prerequisite: the instance security group must allow inbound **80** and
**443** from the internet. If certificate issuance is still pending after
opening the ports, run `sudo systemctl reload caddy`.

## Install / update

```bash
# one-time, as ec2-user
sudo dnf install -y git nodejs20
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sL "https://caddyserver.com/api/download?os=linux&arch=amd64" -o /tmp/caddy
sudo install -m 0755 /tmp/caddy /usr/local/bin/caddy

git clone https://github.com/definitelynotchirag/realtime-backchanneling.git \
  ~/realtime-voice-chat
cp /path/to/.env ~/realtime-voice-chat/.env   # secrets, never committed
chmod 600 ~/realtime-voice-chat/.env

cd ~/realtime-voice-chat
uv sync --extra dev
(cd dashboard && npm ci && npm run build)

sudo useradd -r -s /sbin/nologin caddy 2>/dev/null || true
sudo mkdir -p /etc/caddy /var/lib/caddy && sudo chown caddy:caddy /var/lib/caddy
sudo install -m 644 deploy/Caddyfile /etc/caddy/Caddyfile
sudo install -m 644 deploy/blue-machines-*.service deploy/caddy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now caddy blue-machines-api blue-machines-dashboard blue-machines-worker
```

Update after new commits:

```bash
cd ~/realtime-voice-chat
git pull
uv sync --extra dev
(cd dashboard && npm ci && npm run build)
sudo systemctl restart blue-machines-api blue-machines-dashboard blue-machines-worker
```

## Continuous deployment

Pushes to `master` deploy automatically (`.github/workflows/deploy.yml`):

1. a **self-hosted GitHub runner** on this host (systemd unit
   `actions.runner.definitelynotchirag-realtime-backchanneling.vps-ec2.service`)
   picks up the push — it polls GitHub over outbound HTTPS only, so no
   inbound ports or SSH keys in GitHub are involved;
2. `git reset --hard origin/master` in `~/realtime-voice-chat`, `uv sync`;
3. `uv run pytest` — a failing suite stops the deploy;
4. `npm ci && npm run build` in `dashboard/`;
5. `deploy/Caddyfile` and the systemd units are re-installed and validated;
6. the worker, API, and dashboard restart, then health checks run.

A full deploy takes about 30 seconds. Only pushes to `master` and manual
`workflow_dispatch` trigger it (never fork pull requests, so untrusted code
cannot run on the host).

Useful commands:

```bash
gh run list --limit 5                # recent deploys (from any machine)
gh run watch <run-id>                # follow one live
sudo journalctl -u actions.runner.definitelynotchirag-realtime-backchanneling.vps-ec2 -f
```

Still manual by design: secret changes (`.env` is not in the repo), runner
software updates, and security-group changes.

## LiveKit project fallbacks

`.env` may list extra LiveKit projects as `LIVEKIT_URL_2` /
`LIVEKIT_API_KEY_2` / `LIVEKIT_API_SECRET_2` (and `_3`, ...). With more than one
project configured, the first one that answers a probe call is used, and the
choice is cached in `.runtime/livekit-active.json` for 45 seconds so the worker
and the API agree on it. An expired-credit project fails its probe and the next
one takes over; a token minted by the API always targets the project the worker
is registered to. A partial fallback (a URL without its key) is a startup
error, not a silently ignored entry.

## Operations

```bash
systemctl status blue-machines-worker blue-machines-api blue-machines-dashboard caddy
sudo journalctl -u blue-machines-worker -f        # worker logs
sudo journalctl -u blue-machines-dashboard -f     # dashboard logs
curl -s http://127.0.0.1:8000/health              # API health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3000/   # dashboard
```

One worker only: a second registered worker makes room dispatch and
benchmark labels ambiguous.

## Agent name isolation (important)

The deployment registers its worker as **`blue-machines-prod`** via
`LIVEKIT_AGENT_NAME` in `.env`, and the API mints tokens that dispatch to that
same name (both read the variable). Keep it unique: LiveKit Cloud round-robins
dispatch between *all* workers registered under one agent name, so a
development worker running anywhere else under the default
`blue-machines-baseline` name would steal dispatches for public rooms — the
public demo would then depend on whichever machine happened to win, and on a
laptop that is busy, asleep, or closed, rooms would get no agent at all.

## Notes

- Runtime data lives in `outputs/` on the host (`baseline-events.jsonl` is
  git-ignored by design; `benchmark-events.jsonl` is committed).
- If `sslip.io` certificate issuance ever fails, Caddy logs the reason
  (`journalctl -u caddy`); the fallback is `tls internal` in the Caddyfile
  (self-signed, browser warning) — still a secure context for the mic.
