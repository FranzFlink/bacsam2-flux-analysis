# Deploying the website (password protected) on Cloudflare

The site (`pixi run site` -> `_build/html`) is served as static assets of a Cloudflare Worker.
The Worker in `src/index.js` runs in front of every request and asks for HTTP Basic
authentication; user name and password are Worker secrets, never stored in the repository.

Node and npm come from the pixi `docs` environment, wrangler is installed locally in this
folder (`node_modules/`, ignored by git).

## One-time setup

```
pixi run -e docs npm install --prefix deploy      # installs wrangler
pixi run site-login                               # opens the browser: log in to Cloudflare
```

Set the hostname in `wrangler.jsonc` (`routes[0].pattern`); the zone must be in the same
Cloudflare account. Then set the credentials (prompted interactively, stored at Cloudflare):

```
cd deploy
../.pixi/envs/docs/node.exe node_modules/wrangler/bin/wrangler.js secret put SITE_USER
../.pixi/envs/docs/node.exe node_modules/wrangler/bin/wrangler.js secret put SITE_PASSWORD
```

## Every deployment

```
pixi run site          # rebuild _build/html
pixi run site-deploy   # upload to Cloudflare
```

To change the password, run `secret put SITE_PASSWORD` again (takes effect immediately, no
redeploy needed). To remove the site: `wrangler delete`.

## Troubleshooting

* `403` with body `error code: 1050` on every URL, nothing in `wrangler tail`: Cloudflare Access is
  rejecting the request before the Worker runs, because the account-wide Zero Trust setting
  "Require Cloudflare Access protection for all resources" (default deny) is enabled and no Access
  application allows the hostname. Either switch that setting off (Zero Trust dashboard > Settings >
  Authentication) so that the Basic-auth Worker is the only gate, or add a self-hosted Access
  application for the hostname with a policy of action "Bypass" for everyone.
