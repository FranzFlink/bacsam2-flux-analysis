// Password gate in front of the static site (_build/html) on Cloudflare Workers.
//
// Every request runs through this Worker first (assets.run_worker_first in wrangler.jsonc).
// It checks HTTP Basic authentication against the secrets SITE_USER and SITE_PASSWORD
// (set with `wrangler secret put`, never written to this repository) and only then hands
// the request to the static assets.

const REALM = "BACSAM-II supplement";

function unauthorized() {
  return new Response("Authentication required.", {
    status: 401,
    headers: {
      "WWW-Authenticate": `Basic realm="${REALM}", charset="UTF-8"`,
      "Cache-Control": "no-store",
    },
  });
}

// Constant-time comparison of two strings (avoids leaking the password length or prefix).
function timingSafeEqual(a, b) {
  const enc = new TextEncoder();
  const ab = enc.encode(a);
  const bb = enc.encode(b);
  if (ab.byteLength !== bb.byteLength) {
    // Compare something of equal length anyway so the timing does not depend on the length.
    crypto.subtle.timingSafeEqual(ab, ab);
    return false;
  }
  return crypto.subtle.timingSafeEqual(ab, bb);
}

function credentialsOk(request, env) {
  const header = request.headers.get("Authorization") || "";
  if (!header.startsWith("Basic ")) return false;
  let decoded;
  try {
    decoded = atob(header.slice(6));
  } catch {
    return false;
  }
  const idx = decoded.indexOf(":");
  if (idx < 0) return false;
  const user = decoded.slice(0, idx);
  const pass = decoded.slice(idx + 1);
  const userOk = timingSafeEqual(user, env.SITE_USER || "");
  const passOk = timingSafeEqual(pass, env.SITE_PASSWORD || "");
  return userOk && passOk;
}

export default {
  async fetch(request, env) {
    if (!env.SITE_USER || !env.SITE_PASSWORD) {
      return new Response(
        "Site is locked: SITE_USER / SITE_PASSWORD secrets are not set (wrangler secret put).",
        { status: 503 },
      );
    }
    if (!credentialsOk(request, env)) return unauthorized();

    const response = await env.ASSETS.fetch(request);
    // Keep protected pages out of shared caches.
    const headers = new Headers(response.headers);
    headers.set("Cache-Control", "private, no-store");
    headers.set("X-Robots-Tag", "noindex, nofollow");
    return new Response(response.body, { status: response.status, headers });
  },
};
