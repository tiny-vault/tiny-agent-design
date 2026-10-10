#!/bin/sh
# Start nginx in front of OpenObserve so the web UI opens already logged in.
#
# The OpenObserve UI decides "am I logged in?" in the browser: it needs an `auth_tokens`
# cookie (base64 of {"access_token": "Basic <user:pass>", "refresh_token": ""}) and two
# localStorage entries (`userInfo`, `currentuser`) describing the user. This script builds
# all three from the Docker secrets, so no login is ever typed. Everything is derived at
# start-up; nothing is stored in the image or in git.
#
# Trade-off: anyone who can open http://localhost:5080 on this machine is logged in as the
# OpenObserve admin. The port is published on 127.0.0.1 only, never to the network.
set -eu

USER_NAME=$(cat /run/secrets/openobserve_user)
PASS=$(cat /run/secrets/openobserve_password)

BASIC=$(printf '%s:%s' "$USER_NAME" "$PASS" | base64 | tr -d '\n')
OO_AUTH_COOKIE=$(printf '{"access_token":"Basic %s","refresh_token":""}' "$BASIC" | base64 | tr -d '\n')

NOW=$(date +%s)
EXP=$((NOW + 2592000))   # 30 days, the same lifetime as the cookie below
CLAIMS=$(printf '{"given_name":"%s","auth_time":%s,"name":"%s","exp":%s,"family_name":"","email":"%s"}' \
  "$USER_NAME" "$NOW" "$USER_NAME" "$EXP" "$USER_NAME")
USER_INFO=$(printf '%s' "$CLAIMS" | base64 | tr -d '\n')

# Read by the nginx config template (/etc/nginx/templates), filled in by nginx's own entrypoint.
export OO_AUTH_COOKIE

# The page that sets the browser-side login state, then opens the UI. It holds no password.
cat > /usr/share/nginx/html/auto-login.html <<HTML
<!doctype html>
<meta charset="utf-8">
<title>Signing in to OpenObserve...</title>
<script>
  localStorage.setItem("userInfo", "$USER_INFO");
  localStorage.setItem("currentuser", JSON.stringify($CLAIMS));
  location.replace("/web/");
</script>
<noscript>JavaScript is needed to sign in automatically. <a href="/web/">Open OpenObserve</a>.</noscript>
HTML

exec /docker-entrypoint.sh nginx -g "daemon off;"
