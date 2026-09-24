import type { NextConfig } from "next";

/**
 * Dev-server config.
 *
 * Next 16 blocks dev-only `/_next/*` requests whose Origin is not on this list, so the
 * list has to name every origin the console is reached by. Two reasons it matters here:
 *
 * - The dashboard is served on all interfaces (`next dev --hostname 0.0.0.0`) so a phone
 *   on the LAN can open it. The hostname Next would otherwise trust is then `0.0.0.0`,
 *   not the address in the browser, so `127.0.0.1` and `localhost` count as foreign and
 *   the page renders server-side without ever hydrating - which shows up as a console
 *   with empty panels, no report, and an empty run trace, rather than as an error.
 * - A phone is better served over HTTPS (`cloudflared tunnel --url http://localhost:3001`)
 *   because a LAN address is not a secure context and browsers refuse the microphone
 *   there. A quick tunnel gets a random `*.trycloudflare.com` hostname each time.
 */
const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost", "*.trycloudflare.com", "10.41.0.2"],
};

export default nextConfig;
