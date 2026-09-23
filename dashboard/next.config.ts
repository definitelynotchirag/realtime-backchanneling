import type { NextConfig } from "next";

/**
 * Dev-server config. The only entry here exists so the console can be driven from a
 * phone: `http://<lan-ip>:3001` is not a secure context, so browsers refuse the
 * microphone there, and the workable route is a quick tunnel over HTTPS
 * (`cloudflared tunnel --url http://localhost:3001`). Next 16 blocks
 * dev-only `/_next/*` requests whose Origin is not on this list, and a quick tunnel
 * gets a random `*.trycloudflare.com` hostname each time.
 */
const nextConfig: NextConfig = {
  allowedDevOrigins: ["*.trycloudflare.com"],
};

export default nextConfig;
