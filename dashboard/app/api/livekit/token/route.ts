import { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const origin = process.env.BASELINE_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(new URL("/livekit/token", origin), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await request.json()),
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    const payload = await response.json();
    return Response.json(payload, {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json(
      { error: "The token API is offline. Start the Python API on port 8000." },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
