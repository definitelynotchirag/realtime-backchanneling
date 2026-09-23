export async function GET(request: Request) {
  try {
    const origin = process.env.BASELINE_API_URL || "http://127.0.0.1:8000";
    const target = new URL("/events", origin);
    target.search = new URL(request.url).search || "?limit=5000";
    if (!target.searchParams.has("limit")) target.searchParams.set("limit", "5000");
    const response = await fetch(target, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error("Lifecycle API unavailable");
    const { events } = await response.json();
    if (!Array.isArray(events)) throw new Error("Invalid event response");
    return Response.json({ events }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return Response.json(
      { error: "The event API is offline. Start the Python API on port 8000, then refresh." },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
