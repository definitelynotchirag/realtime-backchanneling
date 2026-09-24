export async function GET(
  _request: Request,
  context: { params: Promise<{ run_id: string }> },
) {
  try {
    const { run_id } = await context.params;
    const origin = process.env.BASELINE_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(`${origin}/runs/${encodeURIComponent(run_id)}/summary`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    const payload = await response.json().catch(() => ({}));
    return Response.json(payload, {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json(
      { detail: "The run summary API is offline. Start the Python API on port 8000, then refresh." },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
