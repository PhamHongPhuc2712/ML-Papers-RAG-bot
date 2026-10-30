import Link from "next/link";

/** Page not found (canvas board 11). */
export default function NotFound() {
  return (
    <main className="page not-found">
      <h1>This page doesn’t exist</h1>
      <p className="muted">The link may be old, or the chat was deleted.</p>
      <Link href="/" className="button-link">Go to home</Link>
    </main>
  );
}
