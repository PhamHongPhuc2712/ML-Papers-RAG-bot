import { LoginForm } from "@/components/LoginForm";
import { Mark } from "@/components/Icon";
import { safeNext } from "@/lib/next-path";

type Props = { searchParams: Promise<{ next?: string; expired?: string; error?: string }> };

/** Log in (canvas board 2's layout; email and password until the sign-up plan brings emailed codes). */
export default async function LoginPage({ searchParams }: Props) {
  const { next, expired, error } = await searchParams;
  const target = safeNext(next);
  const asking = target.startsWith("/?q=");
  return (
    <main className="auth">
      <p className="hero-name"><Mark size={32} /> ML Research Copilot</p>
      <h1>Log in</h1>
      {expired === "1" ? (
        <p className="auth-note" role="status">
          Your session ended. Log in again and you return to {target.startsWith("/chat/") ? "the same chat" : "where you were"}.
        </p>
      ) : asking ? (
        <p className="auth-note">Your question is kept: after you log in it is in the box, ready to send.</p>
      ) : null}
      <LoginForm next={target} initialError={error === "credentials" || error === "unreachable" ? error : null} />
    </main>
  );
}
