import { LoginForm } from "@/components/LoginForm";
import { Mark } from "@/components/Icon";

type Props = { searchParams: Promise<{ next?: string; expired?: string }> };

/** Log in (canvas board 2's layout; email and password until the sign-up plan brings emailed codes). */
export default async function LoginPage({ searchParams }: Props) {
  const { next, expired } = await searchParams;
  // Only same-site paths: an absolute URL here would be an open redirect.
  const target = next && next.startsWith("/") && !next.startsWith("//") ? next : "/";
  const asking = target.startsWith("/?q=");
  return (
    <main className="auth">
      <p className="hero-name"><Mark size={32} /> ML Research Copilot</p>
      <h1>Log in</h1>
      {expired === "1" ? (
        <p className="banner warn" role="status">Your session expired. Log in again to continue.</p>
      ) : asking ? (
        <p className="auth-note">Your question is kept: after you log in it is in the box, ready to send.</p>
      ) : null}
      <LoginForm next={target} />
    </main>
  );
}
