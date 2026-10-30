import { cookies } from "next/headers";
import { LoginForm } from "@/components/LoginForm";
import { Mark } from "@/components/Icon";
import { isLoginError } from "@/lib/login-messages";
import { safeNext } from "@/lib/next-path";
import { LOGIN_EMAIL_COOKIE } from "@/lib/server";

type Props = { searchParams: Promise<{ next?: string; expired?: string; error?: string; step?: string }> };

/** Log in or sign up by emailed code (canvas boards 2–3, sign-up plan A2). */
export default async function LoginPage({ searchParams }: Props) {
  const { next, expired, error, step } = await searchParams;
  const target = safeNext(next);
  const asking = target.startsWith("/?q=");
  // The code step needs the email the code went to, kept in a cookie by /api/session/code.
  const sentTo = step === "code" ? (await cookies()).get(LOGIN_EMAIL_COOKIE)?.value : undefined;
  return (
    <main className="auth">
      <p className="hero-name"><Mark size={32} /> ML Research Copilot</p>
      <h1>Log in or sign up</h1>
      {expired === "1" ? (
        <p className="auth-note" role="status">
          Your session ended. Log in again and you return to {target.startsWith("/chat/") ? "the same chat" : "where you were"}.
        </p>
      ) : asking ? (
        <p className="auth-note">Your question is kept: after you log in it is in the box, ready to send.</p>
      ) : null}
      <LoginForm
        next={target}
        initialStep={sentTo ? "code" : "email"}
        initialEmail={sentTo ?? ""}
        initialError={isLoginError(error) ? error : null}
      />
    </main>
  );
}
