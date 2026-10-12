import { LoginForm } from "@/components/LoginForm";

type Props = { searchParams: Promise<{ next?: string; expired?: string }> };

export default async function LoginPage({ searchParams }: Props) {
  const { next, expired } = await searchParams;
  // Only same-site paths: an absolute URL here would be an open redirect.
  const target = next && next.startsWith("/") && !next.startsWith("//") ? next : "/chat";
  return (
    <main>
      <h1>Sign in</h1>
      {expired === "1" ? (
        <p className="banner warn" role="status">
          Your session expired. Sign in again to continue.
        </p>
      ) : null}
      <LoginForm next={target} />
    </main>
  );
}
