import Link from "next/link";
import { cookies } from "next/headers";
import { SESSION_COOKIE } from "@/lib/server";
import { SignOutButton } from "./SignOutButton";

export async function Header() {
  const signedIn = Boolean((await cookies()).get(SESSION_COOKIE)?.value);
  return (
    <header className="site">
      <nav aria-label="Main">
        <Link href="/">ML Research Copilot</Link>
      </nav>
      {signedIn ? (
        <span className="account">
          <Link href="/account">Account</Link>
          <SignOutButton />
        </span>
      ) : (
        <Link href="/login">Sign in</Link>
      )}
    </header>
  );
}
