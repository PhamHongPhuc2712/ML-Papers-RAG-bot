import Link from "next/link";
import { cookies } from "next/headers";
import { SESSION_COOKIE } from "@/lib/server";
import { SignOutButton } from "./SignOutButton";

export async function Header() {
  const signedIn = Boolean((await cookies()).get(SESSION_COOKIE)?.value);
  return (
    <header className="site">
      <nav aria-label="Main">
        <Link href="/">Search</Link>
        <Link href="/chat">Chat</Link>
      </nav>
      {signedIn ? <SignOutButton /> : <Link href="/login">Sign in</Link>}
    </header>
  );
}
