import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { DeleteAccount } from "@/components/DeleteAccount";
import { SignOutButton } from "@/components/SignOutButton";
import { ThemeToggle } from "@/components/ThemeToggle";
import { SESSION_COOKIE, currentUser } from "@/lib/server";

/** Settings (canvas board 10, the parts that exist today): the account, the theme and deleting the
 *  account. Usage, a password change and "delete all chats" come with the sign-up plan. */
export default async function SettingsPage() {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) redirect("/login?next=%2Faccount");
  const user = await currentUser();
  return (
    <main className="page settings">
      <h1>Settings</h1>
      <section aria-labelledby="settings-account">
        <h2 id="settings-account">Account</h2>
        <div>
          <p>{user?.email ?? "Your account"}</p>
          <SignOutButton className="text-button link">Log out</SignOutButton>
        </div>
      </section>
      <section aria-labelledby="settings-theme">
        <h2 id="settings-theme">Theme</h2>
        <div><ThemeToggle /></div>
      </section>
      <section aria-labelledby="settings-data">
        <h2 id="settings-data">Your data</h2>
        <DeleteAccount />
      </section>
    </main>
  );
}
