import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { SignOutButton } from "@/components/SignOutButton";
import { ThemeToggle } from "@/components/ThemeToggle";
import { YourData } from "@/components/YourData";
import { SESSION_COOKIE, currentUser } from "@/lib/server";

/** Settings (canvas board 10, sign-up plan A4): the account, today's and this month's answers, the
 *  theme, and deleting all chats or the account, each confirmed by an emailed code. */
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
      {user?.usage ? (
        <section aria-labelledby="settings-usage">
          <h2 id="settings-usage">Usage</h2>
          <div>
            <p>
              {user.usage.today} of {user.usage.limit} answers used today, {user.usage.month} this month.
              <span className="settings-hint"> The daily count starts again at 00:00 UTC.</span>
            </p>
          </div>
        </section>
      ) : null}
      <section aria-labelledby="settings-theme">
        <h2 id="settings-theme">Theme</h2>
        <div><ThemeToggle /></div>
      </section>
      <section aria-labelledby="settings-data">
        <h2 id="settings-data">Your data</h2>
        <YourData />
      </section>
    </main>
  );
}
