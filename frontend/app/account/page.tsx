import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { DeleteAccount } from "@/components/DeleteAccount";
import { SESSION_COOKIE } from "@/lib/server";

export default async function AccountPage() {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) redirect("/login?next=%2Faccount");
  return (
    <main style={{ maxWidth: 560 }}>
      <h1>Account</h1>
      <h2>Delete your account</h2>
      <p>Deleting your account removes all of your chats at once. This cannot be undone.</p>
      <DeleteAccount />
    </main>
  );
}
