import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { HistoryView } from "@/components/HistoryView";
import { SESSION_COOKIE } from "@/lib/server";

/** Every chat, searchable by title (canvas board 9). */
export default async function HistoryPage() {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) redirect("/login?next=%2Fhistory");
  return (
    <main className="page">
      <h1>History</h1>
      <HistoryView />
    </main>
  );
}
