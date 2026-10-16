import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { Workspace } from "@/components/Workspace";
import { SESSION_COOKIE } from "@/lib/server";

export default async function ChatLayout({ children }: { children: ReactNode }) {
  if (!(await cookies()).get(SESSION_COOKIE)?.value) redirect("/login?next=%2Fchat");
  return <Workspace>{children}</Workspace>;
}
