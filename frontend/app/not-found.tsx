import Link from "next/link";

export default function NotFound() {
  return (
    <main>
      <h1>Not found</h1>
      <p>
        That page does not exist. <Link href="/">Search papers</Link>.
      </p>
    </main>
  );
}
