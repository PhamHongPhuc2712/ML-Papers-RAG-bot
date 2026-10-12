import { ChatView } from "@/components/ChatView";

type Props = { searchParams: Promise<{ about?: string }> };

export default async function NewChat({ searchParams }: Props) {
  const { about } = await searchParams;
  return <ChatView chatId={null} about={about ?? null} />;
}
