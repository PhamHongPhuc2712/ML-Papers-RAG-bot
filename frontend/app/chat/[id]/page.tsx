import { ChatView } from "@/components/ChatView";

type Props = { params: Promise<{ id: string }> };

export default async function ExistingChat({ params }: Props) {
  const { id } = await params;
  return <ChatView key={id} chatId={id} about={null} />;
}
