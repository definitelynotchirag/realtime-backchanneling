import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice operations console · Blue Machines",
  description: "A measured LiveKit voice operations console for real room and event activity.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
