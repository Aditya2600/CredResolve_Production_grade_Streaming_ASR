export function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function formatErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function getVisibleTranscript({
  display_text,
  canonical_text,
  transcript,
}: {
  display_text?: string;
  canonical_text?: string;
  transcript: string;
}): string {
  return display_text || canonical_text || transcript;
}
