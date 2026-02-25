interface ReasoningTurn {
  role: 'agent' | 'borrower';
  text: string;
}

interface ReasoningRequest {
  calleeName: string;
  amountDue: string;
  borrowerText: string;
  history: ReasoningTurn[];
}

interface ReasoningResponse {
  agent_text: string;
  source: string;
}

const REASONING_URL = import.meta.env.VITE_REASONING_URL || '/v1/reasoning/respond';

export async function requestAgentReply(payload: ReasoningRequest): Promise<{ agentText: string; source: string }> {
  const response = await fetch(REASONING_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      callee_name: payload.calleeName,
      amount_due: payload.amountDue,
      borrower_text: payload.borrowerText,
      history: payload.history,
    }),
  });

  if (!response.ok) {
    throw new Error(`Reasoning request failed (${response.status})`);
  }

  const data = (await response.json()) as ReasoningResponse;
  return {
    agentText: data.agent_text,
    source: data.source,
  };
}

