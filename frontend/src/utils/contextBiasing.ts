import type {
  BiasingContextPayload,
  ContextBiasingMode,
  SessionConfigMessage,
} from '../types/ws';

export interface BiasingFormValues {
  debtorName: string;
  agentName: string;
  lender: string;
  product: string;
  city: string;
  branch: string;
  accountTerms: string;
  priorCallEntities: string;
  campaignVocabulary: string;
  amounts: string;
  dates: string;
}

export interface DemoBiasingSettings {
  enabled: boolean;
  mode: Extract<ContextBiasingMode, 'shadow' | 'active'>;
  values: BiasingFormValues;
  audioProcessing?: {
    vad_enabled: boolean;
    denoise_enabled: boolean;
  };
}

export const DEFAULT_BIASING_FORM_VALUES: BiasingFormValues = {
  debtorName: '',
  agentName: '',
  lender: '',
  product: '',
  city: '',
  branch: '',
  accountTerms: '',
  priorCallEntities: '',
  campaignVocabulary: '',
  amounts: '',
  dates: '',
};

function normalizeListField(value: string): string[] {
  const items: string[] = [];
  let current = '';

  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (char === ',') {
      const prevChar = index > 0 ? value[index - 1] : '';
      const nextChar = index + 1 < value.length ? value[index + 1] : '';
      if (/\d/.test(prevChar) && /\d/.test(nextChar)) {
        current += char;
        continue;
      }
      const trimmed = current.trim();
      if (trimmed) {
        items.push(trimmed);
      }
      current = '';
      continue;
    }
    current += char;
  }

  const trimmed = current.trim();
  if (trimmed) {
    items.push(trimmed);
  }
  return items;
}

export function buildBiasingContextPayload(values: BiasingFormValues): BiasingContextPayload | undefined {
  const payload: BiasingContextPayload = {};

  const assignString = (key: keyof BiasingContextPayload, value: string) => {
    const normalized = value.trim();
    if (normalized) {
      payload[key] = normalized as never;
    }
  };

  const assignList = (key: keyof BiasingContextPayload, value: string) => {
    const normalized = normalizeListField(value);
    if (normalized.length > 0) {
      payload[key] = normalized as never;
    }
  };

  assignString('debtor_name', values.debtorName);
  assignString('agent_name', values.agentName);
  assignString('lender', values.lender);
  assignString('product', values.product);
  assignString('city', values.city);
  assignString('branch', values.branch);
  assignList('account_terms', values.accountTerms);
  assignList('prior_call_entities', values.priorCallEntities);
  assignList('campaign_vocabulary', values.campaignVocabulary);
  assignList('amounts', values.amounts);
  assignList('dates', values.dates);

  return Object.keys(payload).length > 0 ? payload : undefined;
}

export function buildSessionConfigMessage(settings: DemoBiasingSettings): SessionConfigMessage | null {
  const biasingContext = buildBiasingContextPayload(settings.values);
  const audioProcessing = settings.audioProcessing;

  if (!settings.enabled && !audioProcessing) {
    return null;
  }

  const message: SessionConfigMessage = {
    type: 'session_config',
  };

  if (settings.enabled) {
    message.context_biasing = {
      enabled: true,
      mode: settings.mode,
    };
    message.biasing_context = biasingContext;
  }

  if (audioProcessing) {
    message.audio_processing = audioProcessing;
  }

  return message;
}
