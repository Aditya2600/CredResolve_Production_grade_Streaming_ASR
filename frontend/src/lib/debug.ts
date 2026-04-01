const DEBUG_LOGS_ENABLED =
  import.meta.env.DEV || String(import.meta.env.VITE_DEBUG_LOGS || '').toLowerCase() === 'true';

function formatMessage(scope: string, message: string): string {
  return `[frontend:${scope}] ${message}`;
}

export function debugLog(scope: string, message: string, ...args: unknown[]): void {
  if (!DEBUG_LOGS_ENABLED) {
    return;
  }
  console.debug(formatMessage(scope, message), ...args);
}

export function infoLog(scope: string, message: string, ...args: unknown[]): void {
  if (!DEBUG_LOGS_ENABLED) {
    return;
  }
  console.info(formatMessage(scope, message), ...args);
}

export function warnLog(scope: string, message: string, ...args: unknown[]): void {
  console.warn(formatMessage(scope, message), ...args);
}

export function errorLog(scope: string, message: string, ...args: unknown[]): void {
  console.error(formatMessage(scope, message), ...args);
}

export { DEBUG_LOGS_ENABLED };
