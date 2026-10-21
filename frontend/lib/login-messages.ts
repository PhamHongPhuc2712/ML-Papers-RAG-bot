/** What the sign-in page says for each failure (sign-up plan, A2). A plain module, so the server
 *  page can check an `?error=` against it: a value imported from a client file is only a reference. */
export const LOGIN_MESSAGES = {
  email: "Enter a valid email address.",
  unreachable: "The service is not reachable. Try again in a moment.",
  mail_unavailable: "The code could not be sent. Try again in a moment.",
  invalid_code: "That code is not right. Check it and try again.",
  code_exhausted: "Too many wrong tries. Ask for a new code.",
  code_expired: "That code has expired or was already used. Ask for a new code.",
} as const;
export type LoginError = keyof typeof LOGIN_MESSAGES;

export function isLoginError(value: string | undefined): value is LoginError {
  return value !== undefined && Object.hasOwn(LOGIN_MESSAGES, value);
}
