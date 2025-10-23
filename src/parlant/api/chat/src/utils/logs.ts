import { Log } from '@parlant/database';
import { LogType } from '@parlant/interfaces';
import { getChatSettings } from './chat';
import { getSensitiveWords } from './settings';

/**
 * Logs a message to the database.
 * @param chatId The ID of the chat to log the message in.
 * @param userId The ID of the user who sent the message.
 * @param message The message to log.
 * @param type The type of log.
 * @returns The created log.
 */
export const logMessage = async (
  chatId: string,
  userId: string,
  message: string,
  type: LogType
) => {
  const { sensitiveWords, logsEnabled } = await getChatSettings(chatId);

  if (!logsEnabled) {
    return;
  }

  const censoredMessage = censorMessage(message, sensitiveWords);

  const log = await Log.create({
    chatId,
    userId,
    message: censoredMessage,
    type,
  });

  return log;
};

/**
 * Logs a message to the database without fetching chat settings.
 * @param chatId The ID of the chat to log the message in.
 * @param userId The ID of the user who sent the message.
 * @param message The message to log.
 * @param type The type of log.
 * @returns The created log.
 */
export const logMessageWithoutSettings = async (
  chatId: string,
  userId: string,
  message: string,
  type: LogType
) => {
  const sensitiveWords = await getSensitiveWords();
  const censoredMessage = censorMessage(message, sensitiveWords);

  const log = await Log.create({
    chatId,
    userId,
    message: censoredMessage,
    type,
  });

  return log;
};

/**
 * Logs a message to the database from the system.
 * @param chatId The ID of the chat to log the message in.
 * @param message The message to log.
 * @returns The created log.
 */
export const logSystemMessage = async (chatId: string, message: string) => {
  const { sensitiveWords, logsEnabled } = await getChatSettings(chatId);

  if (!logsEnabled) {
    return;
  }

  const censoredMessage = censorMessage(message, sensitiveWords);

  const log = await Log.create({
    chatId,
    message: censoredMessage,
    type: LogType.SYSTEM,
  });

  return log;
};

/**
 * Logs a message to the database from the system without fetching chat settings.
 * @param chatId The ID of the chat to log the message in.
 * @param message The message to log.
 * @returns The created log.
 */
export const logSystemMessageWithoutSettings = async (
  chatId: string,
  message: string
) => {
  const sensitiveWords = await getSensitiveWords();
  const censoredMessage = censorMessage(message, sensitiveWords);

  const log = await Log.create({
    chatId,
    message: censoredMessage,
    type: LogType.SYSTEM,
  });

  return log;
};

/**
 * Logs a message to the database from the bot.
 * @param chatId The ID of the chat to log the message in.
 * @param message The message to log.
 * @returns The created log.
 */
export const logBotMessage = async (chatId: string, message: string) => {
  const { sensitiveWords, logsEnabled } = await getChatSettings(chatId);

  if (!logsEnabled) {
    return;
  }

  const censoredMessage = censorMessage(message, sensitiveWords);

  const log = await Log.create({
    chatId,
    message: censoredMessage,
    type: LogType.BOT,
  });

  return log;
};

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * Censors a message by replacing sensitive words with asterisks.
 * @param message The message to censor.
 * @param sensitiveWords The list of sensitive words to censor.
 * @returns The censored message.
 */
export const censorMessage = (
  message: string,
  sensitiveWords: string[]
): string => {
  if (!sensitiveWords || sensitiveWords.length === 0) {
    return message;
  }

  const pattern = sensitiveWords
    .filter((word) => word.length > 0)
    .map(escapeRegExp)
    .join('|');

  if (pattern === '') {
    return message;
  }

  const regex = new RegExp(pattern, 'gi');

  return message.replace(regex, (match) => '*'.repeat(match.length));
};
