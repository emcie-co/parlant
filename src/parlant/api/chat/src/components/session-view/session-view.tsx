import React, {ReactElement, useEffect, useRef, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {deleteData, postData} from '@/utils/api';
import {EventInterface, SessionInterface} from '@/utils/interfaces';
import {Spacer} from '../ui/custom/spacer';
import {toast} from 'sonner';
import {NEW_SESSION_ID} from '../chat-header/chat-header';
import {useQuestionDialog} from '@/hooks/useQuestionDialog';
import {twJoin, twMerge} from 'tailwind-merge';
import MessageLogs from '../message-logs/message-logs';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, customerAtom, newSessionAtom, sessionAtom, sessionsAtom} from '@/store';
import CopyText from '../ui/custom/copy-text';
import ErrorBoundary from '../error-boundary/error-boundary';
import Messages from './messages/messages';

const emptyPendingMessage: () => EventInterface = () => ({
	kind: 'message',
	source: 'customer',
	creation_utc: new Date(),
	serverStatus: 'pending',
	offset: 0,
	correlation_id: '',
	data: {
		message: '',
	},
});

export default function SessionView(): ReactElement {
	const submitButtonRef = useRef<HTMLButtonElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	const [message, setMessage] = useState('');
	const [pendingMessage, setPendingMessage] = useState<EventInterface>(emptyPendingMessage());
	const [lastOffset, setLastOffset] = useState(0);
	const [messages, setMessages] = useState<EventInterface[]>([]);
	const [showTyping, setShowTyping] = useState(false);
	const {openQuestionDialog, closeQuestionDialog} = useQuestionDialog();
	const [useContentFiltering] = useState(true);
	const [showLogsForMessage, setShowLogsForMessage] = useState<EventInterface | null>(null);
	const [isMissingAgent, setIsMissingAgent] = useState<boolean | null>(null);

	const [agents] = useAtom(agentsAtom);
	const [session, setSession] = useAtom(sessionAtom);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	const [newSession, setNewSession] = useAtom(newSessionAtom);
	const [, setSessions] = useAtom(sessionsAtom);
	const {data: lastEvents, refetch, ErrorTemplate} = useFetch<EventInterface[]>(`sessions/${session?.id}/events`, {min_offset: lastOffset}, [], session?.id !== NEW_SESSION_ID, !!(session?.id && session?.id !== NEW_SESSION_ID), false);

	useEffect(() => {
		if (agents && agent?.id) {
			setIsMissingAgent(!agents?.find((a) => a.id === agent?.id));
		}
	}, [agents, agent?.id]);

	const resetChat = () => {
		setMessage('');
		setLastOffset(0);
		setMessages([]);
		setShowTyping(false);
		setShowLogsForMessage(null);
	};

	const resendMessageDialog = (index: number) => (sessionId: string, text?: string) => {
		const isLastMessage = index === messages.length - 1;
		const lastUserMessageOffset = messages[index].offset;

		if (isLastMessage) {
			setShowLogsForMessage(null);
			return resendMessage(index, sessionId, lastUserMessageOffset, text);
		}

		const onApproved = () => {
			setShowLogsForMessage(null);
			closeQuestionDialog();
			resendMessage(index, sessionId, lastUserMessageOffset, text);
		};

		const question = 'Resending this message would cause all of the following messages in the session to disappear.';
		openQuestionDialog('Are you sure?', question, [{text: 'Resend Anyway', onClick: onApproved, isMainAction: true}]);
	};

	const regenerateMessageDialog = (index: number) => (sessionId: string) => {
		const isLastMessage = index === messages.length - 1;
		const lastUserMessage = messages.findLast((message) => message.source === 'customer' && message.kind === 'message');
		const lastUserMessageOffset = lastUserMessage?.offset || messages.length - 1;

		if (isLastMessage) {
			setShowLogsForMessage(null);
			return regenerateMessage(index, sessionId, lastUserMessageOffset);
		}

		const onApproved = () => {
			setShowLogsForMessage(null);
			closeQuestionDialog();
			regenerateMessage(index, sessionId, lastUserMessageOffset);
		};

		const question = 'Regenerating this message would cause all of the following messages in the session to disappear.';
		openQuestionDialog('Are you sure?', question, [{text: 'Regenerate Anyway', onClick: onApproved, isMainAction: true}]);
	};

	const resendMessage = async (index: number, sessionId: string, offset: number, text?: string) => {
		const event = messages[index];
		const prevAllMessages = messages;
		const prevLastOffset = lastOffset;

		setMessages((messages) => messages.slice(0, index));
		setLastOffset(offset);
		const deleteSession = await deleteData(`sessions/${sessionId}/events?min_offset=${offset}`).catch((e) => ({error: e}));
		if (deleteSession?.error) {
			toast.error(deleteSession.error.message || deleteSession.error);
			setMessages(prevAllMessages);
			setLastOffset(prevLastOffset);
			return;
		}
		postMessage(text ?? event.data?.message);
		refetch();
	};

	const regenerateMessage = async (index: number, sessionId: string, offset: number) => {
		resendMessage(index - 1, sessionId, offset);
	};

	const resetSession = () => {
		if (newSession && session?.id !== NEW_SESSION_ID) setNewSession(null);
		resetChat();
		if (session?.id !== NEW_SESSION_ID) refetch();
		textareaRef?.current?.focus();
	};

	// eslint-disable-next-line react-hooks/exhaustive-deps
	useEffect(resetSession, [session?.id]);

	const createSession = async (): Promise<SessionInterface | undefined> => {
		if (!newSession) return;
		const {customer_id, title} = newSession;
		return postData('sessions?allow_greeting=true', {customer_id, agent_id: agent?.id, title} as object)
			.then((res: SessionInterface) => {
				if (newSession) {
					setSession(res);
					setNewSession(null);
				}
				setSessions((sessions) => [...sessions, res]);
				return res;
			})
			.catch(() => {
				toast.error('Something went wrong');
				return undefined;
			});
	};

	const postMessage = async (content: string): Promise<void> => {
		setPendingMessage((pendingMessage) => ({...pendingMessage, sessionId: session?.id, data: {message: content}}));
		setMessage('');
		const eventSession = newSession ? (await createSession())?.id : session?.id;
		const useContentFilteringStatus = useContentFiltering ? 'auto' : 'none';
		postData(`sessions/${eventSession}/events?moderation=${useContentFilteringStatus}`, {kind: 'message', message: content, source: 'customer'})
			.then(() => {
				refetch();
			})
			.catch(() => toast.error('Something went wrong'));
	};

	const handleKeydownInMessageInput = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submitButtonRef?.current?.click();
		} else if (e.key === 'Enter' && e.shiftKey) e.preventDefault();
	};

	const showLogs = (i: number) => (event: EventInterface) => {
		event.index = i;
		setShowLogsForMessage(event.id === showLogsForMessage?.id ? null : event);
	};

	return (
		<>
			<div className='flex items-center h-full w-full'>
				<div className='h-full min-w-[50%] flex flex-col'>
					<HeaderWrapper className={twJoin('border-e')}>
						{session?.id && (
							<div className='w-full flex items-center h-full'>
								<div className='h-full flex-1 flex items-center ps-[24px] border-e'>
									<div>
										<div>{agent?.name}</div>
										<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
											<CopyText preText='Agent ID:' text={` ${agent?.id}`} textToCopy={agent?.id} />
										</div>
									</div>
								</div>
								<div className='h-full flex-1 flex items-center ps-[24px]'>
									<div>
										<div>{(customer?.id == 'guest' && 'Guest') || customer?.name}</div>
										<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
											<CopyText preText='Customer ID:' text={` ${customer?.id}`} textToCopy={customer?.id} />
										</div>
									</div>
								</div>
							</div>
						)}
					</HeaderWrapper>
					<div className={twMerge('h-[21px] bg-white border-e border-t-0 bg-main')}></div>
					<div className={twMerge('flex flex-col items-center bg-white h-[calc(100%-70px)] mx-auto w-full flex-1 overflow-auto border-e bg-main')}>
						<Messages
							ErrorTemplate={ErrorTemplate}
							lastEvents={lastEvents}
							setShowTyping={setShowTyping}
							setLastOffset={setLastOffset}
							refetch={refetch}
							showTyping={showTyping}
							isMissingAgent={isMissingAgent}
							pendingMessage={pendingMessage}
							regenerateMessageDialog={regenerateMessageDialog}
							resendMessageDialog={resendMessageDialog}
							setPendingMessage={setPendingMessage}
							showLogs={showLogs}
							showLogsForMessage={showLogsForMessage}
						/>
						<div className={twMerge('w-full flex justify-between', isMissingAgent && 'hidden')}>
							<Spacer />
							<div className='group border flex-1 border-muted border-solid rounded-full flex flex-row justify-center items-center bg-white p-[0.9rem] ps-[24px] pe-0 h-[48.67px] max-w-[1200px] relative mb-[26px] hover:bg-main'>
								<img src='icons/edit.svg' alt='' className='me-[8px] h-[14px] w-[14px]' />
								<Textarea
									role='textbox'
									ref={textareaRef}
									placeholder='Message...'
									value={message}
									onKeyDown={handleKeydownInMessageInput}
									onChange={(e) => setMessage(e.target.value)}
									rows={1}
									className='box-shadow-none resize-none border-none h-full rounded-none min-h-[unset] p-0 whitespace-nowrap no-scrollbar font-inter font-light text-[16px] leading-[18px] bg-white group-hover:bg-main'
								/>
								<Button variant='ghost' data-testid='submit-button' className='max-w-[60px] rounded-full hover:bg-white' ref={submitButtonRef} disabled={!message?.trim() || !agent?.id} onClick={() => postMessage(message)}>
									<img src='icons/send.svg' alt='Send' height={19.64} width={21.52} className='h-10' />
								</Button>
							</div>
							<Spacer />
						</div>
					</div>
				</div>
				<ErrorBoundary component={<div className='flex h-full min-w-[50%] justify-center items-center text-[20px]'>Failed to load logs</div>}>
					<div className='flex h-full min-w-[50%]'>
						<MessageLogs
							event={showLogsForMessage}
							regenerateMessageFn={showLogsForMessage?.index ? regenerateMessageDialog(showLogsForMessage.index) : undefined}
							resendMessageFn={showLogsForMessage?.index || showLogsForMessage?.index === 0 ? resendMessageDialog(showLogsForMessage.index) : undefined}
							closeLogs={() => setShowLogsForMessage(null)}
						/>
					</div>
				</ErrorBoundary>
			</div>
		</>
	);
}
