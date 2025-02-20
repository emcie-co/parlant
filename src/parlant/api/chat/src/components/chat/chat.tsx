import React, {ReactElement, useEffect, useRef, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {deleteData, postData} from '@/utils/api';
import {groupBy} from '@/utils/obj';
import Message from '../message/message';
import {EventInterface, SessionInterface} from '@/utils/interfaces';
import {getDateStr} from '@/utils/date';
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

const DateHeader = ({date, isFirst, bgColor}: {date: string | Date; isFirst: boolean; bgColor?: string}): ReactElement => {
	return (
		<div className={twMerge('text-center flex min-h-[30px] z-[1] bg-main h-[30px] pb-[4px] ps-[10px] mb-[60px] pt-[4px] mt-[76px] sticky -top-[1px]', isFirst && 'pt-[1px] !mt-0', bgColor)}>
			<div className='[box-shadow:0_-0.6px_0px_0px_#EBECF0] h-full -translate-y-[-50%] flex-1 ' />
			<div className='w-[136px] border-[0.6px] border-muted font-light text-[12px] bg-white text-[#656565] flex items-center justify-center rounded-[6px]'>{getDateStr(date)}</div>
			<div className='[box-shadow:0_-0.6px_0px_0px_#EBECF0] h-full -translate-y-[-50%] flex-1' />
		</div>
	);
};

export default function Chat(): ReactElement {
	const lastMessageRef = useRef<HTMLDivElement>(null);
	const submitButtonRef = useRef<HTMLButtonElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	const [message, setMessage] = useState('');
	const [pendingMessage, setPendingMessage] = useState<EventInterface>(emptyPendingMessage());
	const [lastOffset, setLastOffset] = useState(0);
	const [messages, setMessages] = useState<EventInterface[]>([]);
	const [showTyping, setShowTyping] = useState(false);
	const [isRegenerating, setIsRegenerating] = useState(false);
	const [isFirstScroll, setIsFirstScroll] = useState(true);
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
	const {data: lastMessages, refetch, ErrorTemplate} = useFetch<EventInterface[]>(`sessions/${session?.id}/events`, {min_offset: lastOffset}, [], session?.id !== NEW_SESSION_ID, !!(session?.id && session?.id !== NEW_SESSION_ID), false);

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

	const regenerateMessageDialog = (index: number) => (sessionId: string) => {
		const isLastMessage = index === messages.length - 1;
		const lastUserMessageOffset = messages[index - 1].offset;
		if (isLastMessage) {
			setShowLogsForMessage(null);
			return regenerateMessage(index, sessionId, lastUserMessageOffset + 1);
		}

		const onApproved = () => {
			setShowLogsForMessage(null);
			closeQuestionDialog();
			regenerateMessage(index, sessionId, lastUserMessageOffset + 1);
		};

		const question = 'Regenerating this message would cause all of the following messages in the session to disappear.';
		openQuestionDialog('Are you sure?', question, [{text: 'Regenerate Anyway', onClick: onApproved, isMainAction: true}]);
	};

	const regenerateMessage = async (index: number, sessionId: string, offset: number) => {
		const prevAllMessages = messages;
		const prevLastOffset = lastOffset;

		setMessages((messages) => messages.slice(0, index));
		setLastOffset(offset);
		setIsRegenerating(true);
		const deleteSession = await deleteData(`sessions/${sessionId}/events?min_offset=${offset}`).catch((e) => ({error: e}));
		if (deleteSession?.error) {
			toast.error(deleteSession.error.message || deleteSession.error);
			setMessages(prevAllMessages);
			setLastOffset(prevLastOffset);
			return;
		}
		postData(`sessions/${sessionId}/events`, {kind: 'message', source: 'ai_agent'});
		refetch();
	};

	useEffect(() => {
		lastMessageRef?.current?.scrollIntoView?.({behavior: isFirstScroll ? 'instant' : 'smooth'});
		if (lastMessageRef?.current && isFirstScroll) setIsFirstScroll(false);
	}, [messages, pendingMessage, isFirstScroll]);

	useEffect(() => {
		setIsFirstScroll(true);
		if (newSession && session?.id !== NEW_SESSION_ID) setNewSession(null);
		resetChat();
		if (session?.id !== NEW_SESSION_ID) refetch();
		textareaRef?.current?.focus();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [session?.id]);

	useEffect(() => {
		if (session?.id === NEW_SESSION_ID) return;
		const lastEvent = lastMessages?.at(-1);
		if (!lastEvent) return;
		const offset = lastEvent?.offset;
		if (offset || offset === 0) setLastOffset(offset + 1);
		const correlationsMap = groupBy(lastMessages || [], (item: EventInterface) => item?.correlation_id.split('::')[0]);
		const newMessages = lastMessages?.filter((e) => e.kind === 'message') || [];
		const withStatusMessages = newMessages.map((newMessage, i) => ({
			...newMessage,
			serverStatus: correlationsMap?.[newMessage.correlation_id.split('::')[0]]?.at(-1)?.data?.status || (newMessages[i + 1] ? 'ready' : null),
		}));
		if (newMessages.length && isRegenerating) setIsRegenerating(false);

		if (pendingMessage.serverStatus !== 'pending' && pendingMessage.data.message) setPendingMessage(emptyPendingMessage);
		setMessages((messages) => {
			const last = messages.at(-1);
			if (last?.source === 'customer' && correlationsMap?.[last?.correlation_id]) last.serverStatus = correlationsMap[last.correlation_id].at(-1)?.data?.status || last.serverStatus;
			return [...messages, ...withStatusMessages] as EventInterface[];
		});

		const lastEventStatus = lastEvent?.data?.status;

		setShowTyping(lastEventStatus === 'typing');
		if (lastEventStatus === 'error') {
			if (isRegenerating) {
				setIsRegenerating(false);
				toast.error('Something went wrong');
			}
		}
		refetch();

		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [lastMessages]);

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
				setPendingMessage((pendingMessage) => ({...pendingMessage, serverStatus: 'accepted'}));
				refetch();
			})
			.catch(() => toast.error('Something went wrong'));
	};

	const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submitButtonRef?.current?.click();
		} else if (e.key === 'Enter' && e.shiftKey) e.preventDefault();
	};

	const isSameDay = (dateA: string | Date, dateB: string | Date): boolean => {
		if (!dateA) return false;
		return new Date(dateA).toLocaleDateString() === new Date(dateB).toLocaleDateString();
	};

	const visibleMessages = session?.id !== NEW_SESSION_ID && pendingMessage?.sessionId === session?.id && pendingMessage?.data?.message ? [...messages, pendingMessage] : messages;

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
						<div className='messages fixed-scroll flex-1 flex flex-col w-full pb-4' aria-live='polite' role='log' aria-label='Chat messages'>
							{ErrorTemplate && <ErrorTemplate />}
							{visibleMessages.map((event, i) => (
								<React.Fragment key={i}>
									{!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc) && <DateHeader date={event.creation_utc} isFirst={!i} bgColor='bg-main' />}
									<div ref={lastMessageRef} className='flex flex-col'>
										<Message
											isRegenerateHidden={!!isMissingAgent}
											event={event}
											isContinual={event.source === visibleMessages[i + 1]?.source}
											regenerateMessageFn={regenerateMessageDialog(i)}
											showLogsForMessage={showLogsForMessage}
											showLogs={showLogs(i)}
										/>
									</div>
								</React.Fragment>
							))}
							{(isRegenerating || showTyping) && (
								<div className='animate-fade-in flex mb-1 justify-between mt-[44.33px]'>
									<Spacer />
									<div className='flex items-center max-w-[1200px] flex-1'>
										<img src='parlant-bubble-muted.svg' alt='' height={36} width={36} className='me-[8px]' />
										<p className='font-medium text-[#A9AFB7] text-[11px] font-inter'>{isRegenerating ? 'Regenerating...' : 'Typing...'}</p>
									</div>
									<Spacer />
								</div>
							)}
						</div>
						<div className={twMerge('w-full flex justify-between', isMissingAgent && 'hidden')}>
							<Spacer />
							<div className='group border flex-1 border-muted border-solid rounded-full flex flex-row justify-center items-center bg-white p-[0.9rem] ps-[24px] pe-0 h-[48.67px] max-w-[1200px] relative mb-[26px] hover:bg-main'>
								<img src='icons/edit.svg' alt='' className='me-[8px] h-[14px] w-[14px]' />
								<Textarea
									role='textbox'
									ref={textareaRef}
									placeholder='Message...'
									value={message}
									onKeyDown={onKeyDown}
									onChange={(e) => setMessage(e.target.value)}
									rows={1}
									className='box-shadow-none resize-none border-none h-full rounded-none min-h-[unset] p-0 whitespace-nowrap no-scrollbar font-inter font-light text-[16px] leading-[18px] bg-white group-hover:bg-main'
								/>
								<Button variant='ghost' data-testid='submit-button' className='max-w-[60px] rounded-full hover:bg-white' ref={submitButtonRef} disabled={!message?.trim() || !agent?.id || isRegenerating} onClick={() => postMessage(message)}>
									<img src='icons/send.svg' alt='Send' height={19.64} width={21.52} className='h-10' />
								</Button>
							</div>
							<Spacer />
						</div>
					</div>
				</div>
				<ErrorBoundary component={<div className='flex h-full min-w-[50%] justify-center items-center text-[20px]'>Failed to load logs</div>}>
					<div className='flex h-full min-w-[50%]'>
						<MessageLogs event={showLogsForMessage} regenerateMessageFn={showLogsForMessage?.index ? regenerateMessageDialog(showLogsForMessage.index) : undefined} closeLogs={() => setShowLogsForMessage(null)} />
					</div>
				</ErrorBoundary>
			</div>
		</>
	);
}
