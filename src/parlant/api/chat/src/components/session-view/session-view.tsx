/* eslint-disable react-hooks/exhaustive-deps */
import React, {ReactElement, useEffect, useRef, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {deleteData, postData} from '@/utils/api';
import {groupBy} from '@/utils/obj';
import Message from '../message/message';
import {EventInterface, ServerStatus, SessionInterface} from '@/utils/interfaces';
import Spacer from '../ui/custom/spacer';
import {toast} from 'sonner';
import {NEW_SESSION_ID} from '../chat-header/chat-header';
import {useQuestionDialog} from '@/hooks/useQuestionDialog';
import {twJoin, twMerge} from 'tailwind-merge';
import MessageDetails from '../message-details/message-details';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, emptyPendingMessage, newSessionAtom, pendingMessageAtom, sessionAtom, sessionsAtom, viewingMessageDetailsAtom} from '@/store';
import ErrorBoundary from '../error-boundary/error-boundary';
import DateHeader from './date-header/date-header';
// import SessoinViewHeader from './session-view-header/session-view-header';
import {getIndexedItemsFromIndexedDB, isSameDay} from '@/lib/utils';
import {DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger} from '../ui/dropdown-menu';
import {ShieldEllipsis} from 'lucide-react';
import {soundDoubleBlip} from '@/utils/sounds';

const SessionView = (): ReactElement => {
	const lastMessageRef = useRef<HTMLDivElement>(null);
	const submitButtonRef = useRef<HTMLButtonElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const messagesRef = useRef<HTMLDivElement>(null);

	const [message, setMessage] = useState('');
	const [lastOffset, setLastOffset] = useState(0);
	const [messages, setMessages] = useState<EventInterface[]>([]);
	const [showTyping, setShowTyping] = useState(false);
	const [showThinking, setShowThinking] = useState(false);
	const [thinkingDisplay, setThinkingDisplay] = useState('');
	const [isFirstScroll, setIsFirstScroll] = useState(true);
	const {openQuestionDialog, closeQuestionDialog} = useQuestionDialog();
	const [useContentFiltering, setUseContentFiltering] = useState(false);
	const [showLogsForMessage, setShowLogsForMessage] = useState<EventInterface | null>(null);
	const [isMissingAgent, setIsMissingAgent] = useState<boolean | null>(null);
	const [isContentFilterMenuOpen, setIsContentFilterMenuOpen] = useState(false);
	const [flaggedItems, setFlaggedItems] = useState<Record<string, string>>({});
	const [refreshFlag, setRefreshFlag] = useState(false);
	const [pendingMessage, setPendingMessage] = useAtom<EventInterface>(pendingMessageAtom);
	const [agents] = useAtom(agentsAtom);
	const [session, setSession] = useAtom(sessionAtom);
	const [agent] = useAtom(agentAtom);
	const [newSession, setNewSession] = useAtom(newSessionAtom);
	const [, setViewingMessage] = useAtom(viewingMessageDetailsAtom);
	const [, setSessions] = useAtom(sessionsAtom);
	const {data: lastEvents, refetch, ErrorTemplate, abortFetch} = useFetch<EventInterface[]>(`sessions/${session?.id}/events`, {min_offset: lastOffset}, [], session?.id !== NEW_SESSION_ID, !!(session?.id && session?.id !== NEW_SESSION_ID), false);

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
		const prevMessages = messages.slice(0, index + 1);
		const lastUserMessageIndex = prevMessages.findLastIndex((message) => message.source === 'customer' && message.kind === 'message');
		const lastUserMessage = prevMessages[lastUserMessageIndex];
		const lastUserMessageOffset = lastUserMessage?.offset ?? messages.length - 1;

		if (isLastMessage) {
			setShowLogsForMessage(null);
			return regenerateMessage(lastUserMessageIndex, sessionId, lastUserMessageOffset);
		}

		const onApproved = () => {
			setShowLogsForMessage(null);
			closeQuestionDialog();
			regenerateMessage(lastUserMessageIndex, sessionId, lastUserMessageOffset);
		};

		const question = 'Regenerating this message would cause all of the following messages in the session to disappear.';
		openQuestionDialog('Are you sure?', question, [{text: 'Regenerate Anyway', onClick: onApproved, isMainAction: true}]);
	};

	const resendMessage = async (index: number, sessionId: string, offset: number, text?: string) => {
		const event = messages[index];

		const deleteSession = await deleteData(`sessions/${sessionId}/events?min_offset=${offset}`).catch((e) => ({error: e}));
		if (deleteSession?.error) {
			toast.error(deleteSession.error.message || deleteSession.error);
			return;
		}
		abortFetch?.();
		setLastOffset(offset);
		setMessages((messages) => messages.slice(0, index));
		postMessage(text ?? event.data?.message);
	};

	const regenerateMessage = async (index: number, sessionId: string, offset: number) => {
		resendMessage(index, sessionId, offset);
	};

	const formatMessagesFromEvents = () => {
		if (session?.id === NEW_SESSION_ID) return;
		const lastEvent = lastEvents?.at(-1);
		const lastStatusEvent = lastEvents?.findLast((e) => e.kind === 'status');
		if (!lastEvent) return;

		const offset = lastEvent?.offset;
		if (offset || offset === 0) setLastOffset(offset + 1);

		const correlationsMap = groupBy(lastEvents || [], (item: EventInterface) => item?.correlation_id.split('::')[0]);

		const newMessages = lastEvents?.filter((e) => e.kind === 'message') || [];
		const withStatusMessages = newMessages.map((newMessage, i) => {
			const data: EventInterface = {...newMessage};
			const item = correlationsMap?.[newMessage.correlation_id.split('::')[0]]?.at(-1)?.data;
			data.serverStatus = (item?.status || (newMessages[i + 1] ? 'ready' : null)) as ServerStatus;
			if (data.serverStatus === 'error') data.error = item?.data?.exception;
			return data;
		});

		setMessages((messages) => {
			const last = messages.at(-1);
			if (last?.source === 'customer' && correlationsMap?.[last?.correlation_id]) {
				last.serverStatus = correlationsMap[last.correlation_id].at(-1)?.data?.status || last.serverStatus;
				if (last.serverStatus === 'error') last.error = correlationsMap[last.correlation_id].at(-1)?.data?.data?.exception;
			}
			if (!withStatusMessages?.length) return [...messages];
			if (pendingMessage?.data?.message) setPendingMessage(emptyPendingMessage());

			const newVals: EventInterface[] = [];
			for (const messageArray of [messages, withStatusMessages]) {
				for (const message of messageArray) {
					newVals[message.offset] = message;
				}
			}
			return newVals.filter((message) => message);
		});

		const lastStatusEventStatus = lastStatusEvent?.data?.status;

		if (newMessages?.length && (showThinking || showTyping)) soundDoubleBlip(true);
		if (lastStatusEventStatus) {
			setShowThinking(lastStatusEventStatus === 'processing');

			if (lastStatusEventStatus === 'processing') {
				setThinkingDisplay(lastStatusEvent?.data?.data?.stage ?? 'Thinking');
			}

			setShowTyping(lastStatusEventStatus === 'typing');
		}
		refetch();
	};

	const scrollToLastMessage = () => {
		lastMessageRef?.current?.scrollIntoView?.({behavior: isFirstScroll ? 'instant' : 'smooth'});
		if (lastMessageRef?.current && isFirstScroll) setIsFirstScroll(false);
	};

	const resetSession = () => {
		setIsFirstScroll(true);
		if (newSession && session?.id !== NEW_SESSION_ID) setNewSession(null);
		resetChat();
		textareaRef?.current?.focus();
	};

	const getSessionFlaggedItems = async () => {
		const flaggedItems = await getIndexedItemsFromIndexedDB('Parlant-flags', 'message_flags', 'sessionIndex', session?.id as string, {name: 'sessionIndex', keyPath: 'sessionId'});
		const asMap = (flaggedItems as {correlationId: string; flagValue: string; sessionId: string}[]).reduce((acc, item) => {
			acc[item.correlationId] = item.flagValue;
			return acc;
		}, {} as Record<string, string>);
		setFlaggedItems(asMap);
	};

	useEffect(() => {
		getSessionFlaggedItems();
	}, [session?.id, refreshFlag]);

	useEffect(() => {
		if (lastOffset === 0) refetch();
	}, [lastOffset]);
	useEffect(() => setViewingMessage(showLogsForMessage), [showLogsForMessage]);
	useEffect(formatMessagesFromEvents, [lastEvents]);
	useEffect(scrollToLastMessage, [messages?.length, pendingMessage, isFirstScroll]);
	useEffect(resetSession, [session?.id]);
	useEffect(() => {
		if (showThinking || showTyping) lastMessageRef?.current?.scrollIntoView({behavior: 'smooth'});
	}, [showThinking, showTyping]);
	useEffect(() => {
		if (agents && agent?.id) setIsMissingAgent(!agents?.find((a) => a.id === agent?.id));
	}, [agents, agent?.id]);

	const createSession = async (): Promise<SessionInterface | undefined> => {
		if (!newSession) return;
		const {customer_id, title} = newSession;
		return postData('sessions?allow_greeting=false', {customer_id, agent_id: agent?.id, title} as object)
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
				soundDoubleBlip();
				refetch();
			})
			.catch(() => toast.error('Something went wrong'));
	};

	const handleTextareaKeydown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submitButtonRef?.current?.click();
		} else if (e.key === 'Enter' && e.shiftKey) e.preventDefault();
	};

	const isCurrSession = (session?.id === NEW_SESSION_ID && !pendingMessage?.id) || (session?.id !== NEW_SESSION_ID && pendingMessage?.sessionId === session?.id);
	const visibleMessages = (!messages?.length || isCurrSession) && pendingMessage?.data?.message ? [...messages, pendingMessage] : messages;

	const showLogs = (i: number) => (event: EventInterface) => {
		event.index = i;
		setShowLogsForMessage(event.id === showLogsForMessage?.id ? null : event);
	};

	return (
		<>
			<div ref={messagesRef} className={twMerge('flex items-center h-full w-full bg-white gap-[14px] rounded-[10px]', showLogsForMessage && 'bg-green-light')}>
				<div className={twMerge('h-full w-full pb-[14px] pt-[10px] rounded-[10px] flex flex-col transition-all duration-500 bg-white', showLogsForMessage && 'w-[calc(100%-min(700px,35vw))]')}>
					<div className='h-full flex flex-col rounded-[10px] m-auto w-full min-w-[unset]'>
						{/* <div className='h-[58px] bg-[#f5f5f9]'></div> */}
						{/* <SessoinViewHeader /> */}
						{/* <div className={twMerge('h-[21px] border-t-0 bg-white')}></div> */}
						<div className={twMerge('flex flex-col rounded-es-[16px] rounded-ee-[16px] items-center bg-white mx-auto w-full flex-1 overflow-hidden')}>
							<div
								className={twJoin(
									'messages fixed-scroll flex-1 flex flex-col w-full pb-4 overflow-x-hidden'
									// '[scroll-snap-type:y_mandatory]'
								)}
								aria-live='polite'
								role='log'
								aria-label='Chat messages'>
								{ErrorTemplate && <ErrorTemplate />}
								{visibleMessages.map((event, i) => (
									<React.Fragment key={(event.correlation_id || 0) + `${i}`}>
										{!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc) && <DateHeader date={event.creation_utc} isFirst={!i} bgColor='bg-white' />}
										<div ref={lastMessageRef} className='flex snap-end flex-col max-w-[min(1020px,100%)] w-[1020px] self-center'>
											<Message
												flaggedChanged={() => {
													setRefreshFlag((val) => !val);
												}}
												flagged={flaggedItems[event.correlation_id]}
												isFirstMessageInDate={!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc)}
												isRegenerateHidden={!!isMissingAgent}
												event={event}
												sameCorrelationMessages={visibleMessages.filter((e) => e.correlation_id === event.correlation_id)}
												isContinual={event.correlation_id === visibleMessages[i - 1]?.correlation_id || (event.source === 'customer' && visibleMessages[i - 1]?.source === 'customer')}
												regenerateMessageFn={regenerateMessageDialog(i)}
												resendMessageFn={resendMessageDialog(i)}
												showLogsForMessage={showLogsForMessage}
												showLogs={showLogs(i)}
											/>
										</div>
									</React.Fragment>
								))}
								{(showTyping || showThinking) && (
									<div ref={lastMessageRef} className='flex snap-end max-w-[min(1020px,100%)] w-[1020px] self-center'>
										<div className='bubblesWrapper snap-end' aria-hidden='true'>
											<div className='bubbles' />
										</div>
										{showTyping && <p className={twMerge('flex items-center font-normal text-[#A9AFB7] text-[14px] font-inter')}>Typing...</p>}
										{showThinking && <p className={twMerge('flex items-center font-normal text-[#A9AFB7] text-[14px] font-inter')}>{thinkingDisplay}...</p>}
									</div>
								)}
							</div>
							<div className={twMerge('w-full flex justify-between', isMissingAgent && 'hidden')}>
								<Spacer />
								<div className='group relative border flex-1 border-muted border-solid rounded-[10px] flex flex-row justify-center items-center bg-white p-[0.9rem] ps-[14px] pe-0 h-[48.67px] max-w-[1000px] mb-[26px]'>
									<DropdownMenu open={isContentFilterMenuOpen} onOpenChange={setIsContentFilterMenuOpen}>
										<DropdownMenuTrigger className='outline-none' data-testid='menu-button' tabIndex={-1} onClick={(e) => e.stopPropagation()}>
											<div className={twMerge('me-[2px] border border-transparent hover:bg-[#F3F5F9] rounded-[6px] size-[25px] flex items-center justify-center', isContentFilterMenuOpen && '!bg-[#f5f6f8]')}>
												{!useContentFiltering && <img src='icons/edit.svg' alt='' className={twMerge('h-[14px] w-[14px]')} />}
												{useContentFiltering && <ShieldEllipsis className={twJoin('size-[18px]')} />}
											</div>
										</DropdownMenuTrigger>
										<DropdownMenuContent side='top' align='start' className='max-w-[480px] -ms-[10px] flex flex-col gap-[8px] py-[14px] px-[10px] border-none [box-shadow:_0px_8px_20px_-8px_#00000012] rounded-[8px]'>
											<DropdownMenuItem
												tabIndex={0}
												onClick={() => setUseContentFiltering(false)}
												className={twMerge('gap-0  cursor-pointer font-normal text-[14px] px-[10px] font-inter capitalize hover:!bg-[#FAF9FF]', !useContentFiltering && '!bg-[#f5f6f8] hover:!bg-[#f5f6f8]')}>
												<img src='icons/edit.svg' alt='' className={twMerge('me-[8px] size-[15px]')} />
												Direct (No Moderation)
											</DropdownMenuItem>
											<DropdownMenuItem
												tabIndex={0}
												onClick={() => setUseContentFiltering(true)}
												className={twMerge('gap-0 !cursor-pointer font-normal text-[14px] items-start px-[10px] font-inter  hover:!bg-[#FAF9FF]', useContentFiltering && '!bg-[#f5f6f8] hover:!bg-[#f5f6f8]')}>
												<ShieldEllipsis className='me-[8px] !size-[17px] mt-[3px]' />
												<div>
													<div>Content Moderation</div>
													<small className='font-light'>
														Messages will be flagged for harmful or illicit content and censored accordingly. The agent will see such messages were sent and the reason why they were censored, but it won't see their content.
													</small>
												</div>
											</DropdownMenuItem>
										</DropdownMenuContent>
									</DropdownMenu>
									<Textarea
										role='textbox'
										ref={textareaRef}
										placeholder='Message...'
										value={message}
										onKeyDown={handleTextareaKeydown}
										onChange={(e) => setMessage(e.target.value)}
										rows={1}
										className='box-shadow-none placeholder:text-[#282828] resize-none border-none h-full rounded-none min-h-[unset] p-0 whitespace-nowrap no-scrollbar font-inter font-light text-[16px] leading-[100%] bg-white'
									/>
									<Button variant='ghost' data-testid='submit-button' className='max-w-[60px] rounded-full hover:bg-white' ref={submitButtonRef} disabled={!message?.trim() || !agent?.id} onClick={() => postMessage(message)}>
										<img src='icons/send.svg' alt='Send' height={19.64} width={21.52} className='h-10' />
									</Button>
								</div>
								<Spacer />
							</div>
							<div className='w-full'>
								<Spacer />
								<div></div>
								<Spacer />
							</div>
						</div>
					</div>
				</div>
				<ErrorBoundary component={<div className='flex h-full min-w-[50%] justify-center items-center text-[20px]'>Failed to load logs</div>}>
					<div
						className={twMerge(
							'fixed top-0 left-[unset] h-full right-0 z-[99] bg-white translate-x-[100%] max-w-[min(700px,35vw)] [box-shadow:0px_0px_30px_0px_#0000001F] w-[min(700px,35vw)] [transition-duration:600ms]',
							showLogsForMessage && 'translate-x-0'
						)}>
						{showLogsForMessage && (
							<MessageDetails
								flaggedChanged={() => {
									setRefreshFlag((val) => !val);
								}}
								sameCorrelationMessages={visibleMessages.filter((e) => e.correlation_id === showLogsForMessage.correlation_id)}
								event={showLogsForMessage}
								regenerateMessageFn={showLogsForMessage?.index ? regenerateMessageDialog(showLogsForMessage.index) : undefined}
								resendMessageFn={showLogsForMessage?.index || showLogsForMessage?.index === 0 ? resendMessageDialog(showLogsForMessage.index) : undefined}
								closeLogs={() => setShowLogsForMessage(null)}
							/>
						)}
					</div>
				</ErrorBoundary>
			</div>
		</>
	);
};

export default SessionView;
