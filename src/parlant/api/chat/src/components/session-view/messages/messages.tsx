import {NEW_SESSION_ID} from '@/components/agents-list/agent-list';
import Message from '@/components/message/message';
import ProgressImage from '@/components/progress-logo/progress-logo';
import {Spacer} from '@/components/ui/custom/spacer';
import {sessionAtom} from '@/store';
import {getDateStr} from '@/utils/date';
import {EventInterface, ServerStatus, SessionInterface} from '@/utils/interfaces';
import {groupBy} from '@/utils/obj';
import {SetStateAction, useAtom} from 'jotai';
import React, {Dispatch, JSXElementConstructor, ReactElement, useEffect, useRef, useState} from 'react';
import {twMerge} from 'tailwind-merge';

interface Props {
	pendingMessage: EventInterface;
	isMissingAgent: boolean | null;
	regenerateMessageDialog: Function;
	resendMessageDialog: Function;
	showLogs: Function;
	setPendingMessage: Function;
	showTyping: boolean;
	setShowTyping: Dispatch<SetStateAction<boolean>>;
	showLogsForMessage: EventInterface | null;
	lastEvents: EventInterface[] | null;
	ErrorTemplate: (() => ReactElement<any, string | JSXElementConstructor<any>>) | null;
	setLastOffset: Function;
	refetch: () => void;
}

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

const isSameDay = (dateA: string | Date, dateB: string | Date): boolean => {
	if (!dateA) return false;
	return new Date(dateA).toLocaleDateString() === new Date(dateB).toLocaleDateString();
};

const DateHeader = ({date, isFirst, bgColor}: {date: string | Date; isFirst: boolean; bgColor?: string}): ReactElement => {
	return (
		<div className={twMerge('text-center flex min-h-[30px] z-[1] bg-main h-[30px] pb-[4px] ps-[10px] mb-[60px] pt-[4px] mt-[76px] sticky -top-[1px]', isFirst && 'pt-[1px] !mt-0', bgColor)}>
			<div className='[box-shadow:0_-0.6px_0px_0px_#EBECF0] h-full -translate-y-[-50%] flex-1 ' />
			<div className='w-[136px] border-[0.6px] border-muted font-light text-[12px] bg-white text-[#656565] flex items-center justify-center rounded-[6px]'>{getDateStr(date)}</div>
			<div className='[box-shadow:0_-0.6px_0px_0px_#EBECF0] h-full -translate-y-[-50%] flex-1' />
		</div>
	);
};

const Messages = ({refetch, showTyping, ErrorTemplate, setShowTyping, pendingMessage, lastEvents, isMissingAgent, setLastOffset, regenerateMessageDialog, resendMessageDialog, showLogsForMessage, showLogs, setPendingMessage}: Props) => {
	const lastMessageRef = useRef<HTMLDivElement>(null);
	const [session] = useAtom(sessionAtom);
	const [messages, setMessages] = useState<EventInterface[]>([]);
	const visibleMessages = session?.id !== NEW_SESSION_ID && pendingMessage?.sessionId === session?.id && pendingMessage?.data?.message ? [...messages, pendingMessage] : messages;
	const [showThinking, setShowThinking] = useState(false);
	const [isFirstScroll, setIsFirstScroll] = useState(true);

	const buildMessageStructure = () => {
		if (session?.id === NEW_SESSION_ID) return;
		const lastEvent = lastEvents?.at(-1);
		if (!lastEvent) return;

		const offset = lastEvent.offset;
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

		if (pendingMessage.serverStatus !== 'pending' && pendingMessage.data.message) setPendingMessage(emptyPendingMessage);

		setMessages((messages) => {
			const last = messages.at(-1);
			if (last?.source === 'customer' && correlationsMap?.[last?.correlation_id]) {
				last.serverStatus = correlationsMap[last.correlation_id].at(-1)?.data?.status || last.serverStatus;
				if (last.serverStatus === 'error') last.error = correlationsMap[last.correlation_id].at(-1)?.data?.data?.exception;
			}
			if (withStatusMessages && pendingMessage) setPendingMessage(emptyPendingMessage);
			return [...messages, ...withStatusMessages] as EventInterface[];
		});

		const lastEventStatus = lastEvent?.data?.status;

		setShowThinking(!!messages?.length && lastEventStatus === 'processing');
		setShowTyping(lastEventStatus === 'typing');

		refetch();
	};

	const scrollToLastMessage = () => {
		lastMessageRef?.current?.scrollIntoView?.({behavior: isFirstScroll ? 'instant' : 'smooth'});
		if (lastMessageRef?.current && isFirstScroll) setIsFirstScroll(false);
	};

	// eslint-disable-next-line react-hooks/exhaustive-deps
	useEffect(buildMessageStructure, [lastEvents]);
	useEffect(scrollToLastMessage, [messages, pendingMessage, isFirstScroll]);

	return (
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
							resendMessageFn={resendMessageDialog(i)}
							showLogsForMessage={showLogsForMessage}
							showLogs={showLogs(i)}
						/>
					</div>
				</React.Fragment>
			))}
			{(showTyping || showThinking) && (
				<div className='animate-fade-in flex mb-1 justify-between mt-[44.33px]'>
					<Spacer />
					<div className='flex items-center max-w-[1200px] flex-1'>
						<ProgressImage phace={showThinking ? 'thinking' : 'typing'} />
						<p className='font-medium text-[#A9AFB7] text-[11px] font-inter'>{showTyping ? 'Typing...' : 'Thinking...'}</p>
					</div>
					<Spacer />
				</div>
			)}
		</div>
	);
};

export default Messages;
