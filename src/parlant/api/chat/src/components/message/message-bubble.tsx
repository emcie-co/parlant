/* eslint-disable react-hooks/exhaustive-deps */
import {useEffect, useRef, useState} from 'react';
import {twJoin, twMerge} from 'tailwind-merge';
import Markdown from '../markdown/markdown';
import Tooltip from '../ui/custom/tooltip';
import {Button} from '../ui/button';
import {useAtom} from 'jotai';
import {agentAtom, customerAtom, dialogAtom, sessionAtom} from '@/store';
import {getAvatarColor} from '../avatar/avatar';
// import MessageRelativeTime from './message-relative-time';
import {copy} from '@/lib/utils';
import {Eye, EyeOff, Flag, Search} from 'lucide-react';
import FlagMessage from '../message-details/flag-message';
import {EventInterface} from '@/utils/interfaces';
import DraftBubble from './draft-bubble';

interface Props {
	event: EventInterface;
	isContinual: boolean;
	isSameSourceAsPrevious?: boolean;
	isRegenerateHidden?: boolean;
	isFirstMessageInDate?: boolean;
	flagged?: string;
	flaggedChanged?: (flagged: string) => void;
	showLogsForMessage?: EventInterface | null;
	regenerateMessageFn?: (sessionId: string) => void;
	resendMessageFn?: (sessionId: string, text?: string) => void;
	showLogs: (event: EventInterface) => void;
	setIsEditing?: React.Dispatch<React.SetStateAction<boolean>>;
	sameTraceMessages?: EventInterface[];
}

const MessageBubble = ({event, isFirstMessageInDate, showLogs, isContinual, showLogsForMessage, setIsEditing, flagged, flaggedChanged, sameTraceMessages: sameTraceMessages}: Props) => {
	const ref = useRef<HTMLDivElement>(null);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	const markdownRef = useRef<HTMLSpanElement>(null);
	const [showDraft, setShowDraft] = useState(false);
	const [, setRowCount] = useState(1);
	const [dialog] = useAtom(dialogAtom);
	const [session] = useAtom(sessionAtom);

	// Buffered reveal system: gradually reveal text at a controlled rate
	const [revealedLength, setRevealedLength] = useState(0);
	const [animatedLength, setAnimatedLength] = useState(0); // tracks what's been animated (for fade effect)
	const messageText = event?.data?.message || '';
	const revealedLengthRef = useRef(0);

	useEffect(() => {
		if (!markdownRef?.current) return;
		const rowCount = Math.floor(markdownRef.current.offsetHeight / 24);
		setRowCount(rowCount + 1);
	}, [markdownRef, showDraft]);

	// Gradually reveal text at a smooth rate
	useEffect(() => {
		// Reset if message is shorter (new message)
		if (messageText.length < revealedLengthRef.current) {
			revealedLengthRef.current = 0;
			setRevealedLength(0);
			return;
		}

		// If we're already caught up, nothing to do
		if (revealedLengthRef.current >= messageText.length) return;

		// Reveal characters gradually
		const revealInterval = 30; // ms between reveals
		const charsPerReveal = 4; // characters to reveal each tick

		const timer = setInterval(() => {
			const currentRevealed = revealedLengthRef.current;
			const targetLength = messageText.length;

			if (currentRevealed >= targetLength) {
				clearInterval(timer);
				return;
			}

			// Reveal next chunk of characters
			const newLength = Math.min(currentRevealed + charsPerReveal, targetLength);
			revealedLengthRef.current = newLength;
			setRevealedLength(newLength);
		}, revealInterval);

		return () => clearInterval(timer);
	}, [messageText]);

	// Update animated length to catch up with revealed length (for fade effect)
	useEffect(() => {
		if (revealedLength > animatedLength) {
			const timer = setTimeout(() => {
				setAnimatedLength(revealedLength);
			}, 200); // Fade duration
			return () => clearTimeout(timer);
		}
		// Reset if revealed length is less (new message)
		if (revealedLength < animatedLength) {
			setAnimatedLength(revealedLength);
		}
	}, [revealedLength, animatedLength]);

	// FIXME:
	// rowCount SHOULD in fact be automatically calculated to
	// benefit from nice, smaller one-line message boxes.
	// However, currently we couldn't make it work in all
	// of the following use cases in draft/canned-response switches:
	// 1. When both draft and canned response are multi-line
	// 2. When both draft and canned response are one-liners
	// 3. When one is a one-liner and the other isn't
	// Therefore for now I'm disabling isOneLiner
	// until fixed.  -- Yam
	const isOneLiner = false; // FIXME: see above

	const isCustomer = event.source === 'customer' || event.source === 'customer_ui';
	const serverStatus = event.serverStatus;
	const isGuest = customer?.id === 'guest';
	const customerName = isGuest ? 'G' : customer?.name?.[0]?.toUpperCase();
	const isViewingCurrentMessage = showLogsForMessage && showLogsForMessage.id === event.id;
	const colorPallete = getAvatarColor((isCustomer ? customer?.id : agent?.id) || '', isCustomer ? 'customer' : 'agent');
	const name = isCustomer ? customer?.name : agent?.name;
	const formattedName = isCustomer && isGuest ? 'Guest' : name;
	const isEditDisabled = sameTraceMessages?.some((msg) => msg.serverStatus && msg.serverStatus !== 'ready' && msg.serverStatus !== 'error');

	// Check if streaming is in progress (chunks property exists and not yet terminated with null)
	// chunks === undefined means block mode, chunks === [] means streaming started but no chunks yet,
	// chunks with null as last element means streaming is complete
	const chunks = event?.data?.chunks;
	const isStreaming = chunks !== undefined && (chunks.length === 0 || chunks[chunks.length - 1] !== null);

	// Check if we're still revealing text (either streaming or reveal hasn't caught up)
	const isRevealing = isStreaming || (chunks !== undefined && revealedLength < messageText.length);

	// Auto-scroll during streaming to keep new text visible
	useEffect(() => {
		if (isRevealing && ref.current) {
			// Check if user is near the bottom of the scroll container before auto-scrolling
			const scrollContainer = ref.current.closest('.messages');
			if (scrollContainer) {
				const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
				const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
				if (isNearBottom) {
					ref.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
				}
			}
		}
	}, [revealedLength, isRevealing]);

	return (
		<>
			<div className={twMerge(isCustomer ? 'justify-end' : 'justify-start', 'flex-1 flex max-w-[min(1000px,100%)] items-end w-[calc(100%-412px)]  max-[1440px]:w-[calc(100%-160px)] max-[900px]:w-[calc(100%-40px)]')}>
				<div className='relative max-w-[80%]'>
					{(!isContinual || isFirstMessageInDate) && (
						<div className={twJoin('flex items-center mb-[12px] mt-[46px] max-w-[min(560px,100%)]', isCustomer && 'justify-self-end', isFirstMessageInDate && 'mt-[0]', isCustomer && 'flex-row-reverse')}>
							<div className={twJoin('flex items-center contents', isCustomer && 'flex-row-reverse')}>
								<div
									className={twMerge('size-[26px] min-h-[26px] min-w-[26px] flex rounded-[6.5px] select-none items-center justify-center font-semibold', isCustomer ? 'ms-[8px]' : 'me-[8px]')}
									style={{color: isCustomer ? 'white' : colorPallete.text, background: isCustomer ? colorPallete.iconBackground : colorPallete?.background}}>
									{(isCustomer ? customerName?.[0] : agent?.name?.[0])?.toUpperCase()}
								</div>
								<div className='font-medium text-[14px] text-[#282828] truncate'>{formattedName}</div>
							</div>
							<div className='flex items-center flex-1 justify-end'>
								{!isCustomer && sameTraceMessages?.some((e: EventInterface) => e.data?.draft) && (
									<div className='flex items-center me-[6px] pe-[6px] border-e border-[#EBECF0]'>
										<Tooltip value={showDraft ? 'Hide Draft' : 'Show Draft'} side='top'>
											<Button data-selected={showDraft} variant='ghost' className='flex p-1 h-fit items-center gap-1' onClick={() => setShowDraft(!showDraft)}>
												<div className='text-[14px] text-[#777] font-normal px-[.25em] flex items-center gap-[6px]'>
													{showDraft ? <Eye size={16} color='#777' /> : <EyeOff size={16} color='#777' />}
													Draft
												</div>
											</Button>
										</Tooltip>
									</div>
								)}
								{flagged && (
									<div className='flex items-center gap-1 pe-[6px] me-[6px] border-e border-[#EBECF0]'>
										<Tooltip value='View comment' side='top'>
											<Button
												variant='ghost'
												className='flex p-1 h-fit items-center gap-1'
												onClick={() =>
													dialog.openDialog('Flag Response', <FlagMessage existingFlagValue={flagged || ''} events={sameTraceMessages || [event]} sessionId={session?.id as string} onFlag={flaggedChanged} />, {width: '600px', height: '636px'})
												}>
												<Flag size={16} color='#777' />
												<div className='text-[14px] text-[#777] font-normal px-[.25em]'>{'Flagged'}</div>
											</Button>
										</Tooltip>
									</div>
								)}
								{/* <MessageRelativeTime event={event} /> */}
								{!isCustomer && (
									<Tooltip value='View message actions and logs' side='top'>
										<Button data-selected={isViewingCurrentMessage} variant='ghost' className='flex p-1 h-fit items-center gap-1' onClick={() => showLogs(event)}>
											<Search size={16} color='#777' />
											<div className='text-[14px] text-[#777] font-normal px-[.25em]'>Inspect</div>
										</Button>
									</Tooltip>
								)}
							</div>
						</div>
					)}
					<DraftBubble open={showDraft} draft={sameTraceMessages?.find((e) => e.data?.draft)?.data?.draft || ''} />
					<div className='group/main relative'>
						<div className={twMerge('flex items-center max-w-full', isCustomer && 'flex-row-reverse')}>
							<div className='max-w-full'>
								<div
									ref={ref}
									tabIndex={0}
									data-testid='message'
									className={twMerge(
										'bg-green-light border-[2px] hover:bg-[#F5F9F3] text-black border-transparent',
										// isViewingCurrentMessage && '!bg-white hover:!bg-white border-[#EEEEEE] shadow-main',
										isCustomer && serverStatus === 'error' && '!bg-[#FDF2F1] hover:!bg-[#F5EFEF]',
										'max-w-[min(560px,100%)] peer w-[560px] flex items-center relative',
										event?.serverStatus === 'pending' && 'opacity-50',
										isOneLiner ? 'p-[13px_22px_17px_22px] rounded-[16px]' : 'p-[20px_22px_24px_22px] rounded-[22px]'
									)}>
									<div className={twMerge('markdown overflow-hidden relative min-w-[200px] max-w-[608px] [word-break:break-word] font-light text-[16px] pe-[38px]')}>
										<span ref={markdownRef}>
											{isRevealing ? (
											<>
												{/* During reveal: split into stable text and newly revealed text with fade */}
												<span className={twJoin(!isOneLiner && 'leading-[26px]')}>{messageText.slice(0, animatedLength)}</span>
												{revealedLength > animatedLength && (
													<span className={twJoin(!isOneLiner && 'leading-[26px]', 'animate-fade-in-fast')}>{messageText.slice(animatedLength, revealedLength)}</span>
												)}
											</>
										) : (
											<Markdown className={twJoin(!isOneLiner && 'leading-[26px]')}>{messageText}</Markdown>
										)}
										{/* Blinking cursor for streaming messages */}
											{isStreaming && (
												<span className="inline-block w-[2px] h-[1em] bg-current align-text-bottom ml-[1px] animate-pulse" />
											)}
										</span>
									</div>
									<div className={twMerge('flex h-full font-normal text-[11px] text-[#AEB4BB] pe-[20px] font-inter self-end items-end whitespace-nowrap leading-[14px]', isOneLiner ? 'ps-[12px]' : '')}></div>
								</div>
							</div>
							<div className={twMerge('mx-[10px] self-stretch relative invisible items-center flex group-hover/main:visible peer-hover:visible hover:visible')}>
								<Tooltip value='Copy' side='top'>
									<div data-testid='copy-button' role='button' onClick={() => copy(event?.data?.message || '')} className='group cursor-pointer'>
										<img src='icons/copy.svg' alt='edit' className='block opacity-50 rounded-[10px] group-hover:bg-[#EBECF0] size-[30px] p-[5px]' />
									</div>
								</Tooltip>
								{isCustomer && !isEditDisabled && (
									<Tooltip value='Edit' side='top'>
										<div data-testid='edit-button' role='button' onClick={() => setIsEditing?.(true)} className='group cursor-pointer'>
											<img src='icons/edit-message.svg' alt='edit' className='block opacity-50 rounded-[10px] group-hover:bg-[#EBECF0] size-[30px] p-[5px]' />
										</div>
									</Tooltip>
								)}
							</div>
						</div>
					</div>
				</div>
			</div>
		</>
	);
};

export default MessageBubble;
