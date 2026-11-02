import {useEffect, useRef, useState} from 'react';
import {spaceClick} from '@/utils/methods';
import {MINIMAL_UI} from '@/config';
import {twJoin, twMerge} from 'tailwind-merge';
import Markdown from '../markdown/markdown';
import Tooltip from '../ui/custom/tooltip';
import {Button} from '../ui/button';
import {useAtom} from 'jotai';
import {agentAtom, customerAtom, dialogAtom, sessionAtom} from '@/store';
import {getAvatarColor} from '../avatar/avatar';
// import MessageRelativeTime from './message-relative-time';
import {copy} from '@/lib/utils';
import {Download, Eye, EyeOff, Flag, Loader2, Search} from 'lucide-react';
import FlagMessage from '../message-details/flag-message';
import {EventInterface} from '@/utils/interfaces';
import {deliveredMessageText} from '@/utils/delivered-text';
import {revealLengthForText} from './reveal';
import DraftBubble from './draft-bubble';
import {BASE_URL} from '@/utils/api';
import {
	DiagnosticReportDownloadError,
	downloadDiagnosticReport,
} from '@/utils/diagnostic-report-download';
import {toast} from 'sonner';

interface Props {
	event: EventInterface;
	isContinual: boolean;
	isSameSourceAsPrevious?: boolean;
	isRegenerateHidden?: boolean;
	isFirstMessageInDate?: boolean;
	isInitializing?: boolean;
	flagged?: string;
	flaggedChanged?: (flagged: string) => void;
	showLogsForMessage?: EventInterface | null;
	regenerateMessageFn?: (sessionId: string) => void;
	resendMessageFn?: (sessionId: string, text?: string) => void;
	showLogs: (event: EventInterface) => void;
	setIsEditing?: React.Dispatch<React.SetStateAction<boolean>>;
	sameTraceMessages?: EventInterface[];
	shouldAutoScroll?: () => boolean;
	scrollToBottom?: (behavior?: ScrollBehavior) => void;
}

const MessageBubble = ({event, isFirstMessageInDate, showLogs, isContinual, isInitializing, showLogsForMessage, setIsEditing, flagged, flaggedChanged, sameTraceMessages: sameTraceMessages, shouldAutoScroll, scrollToBottom}: Props) => {
	const ref = useRef<HTMLDivElement>(null);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	const markdownRef = useRef<HTMLSpanElement>(null);
	const [showDraft, setShowDraft] = useState(false);
	const [, setRowCount] = useState(1);
	const [dialog] = useAtom(dialogAtom);
	const [session] = useAtom(sessionAtom);
	const [isDownloadingDiagnosticReport, setIsDownloadingDiagnosticReport] = useState(false);

	// Buffered reveal system: gradually reveal text at a controlled rate.
	// Voice messages show what was actually spoken, not what was generated.
	const messageText = deliveredMessageText(event);
	const chunks = event?.data?.chunks;
	const isAlreadyComplete = chunks !== undefined && chunks.length > 0 && chunks[chunks.length - 1] === null;

	const [revealedLength, setRevealedLength] = useState(isAlreadyComplete ? messageText.length : 0);
	const [animatedLength, setAnimatedLength] = useState(isAlreadyComplete ? messageText.length : 0); // tracks what's been animated (for fade effect)
	const revealedLengthRef = useRef(isAlreadyComplete ? messageText.length : 0);
	const animationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	useEffect(() => {
		if (!markdownRef?.current) return;
		const rowCount = Math.floor(markdownRef.current.offsetHeight / 24);
		setRowCount(rowCount + 1);
	}, [markdownRef, showDraft]);

	// Gradually reveal text at a smooth rate
	useEffect(() => {
		// Clamp if the text shrank: a voice message gets annotated with the text
		// that was actually spoken once delivery ends, replacing what's on screen
		// with a shorter string. Keep that shorter text fully revealed — this
		// effect won't run again, so resetting would leave the bubble blank.
		if (messageText.length < revealedLengthRef.current) {
			const clamped = revealLengthForText(messageText.length, revealedLengthRef.current);
			revealedLengthRef.current = clamped;
			setRevealedLength(clamped);
			setAnimatedLength(clamped);
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
	const revealedLengthForAnimation = useRef(0);
	revealedLengthForAnimation.current = revealedLength;

	useEffect(() => {
		// Reset if revealed length is less (new message)
		if (revealedLength < animatedLength) {
			if (animationTimerRef.current) {
				clearTimeout(animationTimerRef.current);
				animationTimerRef.current = null;
			}
			setAnimatedLength(revealedLength);
			return;
		}

		// Only start a new timer if there's no active timer and we have text to animate
		if (revealedLength > animatedLength && !animationTimerRef.current) {
			animationTimerRef.current = setTimeout(() => {
				animationTimerRef.current = null;
				// Use current revealedLength so all visible text becomes stable
				setAnimatedLength(revealedLengthForAnimation.current);
			}, 400); // Match animation duration
		}
	}, [revealedLength, animatedLength]);

	// Cleanup timer on unmount
	useEffect(() => {
		return () => {
			if (animationTimerRef.current) {
				clearTimeout(animationTimerRef.current);
			}
		};
	}, []);

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

	const openInspectionTeaser = () => {
		dialog.openDialog(
			'',
			<div className='relative flex h-full flex-col items-center justify-center gap-[16px] p-[32px] text-center font-manrope'>
				<img
					role='button'
					tabIndex={0}
					onKeyDown={spaceClick}
					onClick={() => dialog.closeDialog()}
					className='absolute top-[16px] right-[16px] cursor-pointer rounded-full'
					src='icons/close.svg'
					alt='close'
					width={24}
					height={24}
				/>
				<Search size={28} color='#3C8C71' />
				<div className='text-[18px] font-medium text-[#151515]'>Connect your local environment to Dashboard</div>
				<div className='text-[14px] text-[#777] max-w-[380px]'>
					See the full trace behind every reply — matched policies, tool decisions, and generation logs — in a complete development environment for your agents.
				</div>
				<a
					href='https://dashboard.parlant.io?utm_source=playground&utm_medium=inspect'
					target='_blank'
					rel='noopener noreferrer'
					className='bg-[#3C8C71] hover:bg-[#2f7a60] text-white rounded-[8px] px-[20px] h-[40px] text-[14px] font-medium flex items-center'>
					Open Dashboard →
				</a>
			</div>,
			{width: '480px', height: '320px'}
		);
	};
	const colorPallete = getAvatarColor((isCustomer ? customer?.id : agent?.id) || '', isCustomer ? 'customer' : 'agent');
	const name = isCustomer ? customer?.name : agent?.name;
	const formattedName = isCustomer && isGuest ? 'Guest' : name;
	const isEditDisabled = sameTraceMessages?.some((msg) => msg.serverStatus && msg.serverStatus !== 'ready' && msg.serverStatus !== 'error');
	const diagnosticReportUri = event.metadata?.__parlant__?.diagnostic_report_uri;

	const handleDiagnosticReportDownload = async () => {
		if (!diagnosticReportUri || isDownloadingDiagnosticReport) return;

		setIsDownloadingDiagnosticReport(true);
		const baseUrl = BASE_URL.replace(/\/$/, '');
		const reportUrl = `${baseUrl}/${diagnosticReportUri.replace(/^\//, '')}`;
		try {
			await downloadDiagnosticReport(reportUrl);
		} catch (error) {
			if (error instanceof DiagnosticReportDownloadError && error.status === 403) {
				toast.error('Administrator access is required to download this diagnostic report.');
			} else {
				toast.error('Could not download the diagnostic report. Please try again.');
			}
		} finally {
			setIsDownloadingDiagnosticReport(false);
		}
	};

	// Check if streaming is in progress (chunks property exists and not yet terminated with null)
	// chunks === undefined means block mode, chunks === [] means streaming started but no chunks yet,
	// chunks with null as last element means streaming is complete
	const isStreaming = chunks !== undefined && (chunks.length === 0 || chunks[chunks.length - 1] !== null);

	// Check if we're still revealing text (either streaming, reveal hasn't caught up, or animation hasn't caught up)
	const isRevealing = isStreaming || (chunks !== undefined && (revealedLength < messageText.length || animatedLength < revealedLength));

	// Track previous isRevealing state to detect when streaming completes
	const wasRevealingRef = useRef(false);

	// Auto-scroll during streaming to keep new text visible
	useEffect(() => {
		const scrollContainer = ref.current?.closest('.messages');
		if (!scrollContainer) return;

		const shouldScroll = shouldAutoScroll?.() ?? scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight <= 4;

		if (isRevealing && shouldScroll) {
			// During streaming: scroll to keep message visible
			if (scrollToBottom) {
				scrollToBottom('instant');
			} else {
				ref.current?.scrollIntoView({ behavior: 'instant', block: 'end' });
			}
		} else if (wasRevealingRef.current && !isRevealing && shouldScroll) {
			// Streaming just completed: scroll to very bottom of container
			if (scrollToBottom) {
				scrollToBottom('smooth');
			} else {
				scrollContainer.scrollTo({top: scrollContainer.scrollHeight, behavior: 'smooth'});
			}
		}

		wasRevealingRef.current = isRevealing;
	}, [revealedLength, isRevealing, shouldAutoScroll, scrollToBottom]);

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
									<Tooltip value={MINIMAL_UI ? 'Inspect this message' : 'View message actions and logs'} side='top'>
										<Button
											data-selected={isViewingCurrentMessage}
											variant='ghost'
											className='flex p-1 h-fit items-center gap-1'
											onClick={() => (MINIMAL_UI ? openInspectionTeaser() : showLogs(event))}>
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
											{/* Streamdown renders partial markdown safely as the revealed prefix grows.
											Only complete unbalanced markers while still revealing a prefix; once the
											full message is shown, render it verbatim so a lone "*" isn't auto-closed
											into a spurious trailing asterisk. */}
											<Markdown parseIncomplete={isRevealing} className={twJoin(!isOneLiner && 'leading-[26px]')}>{isRevealing ? messageText.slice(0, revealedLength) : messageText}</Markdown>
										{/* Blinking cursor for streaming messages */}
											{isStreaming && (
												<span className="inline-block w-[2px] h-[1em] bg-current align-text-bottom ml-[1px] animate-pulse" />
											)}
										</span>
									</div>
									<div className={twMerge('flex h-full font-normal text-[11px] text-[#AEB4BB] pe-[20px] font-manrope self-end items-end whitespace-nowrap leading-[14px]', isOneLiner ? 'ps-[12px]' : '')}></div>
								</div>
								{isInitializing && isCustomer && serverStatus === 'pending' && (
									<p className='flex items-center gap-[8px] mt-[10px] ps-[4px] font-normal text-[#A9AFB7] text-[14px] font-manrope animate-fade-in-delayed'>
										<Loader2 className='size-[14px] animate-spin' />
										Initializing session...
									</p>
								)}
								{isCustomer && serverStatus === 'error' && diagnosticReportUri && (
									<Button
										type='button'
										variant='link'
										size='sm'
										disabled={isDownloadingDiagnosticReport}
										onClick={handleDiagnosticReportDownload}
										className='mt-[6px] h-auto gap-[6px] px-[4px] py-[4px] text-[13px] text-[#6B5250] focus-visible:ring-[#9C6E69]'
									>
										{isDownloadingDiagnosticReport ? (
											<Loader2 className='size-[14px] animate-spin' aria-hidden='true' />
										) : (
											<Download className='size-[14px]' aria-hidden='true' />
										)}
										{isDownloadingDiagnosticReport ? 'Downloading...' : 'Download diagnostic report'}
									</Button>
								)}
							</div>
							<div className={twMerge('mx-[10px] self-stretch relative invisible items-center flex group-hover/main:visible peer-hover:visible hover:visible')}>
								<Tooltip value='Copy' side='top'>
									<div data-testid='copy-button' role='button' onClick={() => copy(messageText)} className='group cursor-pointer'>
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
