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
	sameCorrelationMessages?: EventInterface[];
}

const MessageBubble = ({event, isFirstMessageInDate, showLogs, isContinual, showLogsForMessage, setIsEditing, flagged, flaggedChanged, sameCorrelationMessages}: Props) => {
	const ref = useRef<HTMLDivElement>(null);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	const markdownRef = useRef<HTMLSpanElement>(null);
	const [showDraft, setShowDraft] = useState(false);
	const [, setRowCount] = useState(1);
	const [dialog] = useAtom(dialogAtom);
	const [session] = useAtom(sessionAtom);

	useEffect(() => {
		if (!markdownRef?.current) return;
		const rowCount = Math.floor(markdownRef.current.offsetHeight / 24);
		setRowCount(rowCount + 1);
	}, [markdownRef, showDraft]);

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
								{!isCustomer && sameCorrelationMessages?.some((e: EventInterface) => e.data?.draft) && (
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
													dialog.openDialog('Flag Response', <FlagMessage existingFlagValue={flagged || ''} events={sameCorrelationMessages || [event]} sessionId={session?.id as string} onFlag={flaggedChanged} />, {width: '600px', height: '636px'})
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
					<DraftBubble open={showDraft} draft={sameCorrelationMessages?.find((e) => e.data?.draft)?.data?.draft || ''} />
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
											<Markdown className={twJoin(!isOneLiner && 'leading-[26px]')}>{event?.data?.message || ''}</Markdown>
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
								{isCustomer && (
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
