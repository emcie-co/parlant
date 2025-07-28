/* eslint-disable react-hooks/exhaustive-deps */
import {useEffect, useRef, useState} from 'react';
import {twJoin, twMerge} from 'tailwind-merge';
import Markdown from '../markdown/markdown';
import Tooltip from '../ui/custom/tooltip';
import {Button} from '../ui/button';
import {useAtom} from 'jotai';
import {agentAtom, customerAtom, dialogAtom, sessionAtom} from '@/store';
import {getAvatarColor} from '../avatar/avatar';
import MessageRelativeTime from './message-relative-time';
import { Switch } from '../ui/switch';
import { copy } from '@/lib/utils';
import { Flag } from 'lucide-react';
import FlagMessage from '../message-details/flag-message';
import { EventInterface } from '@/utils/interfaces';

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
}


const MessageBubble = ({event, isFirstMessageInDate, showLogs, isContinual, showLogsForMessage, setIsEditing, flagged, flaggedChanged}: Props) => {
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
	// of the following use cases in draft/utterance switches:
	// 1. When both draft and utterance are multi-line
	// 2. When both draft and utterance are one-liners
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
	const formattedName = (isCustomer && isGuest) ? 'Guest' : name;

	return (
		<>
			<div className={twMerge(isCustomer ? 'justify-end' : 'justify-start', 'group/main flex-1 flex max-w-[min(1000px,100%)] items-end w-[calc(100%-412px)]  max-[1440px]:w-[calc(100%-160px)] max-[900px]:w-[calc(100%-40px)]')}>
				<div className='relative max-w-[80%]'>
					{(!isContinual || isFirstMessageInDate) && (
						<div className={twJoin('flex justify-between items-center mb-[12px] mt-[46px]', isFirstMessageInDate && 'mt-[0]', isCustomer && 'flex-row-reverse')}>
							<div className={twJoin('flex gap-[8px] items-center', isCustomer && 'flex-row-reverse')}>
								<div
									className='size-[26px] flex rounded-[6.5px] select-none items-center justify-center font-semibold'
									style={{color: isCustomer ? 'white' : colorPallete.text, background: isCustomer ? colorPallete.iconBackground : colorPallete?.background}}>
									{(isCustomer ? customerName?.[0] : agent?.name?.[0])?.toUpperCase()}
								</div>
								<div className='font-medium text-[14px] text-[#282828]'>{formattedName}</div>
							</div>
							<div className='flex items-center'>
								{!isCustomer && event.data?.draft && (
									<div className="flex">
										<div className='text-[14px] text-[#A9A9A9] font-light mr-1'>
											{'Show draft'}
										</div>
										<div className="mr-4">
											<Switch
												checked={showDraft}
												onCheckedChange={setShowDraft} />
										</div>
									</div>
								)}
								{flagged && (
									<div className='flex items-center gap-1 me-[1em] border-e pe-[1rem]'>
										<Tooltip value='View comment' side='top'>
											<Button
												variant='ghost'
												className='flex p-1 h-fit items-center gap-1'
												onClick={() => dialog.openDialog('Flag Message', <FlagMessage existingFlagValue={flagged || ''} event={event} sessionId={session?.id as string} onFlag={flaggedChanged}/>, {width: '600px', height: '636px'})}>
												<Flag size={16} color='#A1A1A1'/>
												<div className='text-[14px] text-[#A9A9A9] font-light'>{'Flagged'}</div>
											</Button>
										</Tooltip>
									</div>
									)}
								<MessageRelativeTime event={event} />
							</div>
						</div>
					)}
                    {showDraft && <div className='text-gray-400 px-[22px] py-[20px] bg-[#F5F6F8] rounded-[22px] mb-[10px]'>{event?.data?.draft}</div>}
					<div className='flex items-center relative max-w-full'>
						<div className={twMerge('self-stretch absolute invisible -left-[70px] top-[50%] -translate-y-1/2 items-center flex group-hover/main:visible peer-hover:visible hover:visible', !isCustomer && 'left-[unset] !-right-[40px]')}>
							<Tooltip value='Copy' side='top'>
								<div data-testid='copy-button' role='button' onClick={() => copy(event?.data?.message || '')} className='group cursor-pointer'>
									<img src='icons/copy.svg' alt='edit' className='block rounded-[10px] group-hover:bg-[#EBECF0] size-[30px] p-[5px]' />
								</div>
							</Tooltip>
							{isCustomer && <Tooltip value='Edit' side='top'>
								<div data-testid='edit-button' role='button' onClick={() => setIsEditing?.(true)} className='group cursor-pointer'>
									<img src='icons/edit-message.svg' alt='edit' className='block rounded-[10px] group-hover:bg-[#EBECF0] size-[30px] p-[5px]' />
								</div>
							</Tooltip>}
						</div>
						<div className='max-w-full'>
							<div
								ref={ref}
								tabIndex={0}
								data-testid='message'
								onClick={() => showLogs(event)}
								className={twMerge(
									'bg-green-light border-[2px] hover:bg-[#F5F9F3] text-black border-transparent cursor-pointer',
									isViewingCurrentMessage && '!bg-white hover:!bg-white border-[#EEEEEE] shadow-main',
									isCustomer && serverStatus === 'error' && '!bg-[#FDF2F1] hover:!bg-[#F5EFEF]',
									'max-w-[min(560px,100%)] peer w-[560px] flex items-center relative',
									event?.serverStatus === 'pending' && 'opacity-50',
									isOneLiner ? 'p-[13px_22px_17px_22px] rounded-[16px]' : 'p-[20px_22px_24px_22px] rounded-[22px]'
								)}>
								<div className={twMerge('markdown overflow-hidden relative min-w-[200px] max-w-[608px] [word-break:break-word] font-light text-[16px] pe-[38px]')}>
									<span ref={markdownRef}>
										<Markdown className={twJoin(!isOneLiner && 'leading-[26px]')}>
                                            {event?.data?.message || ''}
										</Markdown>
									</span>
								</div>
								<div className={twMerge('flex h-full font-normal text-[11px] text-[#AEB4BB] pe-[20px] font-inter self-end items-end whitespace-nowrap leading-[14px]', isOneLiner ? 'ps-[12px]' : '')}></div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</>
	);
};

export default MessageBubble;