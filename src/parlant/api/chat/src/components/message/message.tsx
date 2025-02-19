/* eslint-disable react-hooks/exhaustive-deps */
import {ReactElement, Ref, useEffect, useRef, useState} from 'react';
import {EventInterface} from '@/utils/interfaces';
import styles from './message.module.scss';
import {Spacer} from '../ui/custom/spacer';
import {twJoin, twMerge} from 'tailwind-merge';
import Markdown from '../markdown/markdown';
import Tooltip from '../ui/custom/tooltip';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {useAtom} from 'jotai';
import {agentAtom, customerAtom, sessionAtom} from '@/store';
import {getAvatarColor} from '../agent-avatar/agent-avatar';
import {timeAgo} from '@/lib/utils';

interface Props {
	event: EventInterface;
	isContinual: boolean;
	isRegenerateHidden?: boolean;
	isFirstMessageInDate?: boolean;
	showLogsForMessage?: EventInterface | null;
	regenerateMessageFn?: (sessionId: string) => void;
	resendMessageFn?: (sessionId: string, text?: string) => void;
	showLogs: (event: EventInterface) => void;
	setIsEditing?: React.Dispatch<React.SetStateAction<boolean>>;
}

const statusIcon = {
	pending: <video src='mp4/loading.mp4' autoPlay loop data-testid='pending' height={12.2} width={12.2} className={'clip- ms-[4px] rounded-full ' + styles.pendingVideo} />,
	accepted: <img src='icons/v.svg' data-testid='accepted' height={11} width={11} className='ms-[4px]' alt='accepted' />,
	acknowledged: <img src='icons/v.svg' data-testid='acknowledged' height={11} width={11} className='ms-[4px]' alt='accepted' />,
	processing: <img src='icons/green-v.svg' data-testid='processing' height={11} width={11} className='ms-[4px]' alt='read' />,
	typing: <img src='icons/green-v.svg' data-testid='typing' height={11} width={11} className='ms-[4px]' alt='read' />,
	ready: <img src='icons/green-v.svg' data-testid='ready' height={11} width={11} className='ms-[4px]' alt='read' />,
	error: <img src='icons/error.svg' data-testid='error' height={11} width={11} className='ms-[4px]' alt='error' />,
	cancelled: <img src='icons/green-v.svg' title='canceled' data-testid='cancelled' height={11} width={11} className='ms-[4px]' alt='read' />,
};

const MessageBubble = ({event, isFirstMessageInDate, isContinual, showLogs, showLogsForMessage, setIsEditing}: Props) => {
	const ref = useRef<HTMLDivElement>(null);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	const markdownRef = useRef<HTMLSpanElement>(null);
	const [rowCount, setRowCount] = useState(1);

	useEffect(() => {
		if (!markdownRef?.current) return;
		const rowCount = Math.floor(markdownRef.current.offsetHeight / 24);
		setRowCount(rowCount + 1);
	}, [markdownRef]);

	const isOneLiner = rowCount === 1;

	const isCustomer = event.source === 'customer' || event.source === 'customer_ui';
	const serverStatus = event.serverStatus;
	const isGuest = customer?.name === '<guest>';
	const customerName = isGuest ? 'G' : customer?.name?.[0]?.toUpperCase();
	const isViewingCurrentMessage = showLogsForMessage && showLogsForMessage.id === event.id;
	const colorPallete = getAvatarColor((isCustomer ? customer?.id : agent?.id) || '');

	return (
		<>
			<div className={(isCustomer ? 'justify-end' : 'justify-start') + ' flex-1 flex max-w-[1200px] items-end w-[calc(100%-412px)]  max-[1440px]:w-[calc(100%-160px)] max-[900px]:w-[calc(100%-40px)]'}>
				{/* {!isCustomer && <div className='flex items-end me-[14px]'>{!isContinual ? <img src='parlant-bubble-muted.svg' alt='Parlant' height={36} width={36} /> : <div className='h-[36px] w-[36px]' />}</div>} */}
				{isCustomer && (
					<div className={twMerge('self-stretch mt-[30px] items-center px-[16px] flex invisible group-hover/main:visible peer-hover:visible hover:visible')}>
						<Tooltip value='Edit' side='left'>
							<div data-testid='edit-button' role='button' onClick={() => setIsEditing?.(true)} className='group cursor-pointer'>
								<img src='icons/edit-message.svg' alt='edit' className='block rounded-[10px] group-hover:bg-[#EBECF0] size-[30px] p-[5px]' />
							</div>
						</Tooltip>
					</div>
				)}
				<div>
					<div className={twJoin('flex justify-between items-center mb-[12px] mt-[46px]', isFirstMessageInDate && 'mt-[0]', isCustomer && 'flex-row-reverse')}>
						<div className={twJoin('flex gap-[8px] items-center', isCustomer && 'flex-row-reverse')}>
							<div className='size-[26px] flex text-white rounded-[6.5px] items-center justify-center font-bold' style={{background: colorPallete?.agentName}}>
								{isCustomer ? customerName?.[0] : agent?.name?.[0]}
							</div>
							<div className='font-medium text-[14px] text-[#282828]'>{isCustomer ? customer?.name : agent?.name}</div>
						</div>
						<div className='text-[14px] text-[#A9A9A9]'>{event.serverStatus === 'pending' ? 'Just Now' : timeAgo(event.creation_utc)}</div>
					</div>
					<div>
						<div
							ref={ref}
							tabIndex={0}
							data-testid='message'
							onClick={() => showLogs(event)}
							style={{
								background: isViewingCurrentMessage ? colorPallete?.background : '',
								borderColor: isViewingCurrentMessage ? colorPallete?.agentName : '',
							}}
							className={twMerge(
								'bg-[#F6F6F6] border-[2px] border-transparent',
								isCustomer && 'text-black hover:bg-[#F5F6F8] cursor-pointer',
								isCustomer && showLogsForMessage && showLogsForMessage.id !== event.id && 'bg-opacity-[0.33]',
								!isCustomer && ' hover:bg-[#F5F6F8] cursor-pointer',
								isCustomer && serverStatus === 'error' && '!bg-[#FDF2F1] hover:!bg-[#F5EFEF]',
								'max-w-fit peer w-fit flex items-center relative',
								isOneLiner ? 'p-[13px_22px_17px_22px] rounded-[16px]' : 'p-[20px_22px_24px_22px] rounded-[22px]'
							)}>
							<div className={twMerge('markdown overflow-hidden relative min-w-[200px] max-w-[608px] [word-break:break-word] font-light text-[16px] pe-[38px]')}>
								<span ref={markdownRef}>
									<Markdown className={twJoin(!isOneLiner && 'leading-[26px]')}>{event?.data?.message}</Markdown>
								</span>
							</div>
							<div className={twMerge('flex h-full font-normal text-[11px] text-[#AEB4BB] pe-[20px] font-inter self-end items-end whitespace-nowrap leading-[14px]', isOneLiner ? 'ps-[12px]' : '')}>
								<div className={twJoin('flex items-center justify-end', isCustomer && 'w-[46px]')}>{isCustomer && !!serverStatus && <div className='w-6'>{statusIcon[serverStatus]}</div>}</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</>
	);
};

const MessageEditing = ({event, resendMessageFn, setIsEditing}: Props) => {
	const ref = useRef<HTMLDivElement>(null);
	const textArea = useRef<HTMLTextAreaElement>(null);
	const [textValue, setTextValue] = useState(event?.data?.message || '');
	const [session] = useAtom(sessionAtom);

	useEffect(() => {
		textArea?.current?.select();
	}, [textArea?.current]);

	useEffect(() => {
		ref?.current?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
	}, [ref?.current]);

	return (
		<div ref={ref} className='w-full p-[16px] ps-[6px] pe-[6px] rounded-[16px] rounded-br-none border origin-bottom bg-[#f5f6f8] ' style={{transformOrigin: 'bottom'}}>
			<Textarea ref={textArea} className='resize-none h-[120px] pe-[108px] !ring-0 !ring-offset-0 border-none ps-[22px] bg-[#f5f6f8]' onChange={(e) => setTextValue(e.target.value)} defaultValue={textValue} />
			<div className='pt-[10px] flex justify-end gap-[10px] pe-[12px]'>
				<Button variant='ghost' onClick={() => setIsEditing?.(false)} className='rounded-[10px] hover:bg-white'>
					Cancel
				</Button>
				<Button disabled={!textValue?.trim() || textValue?.trim() === event?.data?.message} className='rounded-[10px]' onClick={() => resendMessageFn?.(session?.id || '', textValue?.trim())}>
					Apply
				</Button>
			</div>
		</div>
	);
};

export default function Message({event, isFirstMessageInDate, isContinual, showLogs, showLogsForMessage, resendMessageFn}: Props): ReactElement {
	const [isEditing, setIsEditing] = useState(false);

	return (
		<div className={twMerge(isEditing && '[direction:rtl] ')}>
			<div
				className={twMerge(
					'group/main flex my-[12px] mx-0 mb-1 w-full justify-between animate-fade-in scrollbar',
					isEditing && 'flex-1 flex max-w-[1200px] items-end w-[calc(100%-412px)] max-[2100px]:w-[calc(100%-200px)] self-end max-[1700px]:w-[calc(100%-40px)]'
				)}>
				<Spacer />
				{isEditing ? (
					<MessageEditing resendMessageFn={resendMessageFn} setIsEditing={setIsEditing} event={event} isContinual={isContinual} showLogs={showLogs} showLogsForMessage={showLogsForMessage} />
				) : (
					<MessageBubble isFirstMessageInDate={isFirstMessageInDate} setIsEditing={setIsEditing} event={event} isContinual={isContinual} showLogs={showLogs} showLogsForMessage={showLogsForMessage} />
				)}
				<Spacer />
			</div>
		</div>
	);
}
