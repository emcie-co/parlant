import {ReactElement, useEffect, useRef, useState} from 'react';
import {EventInterface} from '@/utils/interfaces';
import {getTimeStr} from '@/utils/date';
import styles from './message.module.scss';
import {Spacer} from '../ui/custom/spacer';
import {twJoin, twMerge} from 'tailwind-merge';
import Markdown from '../markdown/markdown';

interface Props {
	event: EventInterface;
	isContinual: boolean;
	isRegenerateHidden?: boolean;
	showLogsForMessage?: EventInterface | null;
	regenerateMessageFn?: (sessionId: string) => void;
	showLogs: (event: EventInterface) => void;
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

export default function Message({event, isContinual, showLogs, showLogsForMessage}: Props): ReactElement {
	const ref = useRef<HTMLDivElement>(null);
	const markdownRef = useRef<HTMLSpanElement>(null);
	const [rowCount, setRowCount] = useState(1);
	const isClient = event.source === 'customer' || event.source === 'customer_ui';
	const serverStatus = event.serverStatus;

	useEffect(() => {
		if (!markdownRef?.current) return;
		const rowCount = Math.floor(markdownRef.current.offsetHeight / 24);
		setRowCount(rowCount + 1);
	}, [markdownRef]);

	const isOneLiner = rowCount === 1;

	return (
		<div className='group/main flex my-4 mx-0 mb-1 w-full justify-between animate-fade-in scrollbar'>
			<Spacer />
			<div className={(isClient ? 'justify-end' : 'justify-start') + ' flex-1 flex max-w-[1200px] items-end w-[calc(100%-412px)]  max-[1440px]:w-[calc(100%-160px)] max-[900px]:w-[calc(100%-40px)]'}>
				{!isClient && <div className='flex items-end me-[14px]'>{!isContinual ? <img src='parlant-bubble-muted.svg' alt='Parlant' height={36} width={36} /> : <div className='h-[36px] w-[36px]' />}</div>}
				<div
					ref={ref}
					tabIndex={0}
					data-testid='message'
					onClick={() => !isClient && showLogs(event)}
					className={twMerge(
						isClient && 'text-black !rounded-br-none !rounded-tr-[22px]',
						isClient && showLogsForMessage && showLogsForMessage.id !== event.id && 'bg-opacity-[0.33] !border-[0.6px]',
						!isClient && '!rounded-bl-none bg-transparent  rounded-tl-[22px] hover:bg-[#F5F6F8] cursor-pointer',
						isClient && serverStatus === 'error' && '!bg-[#FDF2F1]',
						isContinual && '!rounded-br-[26px] !rounded-bl-[26px] !rounded-tl-[26px] !rounded-tr-[26px]',
						showLogsForMessage && showLogsForMessage.id === event.id && 'border-[#656565] !bg-white [box-shadow:-4.5px_6px_0px_0px_#DBDCE0]',
						'rounded-[26px] max-w-fit peer w-fit flex items-center relative border-[1.3px] border-muted border-solid'
					)}>
					<div className={twMerge('markdown overflow-auto relative max-w-[608px] [word-break:break-word] font-light text-[16px] ps-[32px] pe-[38px]', isOneLiner ? '!pb-[22px] !pt-[18px]' : 'pb-[24px] pt-[20px]')}>
						<span ref={markdownRef}>
							<Markdown className={twJoin(!isOneLiner && 'leading-[26px]')}>{event?.data?.message}</Markdown>
						</span>
					</div>
					<div className={twMerge('flex h-full font-normal text-[11px] text-[#AEB4BB] pb-[16px] pe-[20px] font-inter self-end items-end whitespace-nowrap leading-[14px]', isOneLiner ? '!pb-[10px] ps-[12px]' : '')}>
						<div className={twJoin('flex items-center', isClient && 'w-[46px]')}>
							<div>{getTimeStr(event.creation_utc)}</div>
							{isClient && !!serverStatus && <div className='w-6'>{statusIcon[serverStatus]}</div>}
						</div>
					</div>
				</div>
			</div>
			<Spacer />
		</div>
	);
}
