import { ReactElement } from 'react';
import Markdown from 'react-markdown';
import { EventInterface } from '@/utils/interfaces';
import { getTimeStr } from '@/utils/date';
import styles from './message.module.scss';
import { Spacer } from '../ui/custom/spacer';
import { useSession } from '../chatbot/chatbot';
import Tooltip from '../ui/custom/tooltip';
import { twMerge } from 'tailwind-merge';

interface Props {
    event: EventInterface;
    isContinual: boolean;
    isRegenerateHidden?: boolean;
    regenerateMessageFn?: (sessionId: string) => void;
}

const statusIcon = {
    pending: <video src='mp4/loading.mp4' autoPlay loop data-testid="pending" height={12.2} width={12.2} className={'clip- ms-[4px] rounded-full ' + styles.pendingVideo}/>,
    accepted: <img src='icons/v.svg' data-testid="accepted" height={11} width={11} className='ms-[4px]' alt='accepted'/>,
    acknowledged: <img src='icons/v.svg' data-testid="acknowledged" height={11} width={11} className='ms-[4px]' alt='accepted'/>,
    processing: <img src='icons/green-v.svg' data-testid="processing" height={11} width={11} className='ms-[4px]' alt='read'/>,
    typing: <img src='icons/green-v.svg' data-testid="typing" height={11} width={11} className='ms-[4px]' alt='read'/>,
    ready: <img src='icons/green-v.svg' data-testid="ready" height={11} width={11} className='ms-[4px]' alt='read'/>,
    error: <img src='icons/error.svg' data-testid="error" height={11} width={11} className='ms-[4px]' alt='error'/>,
    cancelled: <img src='icons/green-v.svg' title='canceled' data-testid="cancelled" height={11} width={11} className='ms-[4px]' alt='read'/>,
};

export default function Message({event, isContinual, isRegenerateHidden, regenerateMessageFn}: Props): ReactElement {
    const {sessionId} = useSession();
    const isClient = event.source === 'customer' || event.source === 'customer_ui';
    const serverStatus = event.serverStatus;

    return (
        <div className='flex my-4 mx-0 mb-1 w-full justify-between animate-fade-in'>
            <Spacer/>
            <div className={(isClient ? 'justify-end' : 'justify-start') + ' flex-1 flex max-w-[1200px] items-end'}>
                {!isClient &&
                    <div className='flex items-end me-[14px]'>
                        {!isContinual ? <img src="parlant-bubble-muted.svg" alt="Parlant" height={36} width={36}/> : <div className='h-[36px] w-[36px]'/>}
                    </div>
                }
                <div tabIndex={0} data-testid="message" className={(isClient ? 'bg-white text-black rounded-br-none rounded-tr-[22px]' : 'bg-transparent border-[1.3px] border-muted border-solid rounded-bl-none rounded-tl-[22px]') + (isClient && serverStatus === 'error' ? ' !bg-[#FDF2F1]' : '') + (isContinual ? ' !rounded-br-[26px] !rounded-bl-[26px] !rounded-tl-[26px] !rounded-tr-[26px]' : '') + ' rounded-[26px] peer w-fit max-w-[min(564px,85%)] flex gap-1 items-center relative'}>
                    <div className="markdown relative max-w-full [word-break:break-word] font-light text-[16px] pt-[18px] pb-[22px] ps-[32px] pe-[24px]">
                        <Markdown className={styles.markdown}>{event?.data?.message}</Markdown>
                    </div>
                    <div className='flex h-full font-normal text-[11px] text-[#AEB4BB] pt-[36px] pb-[10px] pe-[12px] font-inter self-end items-end whitespace-nowrap'>
                        <div className='flex items-center w-[46px]'>
                            <div>{getTimeStr(event.creation_utc)}</div>
                            {isClient && serverStatus && <div className="w-6">{statusIcon[serverStatus]}</div>}
                        </div>
                    </div>
                </div>
                {!isClient &&
                <div className={twMerge('self-stretch items-center px-[16px] flex invisible peer-hover:visible hover:visible', isRegenerateHidden && 'hidden')}>
                    <Tooltip value='Regenerate' side='right'>
                        <div data-testid='regenerate-button'role='button' onClick={() => regenerateMessageFn?.(sessionId as string)} className='group cursor-pointer'>
                            <img src="icons/regenerate.svg" alt="regenerate" className='block group-hover:hidden'/>
                            <img src="icons/regenerate-filled.svg" alt="regenerate" className='hidden group-hover:block'/>
                        </div>
                    </Tooltip>
                </div>}
            </div>
            <Spacer/>
        </div>
    );
}