import React, { ReactElement, useEffect, useRef, useState } from 'react';
import useFetch from '@/hooks/useFetch';
import { Textarea } from '../ui/textarea';
import { Button } from '../ui/button';
import { postData } from '@/utils/api';
import { groupBy } from '@/utils/obj';
import Message from '../message/message';
import { useSession } from '../chatbot/chatbot';
import { EventInterface, SessionInterface } from '@/utils/interfaces';
import { getDateStr } from '@/utils/date';
import { Spacer } from '../ui/custom/spacer';
import { toast } from 'sonner';
import { NEW_SESSION_ID } from '../chat-header/chat-header';

const emptyPendingMessage: EventInterface = {
    kind: 'message',
    source: 'end_user',
    creation_utc: new Date(),
    serverStatus: 'pending',
    offset: 0,
    correlation_id: '',
    data: {
        message: ''
    }
};

const DateHeader = ({date, isFirst}: {date: string | Date, isFirst: boolean}): ReactElement => {
    return (
        <div className={'text-center flex min-h-[30px] z-[1] bg-main h-[30px] pb-[4px] mb-[60px] pt-[4px] mt-[76px] sticky top-0' + (isFirst ? ' pt-0 !mt-0' : '')}>
            <hr className='h-full -translate-y-[-50%] flex-1'/>
            <div className='w-[136px] border-[0.6px] border-muted font-light text-[12px] bg-white text-[#656565] flex items-center justify-center rounded-[6px]'>
                {getDateStr(date)}
            </div>
            <hr className='h-full -translate-y-[-50%] flex-1' />
        </div>
    );
};

export default function Chat(): ReactElement {
    const lastMessageRef = useRef<HTMLDivElement>(null);
    const submitButtonRef = useRef<HTMLButtonElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    
    const [message, setMessage] = useState('');
    const [pendingMessage, setPendingMessage] = useState<EventInterface>(emptyPendingMessage);
    const [lastOffset, setLastOffset] = useState(0);
    const [messages, setMessages] = useState<EventInterface[]>([]);
    const [showTyping, setShowTyping] = useState(false);
    const [isFirstScroll, setIsFirstScroll] = useState(true);
    
    const {sessionId, setSessionId, agentId, newSession, setNewSession, setSessions} = useSession();
    const {data: lastMessages, refetch, ErrorTemplate} = useFetch<{events: EventInterface[]}>(`sessions/${sessionId}/events`, {min_offset: lastOffset, wait: true}, [], sessionId !== NEW_SESSION_ID, !!(sessionId && sessionId !== NEW_SESSION_ID));

    const resetChat = () => {
        setMessage('');
        setLastOffset(0);
        setMessages([]);
        setShowTyping(false);
    };

    useEffect(() => {
        lastMessageRef?.current?.scrollIntoView?.({behavior: isFirstScroll ? 'instant' : 'smooth'});
        if (lastMessageRef?.current && isFirstScroll) setIsFirstScroll(false);
    }, [messages, pendingMessage, isFirstScroll]);

    useEffect(() => {
        setIsFirstScroll(true);
        if (newSession && sessionId !== NEW_SESSION_ID) setNewSession(null);
        resetChat();
        if (sessionId !== NEW_SESSION_ID) refetch();
        textareaRef?.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId]);

    useEffect(() => {
        if (sessionId === NEW_SESSION_ID) return;
        const lastEvent = lastMessages?.events?.at(-1);
        if (!lastEvent) return;
        const offset = lastEvent?.offset;
        if (offset || offset === 0) setLastOffset(offset + 1);
        const correlationsMap = groupBy(lastMessages?.events || [], (item: EventInterface) => item?.correlation_id.split('.')[0]);
        const newMessages = lastMessages?.events?.filter(e => e.kind === 'message') || [];
        const withStatusMessages = newMessages.map(newMessage => ({...newMessage, serverStatus: correlationsMap?.[newMessage.correlation_id.split('.')[0]]?.at(-1)?.data?.status}));
        if (pendingMessage.serverStatus !== 'pending' && pendingMessage.data.message) setPendingMessage(emptyPendingMessage);
        setMessages(messages => {
            const last = messages.at(-1);
           if (last?.source === 'end_user' && correlationsMap?.[last?.correlation_id]) last.serverStatus = correlationsMap[last.correlation_id].at(-1)?.data?.status || last.serverStatus;
           return [...messages, ...withStatusMessages] as EventInterface[];
        });

        const lastEventStatus = lastEvent?.data?.status;

        setShowTyping(lastEventStatus === 'typing');
        refetch();
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [lastMessages]);

    const createSession = async (): Promise<{session: SessionInterface} | undefined> => {
        if (!newSession) return;
        const {end_user_id, title} = newSession;
        return postData('sessions?allow_greeting=true', {end_user_id, agent_id: agentId, title} as object)
            .then((res: {session: SessionInterface}) => {
                if (newSession) {
                    setSessionId(res.session.id);
                    setNewSession(null);
                }
                setSessions(sessions => [...sessions, res.session]);
                return res;
            }).catch(() => {
                toast.error('Something went wrong');
                return undefined;
            });
     };

    const postMessage = async (content: string): Promise<void> => {
        setPendingMessage(pendingMessage => ({...pendingMessage, data: {message: content}}));
        setMessage('');
        const eventSession = newSession ? (await createSession())?.session?.id : sessionId;
        postData(`sessions/${eventSession}/events`, { kind: 'message', content, source: 'end_user' }).then(() => {
            setPendingMessage(pendingMessage => ({...pendingMessage, serverStatus: 'accepted'}));
            refetch();
        }).catch(() => toast.error('Something went wrong'));
    };

    const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitButtonRef?.current?.click();
        }
    };

    const isSameDay = (dateA: string | Date, dateB: string | Date): boolean => {
        if (!dateA) return false;
        return new Date(dateA).toLocaleDateString() === new Date(dateB).toLocaleDateString();
    };

    const visibleMessages = sessionId !== NEW_SESSION_ID && pendingMessage?.data?.message ? [...messages, pendingMessage] : messages;

    return (
        <>
        <div className='h-full w-full flex flex-col'>
            <div className="flex flex-col items-center h-full mx-auto w-full flex-1 overflow-auto">
                <div className="messages overflow-auto flex-1 flex flex-col w-full mb-4" aria-live="polite" role="log" aria-label="Chat messages">
                    {ErrorTemplate && <ErrorTemplate />}
                    {visibleMessages.map((event, i) => (
                        <React.Fragment key={i}>
                            {!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc) &&
                            <DateHeader date={event.creation_utc} isFirst={!i}/>}
                            <div ref={lastMessageRef} className="flex flex-col">
                                <Message event={event} isContinual={event.source === visibleMessages[i + 1]?.source}/>
                            </div>
                        </React.Fragment>
                    ))}
                    {showTyping && 
                    <div className='animate-fade-in flex mb-1 justify-between mt-[44.33px]'>
                        <Spacer/>
                        <div className='flex items-center max-w-[1200px] flex-1'>
                            <img src="parlant-bubble-muted.svg" alt="" height={36} width={36} className='me-[8px]'/>
                            <p className='font-medium text-[#A9AFB7] text-[11px] font-inter'>Typing...</p>
                        </div>
                        <Spacer/>
                    </div>}
                </div>
                <div className='w-full flex justify-between'>
                    <Spacer/>
                    <div className="group border flex-1 border-muted border-solid rounded-full flex flex-row justify-center items-center bg-white p-[0.9rem] ps-[24px] pe-0 h-[48.67px] max-w-[1200px] relative mb-[26px] hover:bg-main">
                        <img src="icons/edit.svg" alt="" className="me-[8px] h-[14px] w-[14px]"/>
                        <Textarea role="textbox"
                            ref={textareaRef}
                            placeholder="Message..."
                            value={message}
                            onKeyDown={onKeyDown}
                            onChange={(e) => setMessage(e.target.value)}
                            rows={1}
                            className="box-shadow-none resize-none border-none h-full rounded-none min-h-[unset] p-0 whitespace-nowrap no-scrollbar font-inter font-light text-[16px] leading-[18px] bg-white group-hover:bg-main"/>
                        <Button variant='ghost'
                            data-testid="submit-button"
                            className="max-w-[60px] rounded-full hover:bg-white"
                            ref={submitButtonRef}
                            disabled={!message?.trim() || !agentId}
                            onClick={() => postMessage(message)}>
                            <img src="icons/send.svg" alt="Send" height={19.64} width={21.52} className='h-10'/>
                        </Button>
                    </div>
                    <Spacer/>
                </div>
            </div>
        </div>
        </>
    );
}