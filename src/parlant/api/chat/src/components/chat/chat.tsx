import React, { ReactElement, useEffect, useRef, useState } from 'react';
import useFetch from '@/hooks/useFetch';
import { Textarea } from '../ui/textarea';
import { Button } from '../ui/button';
import { deleteData, postData } from '@/utils/api';
import { groupBy } from '@/utils/obj';
import Message from '../message/message';
import { useSession } from '../chatbot/chatbot';
import { EventInterface, SessionInterface } from '@/utils/interfaces';
import { getDateStr } from '@/utils/date';
import { Spacer } from '../ui/custom/spacer';
import { toast } from 'sonner';
import { NEW_SESSION_ID } from '../chat-header/chat-header';
import { useQuestionDialog } from '@/hooks/useQuestionDialog';
import { twMerge } from 'tailwind-merge';

const emptyPendingMessage: EventInterface = {
    kind: 'message',
    source: 'customer',
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
    const [isRegenerating, setIsRegenerating] = useState(false);
    const [isFirstScroll, setIsFirstScroll] = useState(true);
    const {openQuestionDialog, closeQuestionDialog} = useQuestionDialog();
    const [useContentFiltering] = useState(true);
    const [isMissingAgent, setIsMissingAgent] = useState<boolean | null>(null);

    const {sessionId, setSessionId, agentId, newSession, setNewSession, setSessions, agents} = useSession();
    const {data: lastMessages, refetch, ErrorTemplate} = useFetch<EventInterface[]>(
        `sessions/${sessionId}/events`,
        {min_offset: lastOffset},
        [],
        sessionId !== NEW_SESSION_ID,
        !!(sessionId && sessionId !== NEW_SESSION_ID)
    );

    useEffect(() => {
        if (agents && agentId) {
            setIsMissingAgent(!agents?.find(agent => agent.id === agentId));
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [agents, agentId]);

    const resetChat = () => {
        setMessage('');
        setLastOffset(0);
        setMessages([]);
        setShowTyping(false);
    };

    const regenerateMessageDialog = (index: number) => (sessionId: string, offset: number) => {
        const isLastMessage = index === messages.length - 1;
        if (isLastMessage) return regenerateMessage(index, sessionId, offset);

        const onApproved = () => {
            closeQuestionDialog();
            regenerateMessage(index, sessionId, offset);
        };

        const question = 'Regenerating this message would cause all of the following messages in the session to disappear.';
        openQuestionDialog('Are you sure?', question, [{text: 'Regenerate Anyway', onClick: onApproved, isMainAction: true}]);
    };

    const regenerateMessage = async (index: number, sessionId: string, offset: number) => {
        const prevAllMessages = messages;
        const prevLastOffset = lastOffset;

        setMessages(messages => messages.slice(0, index));
        setLastOffset(offset - 1);
        setIsRegenerating(true);
        const deleteSession = await deleteData(`sessions/${sessionId}/events?min_offset=${offset}`).catch((e) => ({error: e}));
        if (deleteSession?.error) {
            toast.error(deleteSession.error.message || deleteSession.error);
            setMessages(prevAllMessages);
            setLastOffset(prevLastOffset);
            return;
        }
        postData(`sessions/${sessionId}/events`, {kind: 'message', source: 'ai_agent'});
        refetch();
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
        const lastEvent = lastMessages?.at(-1);
        if (!lastEvent) return;
        const offset = lastEvent?.offset;
        if (offset || offset === 0) setLastOffset(offset + 1);
        const correlationsMap = groupBy(lastMessages || [], (item: EventInterface) => item?.correlation_id.split('.')[0]);
        const newMessages = lastMessages?.filter(e => e.kind === 'message') || [];
        const withStatusMessages = newMessages.map(newMessage => ({...newMessage, serverStatus: correlationsMap?.[newMessage.correlation_id.split('.')[0]]?.at(-1)?.data?.status}));
        if (newMessages.length && isRegenerating) setIsRegenerating(false);

        if (pendingMessage.serverStatus !== 'pending' && pendingMessage.data.message) setPendingMessage(emptyPendingMessage);
        setMessages(messages => {
            const last = messages.at(-1);
           if (last?.source === 'customer' && correlationsMap?.[last?.correlation_id]) last.serverStatus = correlationsMap[last.correlation_id].at(-1)?.data?.status || last.serverStatus;
           return [...messages, ...withStatusMessages] as EventInterface[];
        });

        const lastEventStatus = lastEvent?.data?.status;

        setShowTyping(lastEventStatus === 'typing');
        if (lastEventStatus === 'error') {
            if (isRegenerating) {
                setIsRegenerating(false);
                toast.error('Something went wrong');
            }
        }
        refetch();

    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [lastMessages]);

    const createSession = async (): Promise<SessionInterface | undefined> => {
        if (!newSession) return;
        const {customer_id, title} = newSession;
        return postData('sessions?allow_greeting=true', {customer_id, agent_id: agentId, title} as object)
            .then((res: SessionInterface) => {
                if (newSession) {
                    setSessionId(res.id);
                    setNewSession(null);
                }
                setSessions(sessions => [...sessions, res]);
                return res;
            }).catch(() => {
                toast.error('Something went wrong');
                return undefined;
            });
     };

    const postMessage = async (content: string): Promise<void> => {
        setPendingMessage(pendingMessage => ({...pendingMessage, data: {message: content}}));
        setMessage('');
        const eventSession = newSession ? (await createSession())?.id : sessionId;
        const useContentFilteringStatus = useContentFiltering ? 'auto' : 'none';
        postData(`sessions/${eventSession}/events?moderation=${useContentFilteringStatus}`, { kind: 'message', message: content, source: 'customer' }).then(() => {
            setPendingMessage(pendingMessage => ({...pendingMessage, serverStatus: 'accepted'}));
            refetch();
        }).catch(() => toast.error('Something went wrong'));
    };

    const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitButtonRef?.current?.click();
        } else if (e.key === 'Enter' && e.shiftKey) e.preventDefault();
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
                <div className="messages fixed-scroll flex-1 flex flex-col w-full mb-4" aria-live="polite" role="log" aria-label="Chat messages">
                    {ErrorTemplate && <ErrorTemplate />}
                    {visibleMessages.map((event, i) => (
                        <React.Fragment key={i}>
                            {!isSameDay(messages[i - 1]?.creation_utc, event.creation_utc) &&
                            <DateHeader date={event.creation_utc} isFirst={!i}/>}
                            <div ref={lastMessageRef} className="flex flex-col">
                                <Message isRegenerateHidden={!!isMissingAgent} event={event} isContinual={event.source === visibleMessages[i + 1]?.source} regenerateMessageFn={regenerateMessageDialog(i)}/>
                            </div>
                        </React.Fragment>
                    ))}
                    {(isRegenerating || showTyping) && (
                        <div className='animate-fade-in flex mb-1 justify-between mt-[44.33px]'>
                        <Spacer/>
                        <div className='flex items-center max-w-[1200px] flex-1'>
                            <img src="parlant-bubble-muted.svg" alt="" height={36} width={36} className='me-[8px]'/>
                            <p className='font-medium text-[#A9AFB7] text-[11px] font-inter'>{isRegenerating ? 'Regenerating...' : 'Typing...'}</p>
                        </div>
                        <Spacer/>
                    </div>
                    )}
                </div>
                <div className={twMerge('w-full flex justify-between', isMissingAgent && 'hidden')}>
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
                            disabled={!message?.trim() || !agentId || isRegenerating}
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
