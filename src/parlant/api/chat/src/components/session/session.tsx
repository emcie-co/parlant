import { Dispatch, ReactElement, SetStateAction, useEffect, useRef, useState } from 'react';
import { Input } from '../ui/input';
import Tooltip from '../ui/custom/tooltip';
import { Button } from '../ui/button';
import { deleteData, patchData } from '@/utils/api';
import { toast } from 'sonner';
import { useSession } from '../chatbot/chatbot';
import { SessionInterface } from '@/utils/interfaces';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '../ui/dropdown-menu';
import { getDateStr, getTimeStr } from '@/utils/date';
import styles from './session.module.scss';
import AgentAvatar from '../agent-avatar/agent-avatar';
import { NEW_SESSION_ID } from '../chat-header/chat-header';
import { spaceClick } from '@/utils/methods';
import { twJoin } from 'tailwind-merge';

interface Props {
    session: SessionInterface;
    disabled?: boolean;
    isSelected?: boolean;
    editingTitle?: string | null;
    setEditingTitle?: Dispatch<SetStateAction<string | null>>;
    refetch?: () => void;
    tabIndex?: number;
}

export const DeleteDialog = ({session, closeDialog, deleteClicked}: {session: SessionInterface, closeDialog: () => void, deleteClicked: (e: React.MouseEvent) => Promise<void> | undefined}) => (
    <div data-testid="deleteDialogContent">
        <Session session={session} disabled/>
        <div className='h-[80px] flex items-center justify-end pe-[18px]'>
            <Button data-testid="cancel-delete" onClick={closeDialog} className='hover:bg-[#EBE9F5] h-[46px] w-[96px] text-black bg-[#F2F0FC] rounded-[6px] py-[12px] px-[24px] me-[10px] text-[16px] font-normal'>Cancel</Button>
            <Button data-testid="gradient-button" onClick={deleteClicked} className='h-[46px] w-[161px] bg-[#213547] hover:bg-black rounded-[6px] py-[10px] px-[29.5px] text-[15px] font-medium'>Delete Session</Button>
        </div>
    </div>
);

export default function Session({session, isSelected, refetch, editingTitle, setEditingTitle, tabIndex, disabled}: Props): ReactElement {
    const sessionNameRef = useRef<HTMLInputElement>(null);
    const {setSessionId, setAgentId, setNewSession, agents, customers, setSessions, openDialog, closeDialog} = useSession();
    const [agentsMap, setAgentsMap] = useState(new Map());
    const [customerMap, setCustomerMap] = useState(new Map());

    useEffect(() => {
        if (!isSelected) return;
        if (session.id === NEW_SESSION_ID && !session.agent_id) setAgentId(null);
        else setAgentId(session.agent_id);
    }, [isSelected, setAgentId, session.id, session.agent_id, session.title]);

    useEffect(() => {
        if (agents) setAgentsMap(new Map(agents.map(agent => [agent.id, agent])));
    }, [agents]);

    useEffect(() => {
        if (customers) setCustomerMap(new Map(customers.map(customer => [customer.id, customer])));
    }, [customers]);

    const deleteSession = async (e: React.MouseEvent) => {
        e.stopPropagation();
        
        const deleteClicked = (e: React.MouseEvent) => {
            closeDialog();
            e.stopPropagation();
            if (session.id === NEW_SESSION_ID) {
                setNewSession(null);
                setSessionId(null);
                setAgentId(null);
                return;
            }
            return deleteData(`sessions/${session.id}`).then(() => {
                setSessions(sessions => sessions.filter(s => s.id !== session.id));
                if (isSelected) {
                    setSessionId(null);
                    document.title = 'Parlant';
                }
                toast.success(`Session "${session.title}" deleted successfully`, {closeButton: true});
            }).catch(() => {
                toast.error('Something went wrong');
            });
        };

        openDialog('Delete Session', <DeleteDialog closeDialog={closeDialog} deleteClicked={deleteClicked} session={session}/>, {height: '230px', width: '480px'});
    };

    const editTitle = async (e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingTitle?.(session.id);
        setTimeout(() => sessionNameRef?.current?.select(), 0);
    };

    const saveTitleChange = (e: React.MouseEvent | React.KeyboardEvent) => {
        e.stopPropagation();
        const title = sessionNameRef?.current?.value?.trim();
        if (title) {
            if (session.id === NEW_SESSION_ID) {
                setEditingTitle?.(null);
                setNewSession(session => session ? {...session, title} : session);
                toast.success('title changed successfully', {closeButton: true});
                return;
            }
            patchData(`sessions/${session.id}`, {title})
            .then(() => {
                setEditingTitle?.(null);
                refetch?.();
                toast.success('title changed successfully', {closeButton: true});
            }).catch(() => {
                toast.error('Something went wrong');
            });
        }
    };

    const cancel = (e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingTitle?.(null);
    };

    const onInputKeyUp = (e: React.KeyboardEvent) =>{
        if (e.key === 'Enter') saveTitleChange(e);
    };

    const sessionActions = [
        {title: 'rename', onClick: editTitle, imgPath: 'icons/rename.svg'},
        {title: 'delete', onClick: deleteSession, imgPath: 'icons/delete.svg'},
    ];
    const agent = agentsMap.get(session.agent_id);
    const customer = customerMap.get(session.customer_id);

    return (
        <div data-testid="session"
            role="button"
            tabIndex={tabIndex}
            onKeyDown={spaceClick}
            onClick={() => !disabled && !editingTitle && setSessionId(session.id)}
            key={session.id}
            className={'bg-white animate-fade-in text-[14px] font-ubuntu-sans justify-between font-medium border-b-[0.6px] border-b-solid border-muted cursor-pointer p-1 flex items-center ps-[8px] min-h-[80px] h-[80px] ml-0 mr-0 ' + (editingTitle === session.id ? (styles.editSession + ' !p-[4px_2px] ') : editingTitle ? ' opacity-[33%] ' : ' hover:bg-main ') + (isSelected && editingTitle !== session.id ? '!bg-[#FAF9FF]' : '') + (disabled ? ' pointer-events-none' : '')}>
            <div className="flex-1 whitespace-nowrap overflow-hidden max-w-[202px] ms-[16px] h-[39px]">
                {editingTitle !== session.id &&
                    <div className="overflow-hidden overflow-ellipsis flex items-center">
                        <div>
                            <AgentAvatar agent={agent || {id: '', name: 'N/A'}} customer={customer || {id: '', name: 'N/A'}}/>
                        </div>
                        <div className={twJoin(!agent && 'opacity-50')}>
                            {session.title}
                            <small className='text-[12px] text-[#A9A9A9] font-light flex gap-[6px]'>
                                {getDateStr(session.creation_utc)}
                                <img src="icons/dot-separator.svg" alt="" height={18} width={3}/>
                                {getTimeStr(session.creation_utc)}
                            </small>
                        </div>
                    </div>
                }
                {editingTitle === session.id && 
                    <div className='flex items-center ps-[6px]'>
                        <div>{agent && <AgentAvatar agent={agent}/>}</div>
                        <Input data-testid='sessionTitle'
                            ref={sessionNameRef}
                            onKeyUp={onInputKeyUp}
                            onClick={e => e.stopPropagation()}
                            autoFocus
                            defaultValue={session.title}
                            className="box-shadow-none border-none bg-[#F5F6F8] text-foreground h-fit p-1 ms-[6px]"/>
                    </div>
                }
            </div>
            <div className='h-[39px] flex items-center'>
                {!disabled && editingTitle !== session.id && 
                <DropdownMenu>
                    <DropdownMenuTrigger  disabled={!!editingTitle} data-testid="menu-button" tabIndex={-1} onClick={e => e.stopPropagation()}>
                        <div tabIndex={tabIndex} role='button' className='rounded-full me-[24px]' onClick={e => e.stopPropagation()}>
                            <img src='icons/more.svg' alt='more' height={14} width={14}/>
                        </div>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align='start'>
                        {sessionActions.map(sessionAction => (
                            <DropdownMenuItem tabIndex={0} key={sessionAction.title} onClick={sessionAction.onClick} className='gap-0 font-medium text-[14px] font-ubuntu-sans capitalize hover:!bg-[#FAF9FF]'>
                                <img data-testid={sessionAction.title} src={sessionAction.imgPath} height={16} width={18} className='me-[8px]' alt="" />
                                {sessionAction.title}
                            </DropdownMenuItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>}
                
                {editingTitle == session.id &&
                <div className='me-[18px]'>
                    <Tooltip value='Cancel'><Button data-testid="cancel" variant='ghost' className="w-[28px] h-[28px] p-[8px] rounded-full" onClick={cancel}><img src="icons/cancel.svg" alt="cancel" /></Button></Tooltip>
                    <Tooltip value='Save'><Button variant='ghost' className="w-[28px] h-[28px] p-[8px] rounded-full" onClick={saveTitleChange}><img src="icons/save.svg" alt="cancel" /></Button></Tooltip>
                </div>}
            </div>
        </div>
    );
}