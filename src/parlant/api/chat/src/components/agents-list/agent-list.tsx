import { SessionInterface } from '@/utils/interfaces';
import { ReactNode } from 'react';
import { useSession } from '../chatbot/chatbot';

import AgentAvatar from '../agent-avatar/agent-avatar';
import { spaceClick } from '@/utils/methods';

export const NEW_SESSION_ID = 'NEW_SESSION';

const newSessionObj: SessionInterface = {
    end_user_id: '',
    title: 'New Conversation',
    agent_id: '',
    creation_utc: new Date().toLocaleString(),
    id: NEW_SESSION_ID
};

const AgentList = (): ReactNode => {
    const {setAgentId, closeDialog, agents, setSessionId, setNewSession} = useSession();

    const selectAgent = (agentId: string): void => {
        setAgentId(agentId);
        setNewSession(newSessionObj);
        setSessionId(newSessionObj.id);
        closeDialog();
    };

    return (
        <div className='flex flex-col overflow-auto'>
            {agents?.map(agent => (
                <div data-testid="agent" tabIndex={0} onKeyDown={spaceClick} role='button' onClick={() => selectAgent(agent.id)} key={agent.id} className='cursor-pointer hover:bg-[#FBFBFB] min-h-[78px] h-[78px] w-full border-b-[0.6px] border-b-solid border-b-[#EBECF0] flex items-center ps-[30px] pe-[20px]'>
                    <AgentAvatar agent={agent} tooltip={false}/>
                    <div>
                        <div className='text-[16px] font-medium'>{agent.name}</div>
                        <div className='text-[14px] font-light text-[#A9A9A9]'>(id={agent.id})</div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default AgentList;