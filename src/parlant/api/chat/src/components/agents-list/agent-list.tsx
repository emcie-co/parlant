import { SessionInterface } from '@/utils/interfaces';
import { ReactNode, useState } from 'react';
import { useSession } from '../chatbot/chatbot';

import AgentAvatar from '../agent-avatar/agent-avatar';
import { spaceClick } from '@/utils/methods';
import { DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';

export const NEW_SESSION_ID = 'NEW_SESSION';

const newSessionObj: SessionInterface = {
    customer_id: '',
    title: 'New Conversation',
    agent_id: '',
    creation_utc: new Date().toLocaleString(),
    id: NEW_SESSION_ID
};


const AgentList = (): ReactNode => {
    const {setAgentId, closeDialog, agents, setSessionId, setNewSession, customers} = useSession();
    const [agent, setAgent] = useState('');

    const selectAgent = (agentId: string): void => {
        setAgent(agentId);
    };

    const selectCustomer = (customerId: string) => {
        setAgentId(agent);
        setNewSession({...newSessionObj, agent_id: agent, customer_id: customerId});
        setSessionId(newSessionObj.id);
        closeDialog();
    };

    return (
        <div className='h-full flex flex-col'>
            <DialogHeader>
                <DialogTitle>
                    <div className='h-[68px] w-full flex justify-between items-center ps-[30px] pe-[20px] border-b-[#EBECF0] border-b-[0.6px]'>
                        <DialogDescription className='text-[16px] font-bold'>{agent ? 'Select a Customer' : 'Select an Agent'}</DialogDescription>
                        <img role='button' tabIndex={0} onKeyDown={spaceClick} onClick={closeDialog} className='cursor-pointer rounded-full hover:bg-[#F5F6F8] p-[10px]' src="icons/close.svg" alt="close" height={30} width={30}/>
                    </div>
                </DialogTitle>
            </DialogHeader>
            <div className='flex flex-col overflow-auto relative flex-1'>
                {(agent ? customers : agents)?.map(entity => (
                    <div data-testid="agent" tabIndex={0} onKeyDown={spaceClick} role='button' onClick={() => agent ? selectCustomer(entity.id) : selectAgent(entity.id)} key={entity.id} className='cursor-pointer hover:bg-[#FBFBFB] min-h-[78px] h-[78px] w-full border-b-[0.6px] border-b-solid border-b-[#EBECF0] flex items-center ps-[30px] pe-[20px]'>
                        <AgentAvatar agent={entity} tooltip={false}/>
                        <div>
                            <div className='text-[16px] font-medium'>{entity.name}</div>
                            <div className='text-[14px] font-light text-[#A9A9A9]'>(id={entity.id})</div>
                        </div>
                    </div>
                ))}
            </div>
            {agent && 
            <div className='flex justify-end'>
                <Button variant='ghost' onClick={() => selectCustomer('')}>
                    Continue with a guest Customer
                </Button>
            </div>}
        </div>
    );
};

export default AgentList;