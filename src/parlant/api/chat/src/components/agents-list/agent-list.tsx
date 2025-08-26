import {AgentInterface, CustomerInterface, SessionInterface} from '@/utils/interfaces';
import {ReactNode, useEffect} from 'react';

import {spaceClick} from '@/utils/methods';
import {DialogDescription, DialogHeader, DialogTitle} from '../ui/dialog';
import clsx from 'clsx';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, customerAtom, customersAtom, dialogAtom, newSessionAtom, sessionAtom} from '@/store';
import Avatar from '../avatar/avatar';

export const NEW_SESSION_ID = 'NEW_SESSION';

const newSessionObj: SessionInterface = {
	customer_id: '',
	title: 'New Conversation',
	agent_id: '',
	creation_utc: new Date().toLocaleString(),
	id: NEW_SESSION_ID,
};

const AgentList = (): ReactNode => {
	const [, setSession] = useAtom(sessionAtom);
	const [agent, setAgent] = useAtom(agentAtom);
	const [agents] = useAtom(agentsAtom);
	const [customers] = useAtom(customersAtom);
	const [, setCustomer] = useAtom(customerAtom);
	const [, setNewSession] = useAtom(newSessionAtom);
	const [dialog] = useAtom(dialogAtom);

	useEffect(() => {
		if (agents?.length && agents.length === 1) selectAgent(agents[0]);
	}, []);

	const selectAgent = (agent: AgentInterface): void => {
		setAgent(agent);
		if (customers.length < 2) {
			selectCustomer(customers?.[0], agent);
		}
	};

	const selectCustomer = (customer: CustomerInterface, currAgent?: AgentInterface) => {
		setAgent(agent || currAgent || null);
		setCustomer(customer);
		setNewSession({...newSessionObj, agent_id: agent?.id as string, customer_id: customer.id});
		setSession(newSessionObj);
		dialog.closeDialog();
	};

	return (
		<div className='h-full flex flex-col'>
			<DialogHeader>
				<DialogTitle>
					<div className='mb-[12px] mt-[24px] w-full flex justify-between items-center ps-[30px] pe-[20px]'>
						<DialogDescription className='text-[20px] font-semibold'>{agent ? 'Select a Customer' : 'Select an Agent'}</DialogDescription>
						<img role='button' tabIndex={0} onKeyDown={spaceClick} onClick={dialog.closeDialog} className='cursor-pointer rounded-full' src='icons/close.svg' alt='close' height={24} width={24} />
					</div>
				</DialogTitle>
			</DialogHeader>
			<div className='flex flex-col fixed-scroll overflow-auto relative flex-1'>
				{(agent ? customers : agents)?.map((entity) => (
					<div
						data-testid='agent'
						tabIndex={0}
						onKeyDown={spaceClick}
						role='button'
						onClick={() => (agent ? selectCustomer(entity) : selectAgent(entity))}
						key={entity.id}
						className={clsx('cursor-pointer hover:bg-[#FBFBFB] min-h-[78px] h-[78px] w-full border-b-[0.6px] border-b-solid border-b-[#EBECF0] flex items-center ps-[30px] pe-[20px]')}>
						<Avatar agent={entity} tooltip={false} />
						<div>
							<div className='text-[16px] font-medium'>{entity.id === 'guest' ? 'Guest' : entity.name}</div>
							<div className='text-[14px] font-light text-[#A9A9A9]'>(id={entity.id})</div>
						</div>
					</div>
				))}
			</div>
		</div>
	);
};

export default AgentList;
