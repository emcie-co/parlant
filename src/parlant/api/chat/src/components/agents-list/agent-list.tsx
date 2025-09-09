import {AgentInterface, CustomerInterface, SessionInterface} from '@/utils/interfaces';
import {ReactNode, useEffect} from 'react';

import {spaceClick} from '@/utils/methods';
import {DialogDescription, DialogHeader, DialogTitle} from '../ui/dialog';
import clsx from 'clsx';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, customerAtom, customersAtom, dialogAtom, newSessionAtom, sessionAtom} from '@/store';
import Avatar from '../avatar/avatar';

/**
 * Constant identifier used for new session creation
 */
export const NEW_SESSION_ID = 'NEW_SESSION';

/**
 * Default session object template for creating new conversations
 */
const newSessionObj: SessionInterface = {
	customer_id: '',
	title: 'New Conversation',
	agent_id: '',
	creation_utc: new Date().toLocaleString(),
	id: NEW_SESSION_ID,
};

/**
 * Component that renders a dialog for selecting agents and customers.
 * Displays a list of available agents or customers based on current selection state.
 * Automatically selects the first agent if only one is available.
 * 
 * @returns React component that renders the agent/customer selection dialog
 */
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

	/**
	 * Handles agent selection and automatically proceeds to customer selection if only one customer exists
	 * 
	 * @param agent - The selected agent object
	 */
	const selectAgent = (agent: AgentInterface): void => {
		setAgent(agent);
		if (customers.length < 2) {
			selectCustomer(customers?.[0], agent);
		}
	};

	/**
	 * Handles customer selection, creates a new session, and closes the dialog
	 * 
	 * @param customer - The selected customer object
	 * @param currAgent - Optional current agent to use if no agent is set in state
	 */
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
