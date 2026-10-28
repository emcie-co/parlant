import {ReactNode, useEffect} from 'react';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, customersAtom, dialogAtom, sessionAtom} from '@/store';
import AgentList from '../agents-list/agent-list';
import {AgentInterface} from '@/utils/interfaces';
import useFetch from '@/hooks/useFetch';
import {BASE_URL} from '@/utils/api';
import {spaceClick} from '@/utils/methods';

const TopNav = (): ReactNode => {
	const [, setSession] = useAtom(sessionAtom);
	const [, setAgent] = useAtom(agentAtom);
	const [dialog] = useAtom(dialogAtom);
	// In minimal mode the session list (which normally hydrates these atoms)
	// never mounts, so the agent picker depends on this fetch.
	const {data: agentsData} = useFetch<AgentInterface[]>('agents');
	const {data: customersData} = useFetch<AgentInterface[]>('customers');
	const [, setAgents] = useAtom(agentsAtom);
	const [, setCustomers] = useAtom(customersAtom);

	useEffect(() => {
		if (agentsData) setAgents(agentsData);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [agentsData]);

	useEffect(() => {
		if (customersData) setCustomers(customersData);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [customersData]);

	const createNewSession = () => {
		setSession(null);
		setAgent(null);
		dialog.openDialog('', <AgentList />, {height: '536px', width: '604px'});
	};

	return (
		<div className='h-[70px] min-h-[70px] w-full bg-white rounded-[16px] flex items-center justify-between px-[24px] [box-shadow:0px_0px_25px_0px_#0000000A] z-[11]'>
			<div className='flex items-center gap-[10px]'>
				<a href='https://www.parlant.io' target='_blank' rel='noopener noreferrer' aria-label='Parlant website'>
					<img src={`${BASE_URL}/chat/app-logo.svg`} alt='Parlant' className='h-[30px]' />
				</a>
				<span className='text-[#151515] font-medium text-[16px] select-none'>Playground</span>
			</div>
			<a
				href='https://dashboard.parlant.io?utm_source=playground&utm_medium=nav'
				target='_blank'
				rel='noopener noreferrer'
				className='text-[#3C8C71] hover:underline font-medium text-[15px]'>
				Develop in Dashboard ↗
			</a>
			<button
				onClick={createNewSession}
				onKeyDown={spaceClick}
				tabIndex={0}
				className='bg-[#3C8C71] hover:bg-[#2f7a60] text-white rounded-[8px] px-[16px] h-[38px] text-[14px] font-medium cursor-pointer'>
				New Session
			</button>
		</div>
	);
};

export default TopNav;
