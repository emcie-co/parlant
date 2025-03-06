import {ReactElement, useEffect, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import Session from './session-list-item/session-list-item';
import {AgentInterface, SessionInterface} from '@/utils/interfaces';
import VirtualScroll from '../virtual-scroll/virtual-scroll';
import {useAtom} from 'jotai';
import {agentsAtom, customersAtom, sessionAtom, sessionsAtom} from '@/store';

export default function SessionList(): ReactElement {
	const [editingTitle, setEditingTitle] = useState<string | null>(null);
	const [session] = useAtom(sessionAtom);
	const {data, ErrorTemplate, loading, refetch} = useFetch<SessionInterface[]>('sessions');
	const {data: agentsData} = useFetch<AgentInterface[]>('agents');
	const {data: customersData} = useFetch<AgentInterface[]>('customers');
	const [, setAgents] = useAtom(agentsAtom);
	const [, setCustomers] = useAtom(customersAtom);
	const [sessions, setSessions] = useAtom(sessionsAtom);

	useEffect(() => {
		if (agentsData) {
			setAgents(agentsData);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [agentsData]);

	useEffect(() => {
		if (customersData) {
			setCustomers(customersData);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [customersData]);

	useEffect(() => {
		if (data) setSessions(data);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [data]);

	return (
		<div className='flex flex-col items-center h-[calc(100%-70px)] border-e'>
			<div data-testid='sessions' className='bg-white flex-1 justify-center w-[332px] overflow-auto'>
				{loading && !sessions?.length && <div>loading...</div>}
				<VirtualScroll height='80px' className='flex flex-col-reverse'>
					{sessions.map((s, i) => (
						<Session data-testid='session' tabIndex={sessions.length - i} editingTitle={editingTitle} setEditingTitle={setEditingTitle} isSelected={s.id === session?.id} refetch={refetch} session={s} key={s.id} />
					))}
				</VirtualScroll>
				{ErrorTemplate && <ErrorTemplate />}
			</div>
		</div>
	);
}
