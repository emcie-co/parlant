import {ReactElement, useEffect, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import Session from './session-list-item/session-list-item';
import {AgentInterface, SessionInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, customerAtom, customersAtom, sessionAtom, sessionsAtom} from '@/store';
import {NEW_SESSION_ID} from '../agents-list/agent-list';
import {twJoin} from 'tailwind-merge';

export default function SessionList({filterSessionVal}: {filterSessionVal: string}): ReactElement {
	const [editingTitle, setEditingTitle] = useState<string | null>(null);
	const [session] = useAtom(sessionAtom);
	const {data, ErrorTemplate, loading, refetch} = useFetch<SessionInterface[]>('sessions');
	const {data: agentsData} = useFetch<AgentInterface[]>('agents');
	const {data: customersData} = useFetch<AgentInterface[]>('customers');
	const [, setAgents] = useAtom(agentsAtom);
	const [, setCustomers] = useAtom(customersAtom);
	const [agent] = useAtom(agentAtom);
	const [customer] = useAtom(customerAtom);
	const [sessions, setSessions] = useAtom(sessionsAtom);
	const [filteredSessions, setFilteredSessions] = useState(sessions);

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

	useEffect(() => {
		if (!filterSessionVal?.trim()) setFilteredSessions(sessions);
		else {
			setFilteredSessions(sessions.filter((session) => session.title?.toLowerCase()?.includes(filterSessionVal?.toLowerCase()) || session.id?.toLowerCase()?.includes(filterSessionVal?.toLowerCase())));
		}
	}, [filterSessionVal, sessions]);

	return (
		<div className={twJoin('flex flex-col items-center h-[calc(100%-68px)] border-e')}>
			<div data-testid='sessions' className='bg-white px-[12px] border-b-[12px] border-white flex-1 fixed-scroll justify-center w-[352px] overflow-auto rounded-es-[16px] rounded-ee-[16px]'>
				{loading && !sessions?.length && <div>loading...</div>}
				{session?.id === NEW_SESSION_ID && <Session className='opacity-50' data-testid='session' isSelected={true} session={{...session, agent_id: agent?.id || '', customer_id: customer?.id || ''}} key={NEW_SESSION_ID} />}
				{filteredSessions.toReversed().map((s, i) => (
					<Session data-testid='session' tabIndex={sessions.length - i} editingTitle={editingTitle} setEditingTitle={setEditingTitle} isSelected={s.id === session?.id} refetch={refetch} session={s} key={s.id} />
				))}
				{ErrorTemplate && <ErrorTemplate />}
			</div>
		</div>
	);
}
