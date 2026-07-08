import {ReactElement, useCallback, useEffect, useRef, useState} from 'react';
import useFetch from '@/hooks/useFetch';
import Session from './session-list-item/session-list-item';
import {AgentInterface, SessionInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {agentAtom, agentsAtom, customerAtom, customersAtom, sessionAtom, sessionsAtom} from '@/store';
import {NEW_SESSION_ID} from '../agents-list/agent-list';
import {twJoin} from 'tailwind-merge';
import {BASE_URL} from '@/utils/api';

// How often to poll for changed sessions, and how often to do a full reconcile
// (a full fetch catches deletions, which a "changed since" delta can't surface).
const DELTA_POLL_INTERVAL = 3000;
const FULL_RECONCILE_INTERVAL = 60000;

const latestModifiedUtc = (sessions: SessionInterface[]): string | null => {
	let latest: string | null = null;
	for (const session of sessions) {
		const modified = session.modified_utc;
		if (modified && (!latest || Date.parse(modified) > Date.parse(latest))) latest = modified;
	}
	return latest;
};

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
	// High-water mark of the newest `modified_utc` we've seen, used to fetch only
	// sessions changed since the last poll.
	const watermarkRef = useRef<string | null>(null);
	// Coalesces the full reconcile when focus + visibilitychange fire together.
	const lastReconcileRef = useRef(0);

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
		if (data) {
			setSessions(data);
			watermarkRef.current = latestModifiedUtc(data);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [data]);

	// Merge changed sessions in by id (existing keep their place, new ones append),
	// and advance the watermark.
	const upsertSessions = useCallback(
		(changed: SessionInterface[]) => {
			if (!changed.length) return;
			setSessions((prev) => {
				const byId = new Map(prev.map((s) => [s.id, s]));
				for (const s of changed) byId.set(s.id, s);
				return [...byId.values()];
			});
			const newest = latestModifiedUtc(changed);
			if (newest && (!watermarkRef.current || Date.parse(newest) > Date.parse(watermarkRef.current))) {
				watermarkRef.current = newest;
			}
		},
		[setSessions],
	);

	// Full fetch — the only thing that catches deletions, so it runs on a slow timer
	// and on focus. Coalesced so the focus+visibility pair doesn't double-fetch.
	const reconcile = useCallback(async () => {
		const now = Date.now();
		if (now - lastReconcileRef.current < 1000) return;
		lastReconcileRef.current = now;
		try {
			const response = await fetch(`${BASE_URL}/sessions`);
			if (!response.ok) return;
			const all: SessionInterface[] = await response.json();
			setSessions(all);
			watermarkRef.current = latestModifiedUtc(all);
		} catch {
			// Transient network error — the next tick will retry.
		}
	}, [setSessions]);

	// Delta poll — fetch only sessions changed since the watermark.
	const pollDelta = useCallback(async () => {
		const watermark = watermarkRef.current;
		if (!watermark) {
			await reconcile();
			return;
		}
		try {
			const response = await fetch(
				`${BASE_URL}/sessions?min_modified_utc=${encodeURIComponent(watermark)}`,
			);
			if (!response.ok) return;
			const changed: SessionInterface[] = await response.json();
			upsertSessions(changed);
		} catch {
			// Transient network error — the next tick will retry.
		}
	}, [reconcile, upsertSessions]);

	useEffect(() => {
		let deltaTimer: ReturnType<typeof setInterval> | null = null;
		let reconcileTimer: ReturnType<typeof setInterval> | null = null;

		const start = () => {
			if (deltaTimer === null) deltaTimer = setInterval(pollDelta, DELTA_POLL_INTERVAL);
			if (reconcileTimer === null)
				reconcileTimer = setInterval(reconcile, FULL_RECONCILE_INTERVAL);
		};
		const stop = () => {
			if (deltaTimer !== null) clearInterval(deltaTimer);
			if (reconcileTimer !== null) clearInterval(reconcileTimer);
			deltaTimer = null;
			reconcileTimer = null;
		};

		const onVisibilityChange = () => {
			if (document.visibilityState === 'visible') {
				reconcile();
				start();
			} else {
				// Don't poll in the background.
				stop();
			}
		};
		const onFocus = () => reconcile();

		if (document.visibilityState === 'visible') start();
		document.addEventListener('visibilitychange', onVisibilityChange);
		window.addEventListener('focus', onFocus);

		return () => {
			stop();
			document.removeEventListener('visibilitychange', onVisibilityChange);
			window.removeEventListener('focus', onFocus);
		};
	}, [pollDelta, reconcile]);

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
