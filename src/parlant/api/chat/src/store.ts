import {atom} from 'jotai';
import {AgentInterface, CustomerInterface, EventInterface, SessionInterface} from './utils/interfaces';
import {ReactNode} from 'react';
import {Dimensions} from './hooks/useDialog';

export const emptyPendingMessage: () => EventInterface = () => ({
	kind: 'message',
	source: 'customer',
	creation_utc: new Date(),
	serverStatus: 'pending',
	offset: 0,
	correlation_id: '',
	data: {
		message: '',
	},
});

const getLogs = () => {
	try {
		return JSON.parse(localStorage.logs || '{}');
	} catch (e) {
		console.error(e);
		localStorage.removeItem('logs');
		return {};
	}
};

export const haveLogsAtom = atom(getLogs());
export const agentsAtom = atom<AgentInterface[]>([]);
export const customersAtom = atom<CustomerInterface[]>([]);
export const customerAtom = atom<CustomerInterface | null>(null);
export const sessionAtom = atom<SessionInterface | null>(null);
export const agentAtom = atom<AgentInterface | null>(null);
export const newSessionAtom = atom<SessionInterface | null>(null);
export const sessionsAtom = atom<SessionInterface[]>([]);
export const viewingMessageDetailsAtom = atom<EventInterface | null>(null);
export const pendingMessageAtom = atom<EventInterface>(emptyPendingMessage());
export const dialogAtom = atom<{openDialog: (title: string | null, content: ReactNode, dimensions: Dimensions, dialogClosed?: () => void) => void; closeDialog: () => void}>({closeDialog: () => null, openDialog: () => null});
