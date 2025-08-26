/* eslint-disable react-refresh/only-export-components */
import {createContext, ReactElement, useEffect, useState} from 'react';
import SessionList from '../session-list/session-list';
import ErrorBoundary from '../error-boundary/error-boundary';
import ChatHeader from '../chat-header/chat-header';
import {useDialog} from '@/hooks/useDialog';
import {Helmet} from 'react-helmet';
import AgentList, {NEW_SESSION_ID} from '../agents-list/agent-list';
import {useAtom} from 'jotai';
import {agentAtom, dialogAtom, sessionAtom, sessionsAtom} from '@/store';
import {twMerge} from 'tailwind-merge';
import SessionView from '../session-view/session-view';
import {spaceClick} from '@/utils/methods';

export const SessionProvider = createContext({});

const SessionsSection = () => {
	const [filterSessionVal, setFilterSessionVal] = useState('');
	return (
		<div className='bg-white [box-shadow:0px_0px_25px_0px_#0000000A] h-full rounded-[16px] overflow-hidden border-solid w-[352px] min-w-[352px] max-mobile:hidden z-[11] '>
			<ChatHeader setFilterSessionVal={setFilterSessionVal} filterSessionVal={filterSessionVal} />
			<SessionList filterSessionVal={filterSessionVal} />
		</div>
	);
};

export default function Chatbot(): ReactElement {
	// const SessionView = lazy(() => import('../session-view/session-view'));
	const [sessionName, setSessionName] = useState<string | null>('');
	const {openDialog, DialogComponent, closeDialog} = useDialog();
	const [showMessage, setShowMessage] = useState(false);
	const [sessions] = useAtom(sessionsAtom);
	const [session, setSession] = useAtom(sessionAtom);
	const [, setDialog] = useAtom(dialogAtom);
	const [filterSessionVal, setFilterSessionVal] = useState('');
	const [, setAgent] = useAtom(agentAtom);
	const [dialog] = useAtom(dialogAtom);

	useEffect(() => {
		if (sessions) {
			setShowMessage(!!sessions.length);
		}
		setTimeout(() => {
			setShowMessage(true);
		}, 500);
	}, [sessions]);

	useEffect(() => {
		if (session?.id) {
			if (session?.id === NEW_SESSION_ID) setSessionName('Parlant | New Session');
			else {
				const sessionTitle = session?.title;
				if (sessionTitle) setSessionName(`Parlant | ${sessionTitle}`);
			}
		} else setSessionName('Parlant');
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [session?.id]);

	useEffect(() => {
		setDialog({openDialog, closeDialog});
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	const createNewSession = () => {
		setSession(null);
		setAgent(null);
		dialog.openDialog('', <AgentList />, {height: '536px', width: '604px'});
	};

	return (
		<ErrorBoundary>
			<SessionProvider.Provider value={{}}>
				<Helmet defaultTitle={`${sessionName}`} />
				<div data-testid='chatbot' className={'main bg-green-light h-screen flex flex-col rounded-[16px]'}>
					<div className='hidden max-mobile:block rounded-[16px]'>
						<ChatHeader setFilterSessionVal={setFilterSessionVal} filterSessionVal={filterSessionVal} />
					</div>
					<div className={twMerge('flex bg-green-light flex-1 gap-[14px] w-full overflow-auto flex-row py-[14px] px-[14px]')}>
						<SessionsSection />
						{session?.id ? (
							<div className='h-full w-[calc(100vw-352px-40px)] bg-white rounded-[16px] max-w-[calc(100vw-352px-40px)] max-[800px]:max-w-full max-[800px]:w-full '>
								<SessionView />
							</div>
						) : (
							<div className='flex-1 flex flex-col gap-[27px] items-center justify-center'>
								<img className='pointer-events-none' src='select-session.svg' fetchPriority='high' alt='' />
								<p className='text-[#3C8C71] select-none font-light text-[18px] flex flex-col gap-[10px] items-center'>
									{showMessage && !sessions.length ? 'Start a session to begin chatting' : 'Select or start a session to begin chatting'}
									<div className='group'>
										<img src='buttons/new-session.svg' alt='add session' className='shadow-main cursor-pointer group-hover:hidden w-[76px]' tabIndex={1} role='button' onKeyDown={spaceClick} onClick={createNewSession} />
										<img src='buttons/new-session-hover.svg' alt='add session' className='shadow-main cursor-pointer hidden group-hover:block w-[76px]' tabIndex={1} role='button' onKeyDown={spaceClick} onClick={createNewSession} />
									</div>
								</p>
							</div>
						)}
					</div>
				</div>
			</SessionProvider.Provider>
			<DialogComponent />
		</ErrorBoundary>
	);
}
