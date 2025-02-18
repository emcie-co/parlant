import {createContext, lazy, ReactElement, Suspense, useEffect, useState} from 'react';
import ErrorBoundary from '../error-boundary/error-boundary';
import ChatHeader from '../chat-header/chat-header';
import {useDialog} from '@/hooks/useDialog';
import {Helmet} from 'react-helmet';
import {NEW_SESSION_ID} from '../agents-list/agent-list';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useAtom} from 'jotai';
import {dialogAtom, sessionAtom} from '@/store';
import SessionList from '../session-list/session-list';

export const SessionProvider = createContext({});

export default function Chatbot(): ReactElement {
	const SessionView = lazy(() => import('../session-view/session-view'));
	const [sessionName, setSessionName] = useState<string | null>('');
	const {openDialog, DialogComponent, closeDialog} = useDialog();
	const [session] = useAtom(sessionAtom);
	const [, setDialog] = useAtom(dialogAtom);

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
	}, []);

	return (
		<ErrorBoundary>
			<SessionProvider.Provider value={{}}>
				<Helmet defaultTitle={`${sessionName}`} />
				<div data-testid='chatbot' className='main bg-main h-screen flex flex-col'>
					<div className='hidden max-mobile:block'>
						<ChatHeader />
					</div>
					<div className='flex justify-between flex-1 w-full overflow-auto flex-row'>
						<div className='bg-white h-full pb-4 border-solid w-[332px] max-mobile:hidden z-[11] border-e'>
							<ChatHeader />
							<SessionList />
						</div>
						<div className='h-full w-[calc(100vw-332px)] max-w-[calc(100vw-332px)] max-[750px]:max-w-full max-[750px]:w-full '>
							{session?.id ? (
								<Suspense>
									<SessionView />
								</Suspense>
							) : (
								<HeaderWrapper />
							)}
						</div>
					</div>
				</div>
			</SessionProvider.Provider>
			<DialogComponent />
		</ErrorBoundary>
	);
}
